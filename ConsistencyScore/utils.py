import torch
import random
import numpy as np
import scipy.sparse as sp
import copy
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def select_candidate_nodes(data, num_candidates, seed=None):
    """Select random candidate nodes from training set."""
    device = data.x.device
    if seed is not None:
        torch.manual_seed(seed)
    
    train_indices = torch.where(data.train_mask)[0]
    perm = torch.randperm(len(train_indices), device=device)[:num_candidates]
    return train_indices[perm].to(device)

def prepare_dropped_data(data, candidate_nodes):
    """
    Prepare a new data object without candidate nodes.
    These nodes are used for memorization analysis.
    """
    device = data.x.device
    
    # Create node mask (True for nodes to keep) - on CPU initially
    node_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    node_mask[candidate_nodes.cpu()] = False
    
    # Get remaining nodes
    remaining_nodes = torch.where(node_mask)[0]
    
    # Create new data object without candidate nodes
    new_data = copy.deepcopy(data)
    
    # Move remaining_nodes to correct device for indexing
    remaining_nodes = remaining_nodes.to(device)
    
    # Update node features and labels
    new_data.x = data.x[remaining_nodes]
    new_data.y = data.y[remaining_nodes]
    
    # Update masks
    new_data.train_mask = data.train_mask[remaining_nodes]
    new_data.val_mask = data.val_mask[remaining_nodes]
    new_data.test_mask = data.test_mask[remaining_nodes]
    
    # Process edge_index on CPU then move back to device
    edge_index = data.edge_index.cpu()
    node_mask = node_mask.to(torch.bool)  # Ensure boolean type
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    new_edge_index = edge_index[:, edge_mask].to(device)
    
    # Remap node indices in edge_index
    node_idx_map = torch.full((data.num_nodes,), -1, dtype=torch.long, device=device)
    node_idx_map[remaining_nodes] = torch.arange(len(remaining_nodes), device=device)
    new_data.edge_index = node_idx_map[new_edge_index]
    
    return new_data, node_idx_map

def compute_memorization_score(model_f, model_g, data, candidate_nodes, num_passes=1):
    """
    Compute memorization score for candidate nodes by comparing predictions
    of two models - one trained with candidates (f) and one without (g).
    """
    model_f.eval()
    model_g.eval()
    scores = []
    
    with torch.no_grad():
        for _ in range(num_passes):
            out_f = torch.softmax(model_f(data.x, data.edge_index), dim=1)
            out_g = torch.softmax(model_g(data.x, data.edge_index), dim=1)
            
            for node_idx in candidate_nodes:
                true_label = data.y[node_idx].item()
                prob_f = out_f[node_idx, true_label].item()
                prob_g = out_g[node_idx, true_label].item()
                mem_score = prob_f - prob_g
                scores.append({
                    'node_idx': node_idx.item(),
                    'true_label': true_label,
                    'pred_f': out_f[node_idx].argmax().item(),
                    'pred_g': out_g[node_idx].argmax().item(),
                    'conf_f': prob_f,
                    'conf_g': prob_g,
                    'mem_score': mem_score
                })
    
    return pd.DataFrame(scores)

def save_memorization_results(results_df, dataset_name, model_name, seed):
    """Save memorization results to CSV file."""
    os.makedirs('results', exist_ok=True)
    filename = f'results/memorization_{dataset_name}_{model_name}_seed{seed}.csv'
    results_df.to_csv(filename, index=False)
    return filename

def aug_random_edge(input_adj, drop_rate=0.0):
    """
    Randomly drop edges from the graph with given drop rate.
    Returns original adjacency if drop_rate is 0.
    """
    if drop_rate <= 0:
        return input_adj
        
    if sp.issparse(input_adj):
        adj_dense = input_adj.todense()
    else:
        adj_dense = input_adj
    
    rows, cols = np.triu_indices(adj_dense.shape[0], k=1)
    edge_indices = [(r, c) for r, c in zip(rows, cols) if adj_dense[r, c] != 0]
    
    num_edges = len(edge_indices)
    num_drop = int(num_edges * drop_rate)
    
    drop_indices = random.sample(edge_indices, num_drop)
    
    new_adj = np.array(adj_dense)
    for i, j in drop_indices:
        new_adj[i, j] = 0
        new_adj[j, i] = 0
    
    return sp.csr_matrix(new_adj)

def plot_memorization_scores(results_df, dataset_name, model_name, seed, save_dir='plots'):
    """Create and save visualization plots for memorization scores."""
    os.makedirs(save_dir, exist_ok=True)
    #plt.style.use('seaborn-darkgrid')
    
    # Set font sizes
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    
    # Convert data types
    results_df = results_df.copy()
    results_df['mem_score'] = pd.to_numeric(results_df['mem_score'])
    results_df['conf_f'] = pd.to_numeric(results_df['conf_f'])
    results_df['conf_g'] = pd.to_numeric(results_df['conf_g'])
    results_df['node_idx'] = pd.to_numeric(results_df['node_idx'])
    results_df['true_label'] = pd.to_numeric(results_df['true_label'])
    results_df['pred_f'] = pd.to_numeric(results_df['pred_f'])
    results_df['pred_g'] = pd.to_numeric(results_df['pred_g'])
    
    fig = plt.figure(figsize=(20, 16))
    
    # Use a vibrant color palette
    colors = sns.color_palette("husl", n_colors=len(results_df['true_label'].unique()))
    
    # 1. Distribution of memorization scores
    plt.subplot(2, 2, 1)
    sns.histplot(data=results_df['mem_score'].values, bins=30, 
                color='skyblue', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Memorization Scores', pad=20)
    plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2.5, label='Threshold (0.5)')
    plt.xlabel('Memorization Score')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-1, 1)
    
    # 2. Scatter plot of confidence scores (f vs g)
    plt.subplot(2, 2, 2)
    plt.scatter(results_df['conf_f'].values, results_df['conf_g'].values, 
               alpha=0.6, s=150, color='purple', edgecolor='white')  # Increased size, added edge color
    plt.plot([0, 1], [0, 1], 'r--', label='y=x', linewidth=2.5)
    plt.xlabel('Model f Confidence')
    plt.ylabel('Model g Confidence')
    plt.title('Confidence Comparison (f vs g)', pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # 3. Memorization score vs. Model f confidence
    plt.subplot(2, 2, 3)
    unique_labels = sorted(results_df['true_label'].unique())
    
    # If we have many classes, use a continuous colormap instead
    if len(unique_labels) > 10:
        scatter = plt.scatter(results_df['conf_f'].values, 
                            results_df['mem_score'].values,
                            alpha=0.7, s=150,
                            c=results_df['true_label'].values,
                            cmap='nipy_spectral',
                            edgecolor='white')
        plt.colorbar(scatter, label='Class Label')
    else:
        for i, label in enumerate(unique_labels):
            mask = results_df['true_label'] == label
            plt.scatter(results_df[mask]['conf_f'].values, 
                       results_df[mask]['mem_score'].values,
                       alpha=0.7, s=150,
                       label=f'Class {label}',
                       color=colors[i],
                       edgecolor='white')
        plt.legend(title='True Label', bbox_to_anchor=(1.05, 1), loc='upper left', 
                  title_fontsize=MEDIUM_SIZE)
    
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2.5, label='Threshold')
    plt.xlabel('Model f Confidence')
    plt.ylabel('Memorization Score')
    plt.title('Memorization Score vs. Model f Confidence', pad=20)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(-1, 1)
    
    # 4. Prediction agreement analysis
    plt.subplot(2, 2, 4)
    agreement = results_df['pred_f'] == results_df['pred_g']
    counts = [agreement.sum(), (~agreement).sum()]
    bars = plt.bar(['Same Prediction', 'Different Prediction'], counts,
                  color=['forestgreen', 'crimson'], alpha=0.7)
    plt.title('Prediction Agreement between Models', pad=20)
    plt.xlabel('Agreement Type')
    plt.ylabel('Count')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Add a background grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout and save with high DPI
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'memorization_{dataset_name}_{model_name}_seed{seed}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return save_path

def plot_aggregate_results(final_results, dataset_name, model_name, threshold=0.5, save_dir='plots'):
    """Create and save visualization for aggregated results across seeds."""
    os.makedirs(save_dir, exist_ok=True)
    #plt.style.use('seaborn-darkgrid')
    
    # Set font sizes
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    
    # Data preparation
    final_results = final_results.copy()
    if isinstance(final_results.index, pd.MultiIndex):
        final_results.reset_index(inplace=True)
    
    # Convert columns to numeric
    final_results['node_idx'] = pd.to_numeric(final_results['node_idx'])
    final_results[('mem_score', 'mean')] = pd.to_numeric(final_results[('mem_score', 'mean')])
    final_results[('mem_score', 'std')] = pd.to_numeric(final_results[('mem_score', 'std')])
    final_results[('conf_f', 'mean')] = pd.to_numeric(final_results[('conf_f', 'mean')])
    final_results[('conf_g', 'mean')] = pd.to_numeric(final_results[('conf_g', 'mean')])
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Distribution of mean memorization scores
    plt.subplot(2, 2, 1)
    sns.histplot(data=final_results[('mem_score', 'mean')].values, bins=30, 
                color='skyblue', alpha=0.7, edgecolor='black')
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2.5,
                label=f'Threshold ({threshold})')
    plt.title('Distribution of Mean Memorization Scores', pad=20)
    plt.xlabel('Mean Memorization Score')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-1, 1)
    
    # 2. Memorization score vs. std
    plt.subplot(2, 2, 2)
    plt.scatter(final_results[('mem_score', 'mean')].values, 
               final_results[('mem_score', 'std')].values, 
               alpha=0.7, s=150, color='purple',  # Increased size
               edgecolor='white')  # Added edge color
    plt.xlabel('Mean Memorization Score')
    plt.ylabel('Standard Deviation')
    plt.title('Score Stability Analysis', pad=20)
    plt.grid(True, alpha=0.3)
    plt.xlim(-1, 1)
    
    # 3. Mean confidence comparison
    plt.subplot(2, 2, 3)
    plt.scatter(final_results[('conf_f', 'mean')].values, 
               final_results[('conf_g', 'mean')].values, 
               alpha=0.7, s=150, color='darkblue',  # Increased size
               edgecolor='white')  # Added edge color
    plt.plot([0, 1], [0, 1], 'r--', label='y=x', linewidth=2.5)
    plt.xlabel('Mean Model f Confidence')
    plt.ylabel('Mean Model g Confidence')
    plt.title('Mean Confidence Comparison', pad=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    # 4. Top memorized nodes analysis
    plt.subplot(2, 2, 4)
    top_k = 10
    top_nodes = final_results.nlargest(top_k, ('mem_score', 'mean'))
    colors = sns.color_palette("husl", n_colors=top_k)
    bars = plt.bar(range(top_k), 
                  top_nodes[('mem_score', 'mean')].values,
                  tick_label=top_nodes['node_idx'].values,
                  color=colors, alpha=0.8)
    plt.xticks(rotation=45)
    plt.title(f'Top {top_k} Most Memorized Nodes', pad=20)
    plt.xlabel('Node Index')
    plt.ylabel('Mean Memorization Score')
    
    # Add value labels on top of bars with nice formatting
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Add a background grid
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout and save
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'memorization_{dataset_name}_{model_name}_aggregate.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return save_path