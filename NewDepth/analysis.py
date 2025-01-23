import numpy as np
import random
import torch
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from analysis import *
from avgentropybaseline import *
import torch.nn.functional as F
from scipy.stats import entropy
from torch_geometric.data import Data
import copy
import networkx as nx
from nodeli2 import *
from collections import defaultdict
from torch_geometric.utils import to_networkx
import json


def visualize_prediction_depth(depth_df, save_dir, split_idx, mask_name, model):
    """Create visualizations for prediction depth analysis with new depth definition."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Get layer names
    layer_names = get_layer_names(model)
    depth_to_name = {i: name for i, name in enumerate(layer_names)}
    
    # 1. Distribution of Prediction Depths
    plt.figure(figsize=(15, 6))
    depth_counts = depth_df['prediction_depth'].value_counts().sort_index()
    
    # Create bar plot
    bars = plt.bar(depth_counts.index, depth_counts.values, alpha=0.7)
    
    # Add percentage labels on top of bars
    total_nodes = len(depth_df)
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_nodes) * 100
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{percentage:.1f}%',
                ha='center', va='bottom')
    
    plt.xticks(range(len(layer_names)), [depth_to_name[i] for i in range(len(layer_names))], 
               rotation=45, ha='right')
    plt.title(f'Distribution of Prediction Depths ({mask_name} Set)\nNew Definition: Consistent Predictions', 
              fontsize=15)
    plt.xlabel('Layer where predictions stabilize')
    plt.ylabel('Number of nodes')
    plt.tight_layout()
    plt.savefig(save_dir / f'depth_distribution_{mask_name}_split_{split_idx}_{timestamp}.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    # 2. GNN Model Accuracy by Depth
    gnn_accuracy_by_depth = depth_df.groupby('prediction_depth').agg({
        'node_idx': 'count',
        'gnn_correct': 'mean'
    }).rename(columns={'node_idx': 'Number of Nodes', 'gnn_correct': 'GNN Model Accuracy'})

    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax2 = ax1.twinx()

    # Plot bar chart for number of nodes
    sns.barplot(x=gnn_accuracy_by_depth.index, y='Number of Nodes',
                data=gnn_accuracy_by_depth, alpha=0.3, ax=ax1, color='blue')
    
    # Plot line for GNN accuracy
    sns.lineplot(x=gnn_accuracy_by_depth.index, y='GNN Model Accuracy',
                data=gnn_accuracy_by_depth, ax=ax2, color='red', marker='o')

    # Add percentage labels for accuracy
    for i, acc in enumerate(gnn_accuracy_by_depth['GNN Model Accuracy']):
        ax2.text(i, acc, f'{acc*100:.1f}%', ha='center', va='bottom', color='red')

    ax1.set_xticks(range(len(layer_names)))
    ax1.set_xticklabels([depth_to_name[i] for i in range(len(layer_names))], rotation=45, ha='right')
    
    ax1.set_xlabel('Prediction Depth (Layer where predictions stabilize)', fontsize=12)
    ax1.set_ylabel('Number of Nodes', color='blue', fontsize=12)
    ax2.set_ylabel('GNN Model Accuracy', color='red', fontsize=12)
    plt.title(f'GNN Accuracy by Prediction Depth ({mask_name} Set)\nNew Definition: Consistent Predictions', 
              fontsize=15)
    plt.tight_layout()
    plt.savefig(save_dir / f'gnn_depth_accuracy_{mask_name}.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    # 3. Early vs Late Learners Analysis
    early_learners = depth_df[depth_df['prediction_depth'] <= 1]
    late_learners = depth_df[depth_df['prediction_depth'] >= 2]
    
    # Create summary statistics
    summary_stats = {
        'early_learners': {
            'count': len(early_learners),
            'percentage': len(early_learners) / len(depth_df) * 100,
            'accuracy': early_learners['gnn_correct'].mean() * 100
        },
        'late_learners': {
            'count': len(late_learners),
            'percentage': len(late_learners) / len(depth_df) * 100,
            'accuracy': late_learners['gnn_correct'].mean() * 100
        }
    }

    # Plot early vs late learners comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distribution plot
    categories = ['Early\n(≤ Layer 1)', 'Late\n(≥ Layer 2)']
    counts = [summary_stats['early_learners']['count'], 
             summary_stats['late_learners']['count']]
    percentages = [summary_stats['early_learners']['percentage'],
                  summary_stats['late_learners']['percentage']]
    
    bars = ax1.bar(categories, counts, alpha=0.7)
    ax1.set_title('Distribution of Early vs Late Learners')
    ax1.set_ylabel('Number of Nodes')
    
    # Add percentage labels
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{percentage:.1f}%',
                ha='center', va='bottom')

    # Accuracy comparison
    accuracies = [summary_stats['early_learners']['accuracy'],
                 summary_stats['late_learners']['accuracy']]
    
    bars = ax2.bar(categories, accuracies, alpha=0.7)
    ax2.set_title('Accuracy: Early vs Late Learners')
    ax2.set_ylabel('Accuracy (%)')
    
    # Add accuracy labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom')

    plt.suptitle(f'Early vs Late Learners Analysis ({mask_name} Set)\nNew Definition: Consistent Predictions',
                fontsize=15)
    plt.tight_layout()
    plt.savefig(save_dir / f'early_vs_late_learners_{mask_name}.png',
                bbox_inches='tight', dpi=300)
    plt.close()

    # Fix the DataFrame creation by ensuring all arrays have the same length
    depth_counts = depth_df['prediction_depth'].value_counts().sort_index()
    max_depth = len(layer_names) - 1
    
    # Create arrays of equal length by filling in missing depths with 0
    all_depths = np.arange(max_depth + 1)
    counts_array = np.zeros_like(all_depths, dtype=float)
    for depth, count in depth_counts.items():
        if depth <= max_depth:  # Only include valid depths
            counts_array[int(depth)] = count
    
    # Calculate accuracy for each depth
    gnn_accuracy = []
    for depth in all_depths:
        mask = depth_df['prediction_depth'] == depth
        if mask.any():
            accuracy = depth_df[mask]['gnn_correct'].mean()
        else:
            accuracy = 0.0  # or np.nan if you prefer
        gnn_accuracy.append(accuracy)
    
    # Now create DataFrame with equal-length arrays
    results_df = pd.DataFrame({
        'depth': all_depths,
        'node_count': counts_array,
        'percentage': (counts_array / len(depth_df)) * 100,
        'gnn_accuracy': gnn_accuracy
    })
    
    results_df.to_csv(save_dir / f'prediction_depth_analysis_{mask_name}_split_{split_idx}.csv')

    # Add verification calculation
    total_nodes = 0
    correct_nodes = 0
    
    for depth in gnn_accuracy_by_depth.index:
        nodes_at_depth = gnn_accuracy_by_depth.loc[depth, 'Number of Nodes']
        accuracy_at_depth = gnn_accuracy_by_depth.loc[depth, 'GNN Model Accuracy']
        
        total_nodes += nodes_at_depth
        correct_nodes += nodes_at_depth * accuracy_at_depth

    weighted_avg_accuracy = (correct_nodes / total_nodes) * 100 if total_nodes > 0 else 0
    final_model_accuracy = depth_df['gnn_correct'].mean() * 100

    # Print verification results
    print("\n" + "="*50)
    print(f"Accuracy Verification for {mask_name} set (Split {split_idx}):")
    print(f"Depth-wise breakdown:")
    for depth in gnn_accuracy_by_depth.index:
        nodes = gnn_accuracy_by_depth.loc[depth, 'Number of Nodes']
        acc = gnn_accuracy_by_depth.loc[depth, 'GNN Model Accuracy'] * 100
        print(f"Depth {depth}: {nodes:.0f} nodes with {acc:.2f}% accuracy")
    print("-"*50)
    print(f"Weighted Average from Depth Analysis: {weighted_avg_accuracy:.2f}%")
    print(f"Final Model Accuracy: {final_model_accuracy:.2f}%")
    print(f"Absolute Difference: {abs(weighted_avg_accuracy - final_model_accuracy):.4f}%")
    print("="*50 + "\n")

    return weighted_avg_accuracy, final_model_accuracy


def visualize_prediction_confidence_and_entropy(model, data, predictions, confidences, train_mask, test_mask, split_idx, true_labels, noise_level, save_dir):
    # Calculate KD retention (assuming this function is defined elsewhere)
    delta_entropy = kd_retention(model, data, noise_level)
    
    # Add timestamp to filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    def plot_set(mask, set_name):
        # Ensure all inputs are on CPU and in numpy format
        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
        
        # Filter predictions for the current set (train or test)
        set_predictions = predictions[mask_np]  # predictions is already numpy
        set_confidences = confidences[mask]     # confidences is still a tensor
        set_true_labels = true_labels[mask].cpu().numpy() if torch.is_tensor(true_labels) else true_labels[mask]
        set_delta_entropy = delta_entropy[mask_np]  # delta_entropy is numpy

        # Determine high delta entropy nodes (e.g., top 10%)
        high_entropy_threshold = np.percentile(set_delta_entropy, 90)
        high_entropy_mask = set_delta_entropy >= high_entropy_threshold

        # Get max confidence for each prediction
        max_confidences = torch.max(set_confidences, dim=1).values.detach().cpu().numpy()

        # Determine correctness of predictions
        correct_predictions = (set_predictions == set_true_labels)

        # Create a colormap for the two possible cases
        colormap = {
            True: 'green',  # Correct prediction
            False: 'red'    # Wrong prediction
        }
        colors = [colormap[cp] for cp in correct_predictions]

        # Calculate category counts
        category_counts = {
            'green': sum(1 for c in colors if c == 'green'),
            'red': sum(1 for c in colors if c == 'red')
        }
        
        # Calculate high entropy counts for each category
        high_entropy_counts = {
            'green': sum(1 for c, he in zip(colors, high_entropy_mask) if c == 'green' and he),
            'red': sum(1 for c, he in zip(colors, high_entropy_mask) if c == 'red' and he)
        }

        # Plotting
        fig, ax = plt.subplots(figsize=(15, 10))

        # Plot low entropy nodes
        low_entropy_scatter = ax.scatter(
            max_confidences[~high_entropy_mask],
            set_delta_entropy[~high_entropy_mask],
            c=[c for c, he in zip(colors, high_entropy_mask) if not he],
            alpha=0.6,
            marker='o'
        )

        # Plot high entropy nodes with a different marker
        high_entropy_scatter = ax.scatter(
            max_confidences[high_entropy_mask],
            set_delta_entropy[high_entropy_mask],
            c=[c for c, he in zip(colors, high_entropy_mask) if he],
            alpha=0.6,
            marker='*',
            s=100  # Larger size for visibility
        )

        ax.set_title(f'Confidence vs Delta Entropy Plot ({set_name} Set, Split {split_idx})')
        ax.set_xlabel('Model Confidence')
        ax.set_ylabel('Delta Entropy')

        # Move legend outside the plot
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label=f'Correct Predictions: {category_counts["green"]} (High Δ Entropy: {high_entropy_counts["green"]})', markerfacecolor='green', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label=f'Wrong Predictions: {category_counts["red"]} (High Δ Entropy: {high_entropy_counts["red"]})', markerfacecolor='red', markersize=10),
            plt.Line2D([0], [0], marker='*', color='w', label=f'High Δ Entropy: {sum(high_entropy_mask)}', markerfacecolor='gray', markersize=15),
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5))

        # Adjust layout to prevent legend overlap
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make room for legend
        
        filename_base = save_dir / f'baseline_confidence_vs_entropy_{set_name.lower()}_split_{split_idx}_{timestamp}'
        plt.savefig(f'{filename_base}.png', bbox_inches='tight', dpi=300)
        plt.close()

        return category_counts, high_entropy_counts

    # Plot for test set
    test_counts, test_high_entropy_counts = plot_set(test_mask, "Test")

    # Plot for training set
    train_counts, train_high_entropy_counts = plot_set(train_mask, "Training")

    return {
        'test': {'counts': test_counts, 'high_entropy_counts': test_high_entropy_counts},
        'train': {'counts': train_counts, 'high_entropy_counts': train_high_entropy_counts}
    }


def kd_retention(model, data: Data, noise_level: float):
    device = next(model.parameters()).device
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        out_teacher = model(data.x, data.edge_index)
        data_teacher = F.softmax(out_teacher, dim=-1).cpu().numpy()
        weight_t = np.array([entropy(dt) for dt in data_teacher])
        feats_noise = copy.deepcopy(data.x)
        feats_noise += torch.randn_like(feats_noise) * noise_level
        data_noise = Data(x=feats_noise, edge_index=data.edge_index).to(device)
    with torch.no_grad():
        out_noise = model(data_noise.x, data_noise.edge_index)
        out_noise = F.softmax(out_noise, dim=-1).cpu().numpy()
        weight_s = np.abs(np.array([entropy(on) for on in out_noise]) - weight_t)
        delta_entropy = weight_s / np.max(weight_s)
    return delta_entropy
    

def create_knn_probe(features, labels, n_neighbors=10):
    """Create and train a KNN probe."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(features, labels)
    return knn

def get_layer_representations(model, data, edge_index):
    """Get representations at each layer of the model."""
    model.eval()
    layer_representations = []
    x = data.x
    
    with torch.no_grad():
        for i, conv in enumerate(model.convs[:-1]):
            x = conv(x, edge_index)
            if hasattr(model, 'bns') and i < len(model.bns):
                x = model.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=model.dropout if hasattr(model, 'dropout') else 0.0, training=False)
            layer_representations.append(x.cpu().numpy())
        
        # Last layer
        x = model.convs[-1](x, edge_index)
        layer_representations.append(x.cpu().numpy())
    
    return layer_representations

def get_layer_names(model):
    """Get descriptive names for each layer."""
    layer_names = []
    
    # Add names for each GNN layer with their dimensions
    for i, layer in enumerate(model.convs):
        # Get input and output dimensions
        in_dim = layer.in_channels
        out_dim = layer.out_channels
        
        if i == 0:
            layer_name = f"Layer 1\n({in_dim}→{out_dim})"
        elif i == len(model.convs) - 1:
            layer_name = f"Layer {i+1}\n(Output: {in_dim}→{out_dim})"
        else:
            layer_name = f"Layer {i+1}\n({in_dim}→{out_dim})"
            
        layer_names.append(layer_name)
    
    return layer_names


# def calculate_local_informativeness(graph, labels, neighborhood_size=2):
#     """Calculate local informativeness measures for each node."""
#     print("\nStarting local informativeness calculation...")
    
#     num_nodes = len(labels)
#     print(f"Total nodes to process: {num_nodes}")
    
#     local_measures = {
#         'node': np.zeros(num_nodes),
#         'edge': np.zeros(num_nodes),
#         'homophily': np.zeros(num_nodes)
#     }
    
#     # Debug counters
#     processed = 0
#     skipped_small = 0
#     skipped_no_edges = 0
#     errors = 0
#     success = 0
    
#     for node in graph.nodes():
#         processed += 1
#         #if processed % 100 == 0:  # Print progress every 100 nodes
#          #   print(f"Processed {processed}/{num_nodes} nodes")
            
#         try:
#             # Get k-hop neighborhood
#             neighborhood = nx.ego_graph(graph, node, radius=neighborhood_size)
            
#             # Skip if neighborhood is too small
#             if len(neighborhood) < 2:
#                 skipped_small += 1
#                 continue
                
#             # Create mapping between original and local node indices
#             neighborhood_nodes = list(neighborhood.nodes())
#             node_mapping = {old: new for new, old in enumerate(neighborhood_nodes)}
#             reverse_mapping = {new: old for old, new in node_mapping.items()}
            
#             # Create new subgraph with remapped nodes
#             subgraph = nx.Graph()
#             subgraph.add_nodes_from(range(len(neighborhood_nodes)))
            
#             # Add edges with remapped indices
#             for u, v in neighborhood.edges():
#                 subgraph.add_edge(node_mapping[u], node_mapping[v])
            
#             # Create local labels array
#             local_labels = np.array([labels[reverse_mapping[i]] for i in range(len(neighborhood_nodes))])
            
#             # Skip if no edges
#             if len(subgraph.edges()) == 0:
#                 skipped_no_edges += 1
#                 continue
            
#             # Calculate local measures
#             node_info = li_node(subgraph, local_labels)
#             edge_info = li_edge(subgraph, local_labels)
#             homoph = h_edge(subgraph, local_labels)
            
#             local_measures['node'][node] = node_info
#             local_measures['edge'][node] = edge_info
#             local_measures['homophily'][node] = homoph
            
#             if node_info != 0 or edge_info != 0 or homoph != 0:
#                 success += 1
#                 #if success % 100 == 0:  # Print details every 100 successful calculations
#                 #    print(f"\nSuccessful calculation for node {node}:")
#                     # print(f"Node informativeness: {node_info}")
#                     # print(f"Edge informativeness: {edge_info}")
#                     # print(f"Homophily: {homoph}")
#                     # print(f"Neighborhood size: {len(neighborhood)}")
#                     # print(f"Number of edges: {len(subgraph.edges())}")
#                     # print(f"Unique labels: {np.unique(local_labels)}")
            
#         except Exception as e:
#             errors += 1
#             print(f"\nError processing node {node}:")
#             print(f"Error message: {str(e)}")
#             if 'neighborhood' in locals():
#                 print(f"Neighborhood size: {len(neighborhood)}")
#             if 'subgraph' in locals():
#                 print(f"Number of edges: {len(subgraph.edges())}")
#             continue
    
#     # Print final statistics
#     print("\n=== Local Informativeness Calculation Summary ===")
#     print(f"Total nodes processed: {processed}")
#     print(f"Successful calculations (non-zero): {success}")
#     print(f"Skipped (small neighborhood): {skipped_small}")
#     print(f"Skipped (no edges): {skipped_no_edges}")
#     print(f"Errors encountered: {errors}")
    
#     # for measure_name, values in local_measures.items():
#     #     non_zero = np.sum(values != 0)
#     #     mean_val = np.mean(values)
#     #     max_val = np.max(values)
#     #     min_val = np.min(values[values != 0]) if non_zero > 0 else "N/A"
        
#         # print(f"\n{measure_name.upper()} Statistics:")
#         # print(f"Non-zero values: {non_zero}/{num_nodes} ({(non_zero/num_nodes)*100:.2f}%)")
#         # print(f"Mean: {mean_val:.4f}")
#         # print(f"Max: {max_val:.4f}")
#         # print(f"Min (non-zero): {min_val}")
    
#     return local_measures

# def visualize_informativeness_depth_relationship(depth_df, save_dir, split_idx, mask_name):
#     """Create visualizations for the relationship between informativeness and prediction depth."""
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
    
#     # Create scatter plots for each informativeness measure
#     measures = {
#         'local_node_informativeness': 'Local Node Informativeness',
#         'local_edge_informativeness': 'Local Edge Informativeness',
#         'local_homophily': 'Local Homophily'
#     }
    
#     fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
#     for ax, (measure, title) in zip(axes, measures.items()):
#         scatter = ax.scatter(
#             depth_df['prediction_depth'],
#             depth_df[measure],
#             c=depth_df['gnn_correct'],
#             cmap='coolwarm',
#             alpha=0.6
#         )
        
#         # Add trend line
#         z = np.polyfit(depth_df['prediction_depth'], depth_df[measure], 1)
#         p = np.poly1d(z)
#         ax.plot(np.unique(depth_df['prediction_depth']), 
#                 p(np.unique(depth_df['prediction_depth'])), 
#                 "r--", alpha=0.8)
        
#         ax.set_title(f'{title} vs Prediction Depth')
#         ax.set_xlabel('Prediction Depth')
#         ax.set_ylabel(title)
        
#         # Add correlation coefficient
#         corr = depth_df['prediction_depth'].corr(depth_df[measure])
#         ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
#                 transform=ax.transAxes, 
#                 bbox=dict(facecolor='white', alpha=0.8))
        
#     plt.tight_layout()
#     plt.savefig(save_dir / f'informativeness_depth_{mask_name}_split_{split_idx}_{timestamp}.png',
#                 bbox_inches='tight', dpi=300)
#     plt.close()
    
#     # Create summary statistics
#     summary_stats = depth_df.groupby('prediction_depth').agg({
#         'local_node_informativeness': ['mean', 'std', 'count'],
#         'local_edge_informativeness': ['mean', 'std', 'count'],
#         'local_homophily': ['mean', 'std', 'count'],
#         'gnn_correct': 'mean'
#     }).round(4)
    
#     summary_stats.to_csv(save_dir / f'informativeness_depth_stats_{mask_name}_split_{split_idx}_{timestamp}.csv')

def analyze_prediction_depth(model, data, train_mask, test_mask, save_dir, split_idx):
    print("\n=== Starting Prediction Depth Analysis ===")
    
    print("Getting layer representations...")
    layer_representations = get_layer_representations(model, data, data.edge_index)
    num_layers = len(layer_representations)
    
    print("Calculating delta entropy...")
    delta_entropy = kd_retention(model, data, noise_level=1.0)
    
    # print("\nPreparing graph informativeness measures...")
    # graph, labels = get_graph_and_labels_from_pyg_dataset(data)
    
    # print("\nCalculating global informativeness measures...")
    # global_li_node = li_node(graph, labels)
    # global_li_edge = li_edge(graph, labels)
    # global_h_edge = h_edge(graph, labels)
    # global_h_adj = h_adj(graph, labels)
    
    # print(f"\nGlobal measures:")
    # print(f"Node informativeness: {global_li_node:.4f}")
    # print(f"Edge informativeness: {global_li_edge:.4f}")
    # print(f"Edge homophily: {global_h_edge:.4f}")
    # print(f"Adjusted homophily: {global_h_adj:.4f}")
    
    #print("Calculating local informativeness measures...")
    #local_info_measures = calculate_local_informativeness(graph, labels)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    for mask_name, mask in [('train', train_mask), ('test', test_mask)]:
        print(f"\nAnalyzing {mask_name} set:")
        mask = mask.cpu().numpy()
        nodes_indices = np.where(mask)[0]
        
        # Get final GNN predictions
        model.eval()
        with torch.no_grad():
            final_out = model(data.x, data.edge_index)
            final_predictions = final_out.argmax(dim=1).cpu().numpy()
            confidences = F.softmax(final_out, dim=1)
        
        node_info = {}
        
        print(f"Processing {len(nodes_indices)} nodes...")
        for node_idx in nodes_indices:
            final_pred = final_predictions[node_idx]
            prediction_history = []
            knn_correct_history = []
            
            # Track predictions at each layer
            for layer_idx, layer_repr in enumerate(layer_representations):
                train_features = layer_repr[train_mask.cpu()]
                train_labels = data.y[train_mask].cpu()
                knn = create_knn_probe(train_features, train_labels)
                
                node_features = layer_repr[node_idx].reshape(1, -1)
                knn_pred = knn.predict(node_features)[0]
                prediction_history.append(knn_pred)
                knn_correct_history.append(knn_pred == data.y[node_idx].item())
            
            prediction_depth = num_layers - 1
            consistent_from = None
            
            # Find prediction depth
            if all(pred == final_pred for pred in prediction_history):
                prediction_depth = 0
                consistent_from = 0
            else:
                for i in range(len(prediction_history)):
                    if (prediction_history[i] == final_pred and 
                        all(pred == final_pred for pred in prediction_history[i:])):
                        prediction_depth = i
                        consistent_from = i
                        break
            
            # Store all information for this node
            node_info[node_idx] = {
                'node_idx': node_idx,
                'prediction_depth': prediction_depth,
                'consistent_from': consistent_from,
                'prediction_history': str(prediction_history),
                'knn_correct_history': str(knn_correct_history),
                'final_prediction': final_pred,
                'true_label': data.y[node_idx].item(),
                'gnn_correct': final_pred == data.y[node_idx].item(),
                'knn_correct_at_depth': knn_correct_history[prediction_depth] if prediction_depth < len(knn_correct_history) else False,
                'delta_entropy': float(delta_entropy[node_idx]),
                #'local_node_informativeness': float(local_info_measures['node'][node_idx]),
               # 'local_edge_informativeness': float(local_info_measures['edge'][node_idx]),
                #'local_homophily': float(local_info_measures['homophily'][node_idx]),
                #'global_li_node': float(global_li_node),
                #'global_li_edge': float(global_li_edge),
                #'global_h_edge': float(global_h_edge),
               # 'global_h_adj': float(global_h_adj)
            }
        
        print(f"Creating DataFrame and visualizations for {mask_name} set...")
        # Create DataFrame
        depth_df = pd.DataFrame.from_dict(node_info, orient='index')
        
        # Save the full depth analysis DataFrame
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        depth_df.to_csv(save_dir / f'full_depth_analysis_{mask_name}_split_{split_idx}_{timestamp}.csv', index=True)
        
        # Create all visualizations
        print(f"Generating visualizations for {mask_name} set...")
        
        # Original visualization functions
        visualize_prediction_depth(depth_df, save_dir, split_idx, mask_name, model)
        
        ##Confidence and entropy visualization
        visualize_prediction_confidence_and_entropy(
            model=model,
            data=data,
            predictions=final_predictions,
            confidences=confidences,
            train_mask=train_mask,
            test_mask=test_mask,
            split_idx=split_idx,
            true_labels=data.y,
            noise_level=1.0,
            save_dir=save_dir
        )
        
        # New informativeness visualizations
        #visualize_informativeness_depth_relationship(depth_df, save_dir, split_idx, mask_name)
        
        # Add gradient pattern analysis
        print(f"Analyzing gradient patterns for {mask_name} set...")
        gradient_results = analyze_gradient_patterns_by_depth(
            model=model,
            data=data,
            depth_df=depth_df,
            criterion=criterion,
            save_dir=save_dir,
            split_idx=split_idx,
            mask_name=mask_name
        )
        
        # Store gradient results in depth_df
        for depth, grad_norms in gradient_results.items():
            for layer_name, norm in grad_norms.items():
                col_name = f'grad_norm_{layer_name.replace(".", "_")}'
                depth_df.loc[depth_df['prediction_depth'] == depth, col_name] = norm
        
        print(f"Completed analysis for {mask_name} set")

        # Add neighborhood analysis for late learners
        stats_df, summary = analyze_late_learner_neighborhoods(
            model=model,
            data=data,
            depth_df=depth_df,
            save_dir=save_dir,
            split_idx=split_idx,
            mask_name=mask_name
        )

    return depth_df

def visualize_gradient_patterns(gradient_results, save_dir, split_idx, mask_name):
    """
    Create line plot visualizations for gradient patterns across different depths
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Plot line for each prediction depth
    depths = list(gradient_results.keys())
    layer_names = list(next(iter(gradient_results.values())).keys())
    
    # Create x-axis positions
    x = range(len(layer_names))
    
    # Plot with different colors for each depth
    for depth in depths:
        gradient_values = [gradient_results[depth][layer] for layer in layer_names]
        plt.plot(x, gradient_values, marker='o', label=f'Depth {depth}', linewidth=2, markersize=8)
    
    # Customize plot
    plt.title(f'Gradient Norms Across Layers ({mask_name} Set)', fontsize=20)
    plt.xlabel('Layer', fontsize=20)
    plt.ylabel('Normalized Gradient Norm', fontsize=20)
    
    # Set x-axis ticks to layer names
    plt.xticks(x, layer_names, rotation=45, ha='right')
    
    # Add legend
    plt.legend(title='Prediction Depth', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Use log scale for y-axis if values span multiple orders of magnitude
    if any(any(v > 10 * min(grad_dict.values()) for v in grad_dict.values()) 
           for grad_dict in gradient_results.values()):
        plt.yscale('log')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_dir / f'gradient_patterns_{mask_name}.png',
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save numerical results
    results_df = pd.DataFrame({
        f'Depth_{depth}': [gradient_results[depth][layer] for layer in layer_names]
        for depth in depths
    }, index=layer_names)
    results_df.to_csv(save_dir / f'gradient_patterns_{mask_name}.csv')
    
    # Print some basic statistics
    print(f"\nGradient Statistics for {mask_name} set:")
    for depth in depths:
        grad_values = [gradient_results[depth][layer] for layer in layer_names]
        print(f"\nDepth {depth}:")
        print(f"  Mean gradient norm: {np.mean(grad_values):.2e}")
        print(f"  Max gradient norm:  {np.max(grad_values):.2e}")
        print(f"  Min gradient norm:  {np.min(grad_values):.2e}")

def compute_normalized_gradient_norms(model, node_indices, data, criterion):
    """
    Compute normalized gradient norms for specific nodes at each layer
    """
    layer_grads = defaultdict(list)
    model.train()  # Set to train mode to ensure gradients are computed
    
    # Ensure model parameters require gradients
    for param in model.parameters():
        param.requires_grad = True
    
    for node_idx in node_indices:
        model.zero_grad()
        
        # Forward pass with gradient tracking
        with torch.set_grad_enabled(True):
            out = model(data.x, data.edge_index)
            # Ensure output requires gradients
            if not out.requires_grad:
                out.requires_grad = True
            
            # Compute loss for single node
            node_out = out[node_idx].unsqueeze(0)
            node_label = data.y[node_idx].unsqueeze(0)
            loss = criterion(node_out, node_label)
            
            # Backward pass
            loss.backward()
        
            # Collect gradients layer by layer
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Normalize by sqrt(number of parameters)
                    norm = param.grad.norm().item() / np.sqrt(param.numel())
                    layer_grads[name].append(norm)
    
    # Average the gradients for each layer
    avg_layer_grads = {
        name: np.mean(grads) for name, grads in layer_grads.items()
    }
    
    return avg_layer_grads

def analyze_gradient_patterns_by_depth(model, data, depth_df, criterion, save_dir, split_idx, mask_name):
    """
    Analyze gradient patterns for nodes learned at different depths
    """
    print(f"\nAnalyzing gradient patterns for {mask_name} set...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Move data to same device as model
    device = next(model.parameters()).device
    data = data.to(device)
    
    # Get all unique depths
    depths = sorted(depth_df['prediction_depth'].unique())
    gradient_results = {}
    
    # Process each depth
    for depth in depths:
        print(f"Processing nodes at depth {depth}...")
        nodes = depth_df[depth_df['prediction_depth'] == depth]['node_idx'].values
        
        # Sample nodes if too many (up to 50 per depth)
        if len(nodes) > 50:
            nodes = np.random.choice(nodes, 50, replace=False)
            
        # Compute gradient norms for these nodes
        grad_norms = compute_normalized_gradient_norms(model, nodes, data, criterion)
        gradient_results[depth] = grad_norms
    
    # Prepare for plotting
    layer_names = list(next(iter(gradient_results.values())).keys())
    plt.figure(figsize=(12, 6))
    x = range(len(layer_names))
    
    # Plot line for each depth with different colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(depths)))
    for depth, color in zip(depths, colors):
        values = [gradient_results[depth][layer] for layer in layer_names]
        plt.plot(x, values, '-o', label=f'Depth {depth}', color=color, 
                linewidth=2, markersize=8)
    
    plt.title(f'Gradient Norms Across Layers ({mask_name} Set)', fontsize=15)
    plt.xlabel('Model Layer', fontsize=12)
    plt.ylabel('Normalized Gradient Norm', fontsize=12)
    
    # Set x-axis ticks to layer names
    plt.xticks(x, layer_names, rotation=45, ha='right')
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Use log scale if values span multiple orders of magnitude
    if any(any(v > 10 * min(layer_dict.values()) for v in layer_dict.values()) 
           for layer_dict in gradient_results.values()):
        plt.yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_dir / f'gradient_patterns_{mask_name}.png',
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save numerical results
    results_df = pd.DataFrame({
        f'Depth_{depth}': [gradient_results[depth][layer] for layer in layer_names]
        for depth in depths
    }, index=layer_names)
    results_df.to_csv(save_dir / f'gradient_patterns_{mask_name}.csv')
    
    return gradient_results

def aggregate_gradient_patterns(all_gradient_results, save_dir, mask_name):
    """
    Aggregate gradient patterns across multiple seeds
    """
    print("\nAggregating gradient patterns across seeds...")
    
    # Create depth-specific plots (existing code)
    mean_results, ci_results = create_depth_specific_plots(all_gradient_results, save_dir, mask_name)
    
    # Create average gradient norms plot
    plot_average_gradient_norms(all_gradient_results, save_dir, mask_name)
    
    return mean_results, ci_results

def plot_average_gradient_norms(all_gradient_results, save_dir, mask_name):
    """
    Create a plot showing average gradient norms across all seeds and depths
    """
    print("\nCreating average gradient norms plot...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Initialize storage for layer-wise gradients
    layer_gradients = defaultdict(list)
    
    # Collect all gradient values for each layer across all seeds and depths
    for seed_results in all_gradient_results:
        for depth_results in seed_results.values():
            for layer_name, grad_value in depth_results.items():
                layer_gradients[layer_name].append(grad_value)
    
    # Calculate means and confidence intervals
    layer_means = {}
    layer_cis = {}
    
    for layer_name, grads in layer_gradients.items():
        layer_means[layer_name] = np.mean(grads)
        layer_cis[layer_name] = 1.96 * np.std(grads) / np.sqrt(len(grads))
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Get layer names and sort them if needed
    layer_names = sorted(layer_means.keys())
    x = range(len(layer_names))
    
    # Plot means with error bars
    means = [layer_means[layer] for layer in layer_names]
    cis = [layer_cis[layer] for layer in layer_names]
    
    plt.errorbar(x, means, yerr=cis,
                marker='o', color='blue', capsize=5, capthick=1,
                elinewidth=1, linewidth=2, markersize=8,
                label='Average across all depths')
    
    plt.title(f'Average Gradient Norms Across All Seeds and Depths ({mask_name} Set)', fontsize=15)
    plt.xlabel('Model Layer', fontsize=12)
    plt.ylabel('Normalized Gradient Norm', fontsize=12)
    
    plt.xticks(x, layer_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Use log scale if values span multiple orders of magnitude
    if max(means) > 10 * min(means):
        plt.yscale('log')
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_dir / f'average_gradient_norms_{mask_name}_{timestamp}.png',
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save numerical results
    results_df = pd.DataFrame({
        'layer_name': layer_names,
        'mean_gradient': means,
        'confidence_interval': cis
    })
    results_df.to_csv(save_dir / f'average_gradient_norms_{mask_name}_{timestamp}.csv')
    
    # Print summary statistics
    print("\nOverall Gradient Statistics:")
    print(f"Mean gradient norm: {np.mean(means):.2e}")
    print(f"Max gradient norm:  {np.max(means):.2e}")
    print(f"Min gradient norm:  {np.min(means):.2e}")

def analyze_late_learner_neighborhoods(model, data, depth_df, save_dir, split_idx, mask_name):
    """Analyze neighborhoods of late learners (nodes learned in layers 4-5)."""
    print("\nAnalyzing neighborhoods of late learners...")
    
    # Get graph and labels
    graph = to_networkx(data)
    labels = data.y.cpu().numpy()
    
    # Identify late learners
    late_learners = depth_df[depth_df['prediction_depth'] >= 4]['node_idx'].values
    print(f"Found {len(late_learners)} late learners in {mask_name} set")
    
    if len(late_learners) == 0:
        print("No late learners found. Skipping neighborhood analysis.")
        return None, None
    
    neighborhood_stats = []
    
    for node in late_learners:
        try:
            # Get 2-hop neighborhood
            neighborhood = nx.ego_graph(graph, int(node), radius=2)
            
            if len(neighborhood) < 2:
                continue
                
            # Create mapping from original node IDs to consecutive integers
            neighborhood_nodes = list(neighborhood.nodes())
            node_mapping = {old: new for new, old in enumerate(neighborhood_nodes)}
            
            # Create remapped subgraph
            subgraph = nx.relabel_nodes(neighborhood, node_mapping)
            
            # Remap labels to match new node indices
            neighborhood_labels = np.array([labels[original_idx] for original_idx in neighborhood_nodes])
            
            # Calculate metrics for neighborhood
            stats = {
                'node_idx': int(node),
                'prediction_depth': float(depth_df[depth_df['node_idx'] == node]['prediction_depth'].iloc[0]),
                'neighborhood_size': len(neighborhood),
                'neighborhood_edges': len(subgraph.edges()),
                'node_degree': graph.degree(int(node)),
                'neighborhood_li_node': float(li_node(subgraph, neighborhood_labels)),
                'neighborhood_li_edge': float(li_edge(subgraph, neighborhood_labels)),
                'neighborhood_homophily': float(h_edge(subgraph, neighborhood_labels)),
                'center_node_label': int(labels[node]),
                'unique_neighbor_labels': len(np.unique(neighborhood_labels))
            }
            
            # Add entropy information if available
            if 'delta_entropy' in depth_df.columns:
                stats['delta_entropy'] = float(depth_df[depth_df['node_idx'] == node]['delta_entropy'].iloc[0])
            
            neighborhood_stats.append(stats)
            
        except Exception as e:
            print(f"Error processing node {node}: {str(e)}")
            continue
    
    if not neighborhood_stats:
        print("No valid neighborhoods found. Skipping visualization.")
        return None, None
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(neighborhood_stats)
    
    # Create visualizations only if we have data
    if not stats_df.empty:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        try:
            # 1. Neighborhood characteristics distribution
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            sns.histplot(data=stats_df, x='neighborhood_size', bins=20, ax=axes[0])
            axes[0].set_title('Distribution of Neighborhood Sizes')
            
            sns.scatterplot(data=stats_df, x='neighborhood_li_node', y='neighborhood_homophily', ax=axes[1])
            axes[1].set_title('Node Informativeness vs Homophily')
            
            sns.boxplot(data=stats_df, x='prediction_depth', y='neighborhood_li_edge', ax=axes[2])
            axes[2].set_title('Edge Informativeness by Depth')
            
            if 'delta_entropy' in stats_df.columns:
                sns.scatterplot(data=stats_df, x='delta_entropy', y='neighborhood_homophily', ax=axes[3])
                axes[3].set_title('Delta Entropy vs Neighborhood Homophily')
            
            plt.tight_layout()
            plt.savefig(save_dir / f'late_learner_neighborhood_analysis_{mask_name}_split_{split_idx}_{timestamp}.png',
                        bbox_inches='tight', dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
    
    # Save numerical results
    try:
        stats_df.to_csv(save_dir / f'late_learner_neighborhood_stats_{mask_name}_split_{split_idx}_{timestamp}.csv')
    except Exception as e:
        print(f"Error saving stats DataFrame: {str(e)}")
    
    # Calculate and print summary statistics
    if not stats_df.empty:
        summary = {
            'avg_neighborhood_size': stats_df['neighborhood_size'].mean(),
            'avg_neighborhood_edges': stats_df['neighborhood_edges'].mean(),
            'avg_node_degree': stats_df['node_degree'].mean(),
            'avg_li_node': stats_df['neighborhood_li_node'].mean(),
            'avg_li_edge': stats_df['neighborhood_li_edge'].mean(),
            'avg_homophily': stats_df['neighborhood_homophily'].mean(),
            'avg_unique_labels': stats_df['unique_neighbor_labels'].mean()
        }
        
        print(f"\nLate Learner Neighborhood Summary ({mask_name} set):")
        for metric, value in summary.items():
            print(f"{metric}: {value:.4f}")
    else:
        summary = None
    
    return stats_df, summary

# def create_aggregate_neighborhood_analysis(all_neighborhood_stats, save_dir):
#     """
#     Create aggregate analysis of neighborhood statistics across all seeds.
    
#     Args:
#         all_neighborhood_stats: List of mask names ('train', 'test') or actual stats
#         save_dir: Directory to save aggregated results
#     """
#     print("\nCreating aggregate neighborhood analysis...")
#     timestamp = time.strftime("%Y%m%d_%H%M%S")
    
#     # Combine all DataFrames with more flexible handling
#     all_dfs = []
#     for stats in all_neighborhood_stats:
#         try:
#             # Handle different input formats
#             if isinstance(stats, tuple):
#                 stats_df, _ = stats  # If it's a tuple (stats_df, summary)
#                 all_dfs.append(stats_df)
#             elif isinstance(stats, pd.DataFrame):
#                 all_dfs.append(stats)  # If it's directly a DataFrame
#             elif isinstance(stats, str):
#                 # Look for CSV files matching the mask pattern
#                 mask_name = stats.lower()
#                 matching_files = list(save_dir.glob(f'late_learner_neighborhood_stats_{mask_name}_split_*.csv'))
                
#                 if matching_files:
#                     print(f"Found {len(matching_files)} files for {mask_name} mask")
#                     for file_path in matching_files:
#                         try:
#                             df = pd.read_csv(file_path)
#                             df['mask'] = mask_name  # Add mask type as a column
#                             all_dfs.append(df)
#                             print(f"Successfully loaded DataFrame from {file_path}")
#                         except Exception as e:
#                             print(f"Error loading {file_path}: {str(e)}")
#                 else:
#                     print(f"No matching files found for mask: {mask_name}")
#             else:
#                 print(f"Skipping invalid stats format: {type(stats)}")
#                 continue
                
#         except Exception as e:
#             print(f"Error processing stats: {str(e)}")
#             continue
    
#     if not all_dfs:
#         print("No valid neighborhood statistics found. Skipping aggregation.")
#         return
    
#     # Rest of the function remains the same
#     combined_df = pd.concat(all_dfs, ignore_index=True)
    
#     # Create visualizations for aggregate data
#     plt.figure(figsize=(15, 10))
    
#     # 1. Box plots for key metrics
#     metrics = ['neighborhood_size', 'neighborhood_li_node', 'neighborhood_li_edge', 'neighborhood_homophily']
#     fig, axes = plt.subplots(2, 2, figsize=(15, 12))
#     axes = axes.ravel()
    
#     for i, metric in enumerate(metrics):
#         # Add mask to the plot if available
#         if 'mask' in combined_df.columns:
#             sns.boxplot(data=combined_df, x='prediction_depth', y=metric, hue='mask', ax=axes[i])
#         else:
#             sns.boxplot(data=combined_df, x='prediction_depth', y=metric, ax=axes[i])
#         axes[i].set_title(f'{metric.replace("_", " ").title()} by Depth')
#         axes[i].tick_params(axis='x', rotation=45)
    
#     plt.tight_layout()
#     plt.savefig(save_dir / f'aggregate_neighborhood_metrics_{timestamp}.png',
#                 bbox_inches='tight', dpi=300)
#     plt.close()
    
#     # 2. Correlation heatmap
#     plt.figure(figsize=(10, 8))
#     correlation_matrix = combined_df[metrics + ['prediction_depth']].corr()
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
#     plt.title('Correlation between Neighborhood Metrics')
#     plt.tight_layout()
#     plt.savefig(save_dir / f'neighborhood_metrics_correlation_{timestamp}.png',
#                 bbox_inches='tight', dpi=300)
#     plt.close()
    
#     # Calculate and save aggregate statistics
#     aggregate_stats = {
#         'metrics': {
#             metric: {
#                 'mean': combined_df[metric].mean(),
#                 'std': combined_df[metric].std(),
#                 'median': combined_df[metric].median(),
#                 'q1': combined_df[metric].quantile(0.25),
#                 'q3': combined_df[metric].quantile(0.75)
#             } for metric in metrics
#         },
#         'depth_distribution': combined_df['prediction_depth'].value_counts().to_dict(),
#         'total_nodes_analyzed': len(combined_df),
#         'correlation_matrix': correlation_matrix.to_dict()
#     }
    
#     # Add mask-specific statistics if available
#     if 'mask' in combined_df.columns:
#         aggregate_stats['mask_specific'] = {
#             mask_type: {
#                 metric: {
#                     'mean': mask_data[metric].mean(),
#                     'std': mask_data[metric].std()
#                 } for metric in metrics
#             } for mask_type, mask_data in combined_df.groupby('mask')
#         }
    
#     # Save aggregate statistics
#     with open(save_dir / f'aggregate_neighborhood_stats_{timestamp}.json', 'w') as f:
#         json.dump(aggregate_stats, f, indent=4)
    
#     # Save combined DataFrame
#     combined_df.to_csv(save_dir / f'combined_neighborhood_stats_{timestamp}.csv', index=False)
    
#     # Print summary
#     print("\nAggregate Neighborhood Analysis Summary:")
#     print(f"Total nodes analyzed: {len(combined_df)}")
#     if 'mask' in combined_df.columns:
#         for mask_type, mask_data in combined_df.groupby('mask'):
#             print(f"\n{mask_type.upper()} set statistics:")
#             for metric in metrics:
#                 print(f"{metric}: {mask_data[metric].mean():.4f} (±{mask_data[metric].std():.4f})")
#     else:
#         print("\nMean values across all seeds:")
#         for metric in metrics:
#             print(f"{metric}: {combined_df[metric].mean():.4f} (±{combined_df[metric].std():.4f})")