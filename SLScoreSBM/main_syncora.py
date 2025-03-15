import argparse
import torch
import os
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import networkx as nx
from scipy import sparse

from dataset import CustomDataset
from model import NodeGCN, NodeGAT, NodeGraphConv
from memorization import calculate_node_memorization_score
from main import (set_seed, train_models, verify_no_data_leakage, 
                 setup_logging, get_model, test)
from nodeli import li_node


def load_and_process_dataset(args, dataset_path, logger):
    """Load synthetic Cora dataset and convert to PyG format"""
    dataset = CustomDataset(root="syn-cora", name=dataset_path, setting="gcn")
    
    # Convert to PyG format
    edge_index = torch.from_numpy(np.vstack(dataset.adj.nonzero())).long()
    
    # Convert sparse features to dense numpy array
    if sparse.issparse(dataset.features):
        x = torch.from_numpy(dataset.features.todense()).float()
    else:
        x = torch.from_numpy(dataset.features).float()
    
    y = torch.from_numpy(dataset.labels).long()
    
    # Create train/val/test masks
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    
    train_mask[dataset.idx_train] = True
    val_mask[dataset.idx_val] = True
    test_mask[dataset.idx_test] = True
    
    # Convert to networkx for informativeness calculation
    G = nx.Graph()
    G.add_nodes_from(range(len(y)))
    G.add_edges_from(edge_index.t().numpy())
    
    # Calculate label informativeness using existing function
    informativeness = li_node(G, dataset.labels)
    
    # Calculate homophily (edge homophily)
    edges = edge_index.t().numpy()
    same_label = dataset.labels[edges[:, 0]] == dataset.labels[edges[:, 1]]
    homophily = same_label.mean()
    
    # Create a data object
    data = type('Data', (), {
        'x': x,
        'y': y,
        'edge_index': edge_index,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'num_nodes': len(y),
        'informativeness': informativeness,
        'homophily': homophily
    })()
    
    logger.info(f"\nDataset Statistics:")
    logger.info(f"Number of nodes: {data.num_nodes}")
    logger.info(f"Number of edges: {len(edges)}")
    logger.info(f"Number of features: {x.shape[1]}")
    logger.info(f"Number of classes: {len(torch.unique(y))}")
    logger.info(f"Homophily: {homophily:.4f}")
    logger.info(f"Label Informativeness: {informativeness:.4f}")
    
    return data

def create_visualization(results_df, save_path, args):
    """Create scatter plot of homophily vs informativeness colored by memorization rate"""
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(
        results_df['homophily'],
        results_df['informativeness'],
        c=results_df['percent_memorized'],
        cmap='viridis',
        s=100,
        alpha=0.7
    )
    
    plt.xlabel('Homophily', fontsize=12)
    plt.ylabel('Label Informativeness', fontsize=12)
    plt.title(f'Memorization Analysis\nModel: {args.model_type.upper()}', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Memorization Rate (%)', fontsize=10)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def get_node_splits(data, train_mask, swap_candidate_independent=False):
    """
    Create node splits using all available training nodes.
    
    Args:
        data: PyG data object
        train_mask: Mask for train nodes
        swap_candidate_independent: If True, swap the roles of candidate and independent nodes
    """
    # Get train indices in their original order
    train_indices = torch.where(train_mask)[0]
    num_train_nodes = len(train_indices)
    
    # Calculate split sizes: 50% shared, 25% candidate, 25% independent
    shared_size = int(0.50 * num_train_nodes)
    remaining = num_train_nodes - shared_size
    split_size = remaining // 2
    
    # Split indices sequentially without shuffling
    shared_idx = train_indices[:shared_size].tolist()
    candidate_idx = train_indices[shared_size:shared_size + split_size].tolist()
    independent_idx = train_indices[shared_size + split_size:].tolist()
    
    # Return swapped indices if requested
    if swap_candidate_independent:
        return shared_idx, independent_idx, candidate_idx
    else:
        return shared_idx, candidate_idx, independent_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='gcn',
                       choices=['gcn', 'gat', 'graphconv'])
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--gat_heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_passes', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='results/syncora_analysis')
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'syncora_analysis_{args.model_type}_{timestamp}'
    log_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging with both file and console output
    logger = logging.getLogger('syncora_analysis')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear any existing handlers
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'analysis.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Process each synthetic Cora dataset
    results = []
    dataset_files = sorted([f[:-4] for f in os.listdir('syn-cora') if f.endswith('.npz')])
    
    for dataset_name in tqdm(dataset_files, desc="Processing datasets"):
        logger.info(f"\nProcessing dataset: {dataset_name}")
        
        # Load and process dataset
        data = load_and_process_dataset(args, dataset_name, logger)
        #data = data.to(device)
        
        # Get node splits
        shared_idx, candidate_idx, independent_idx = get_node_splits(
            data, data.train_mask, swap_candidate_independent=False
        )
        
        # Get extra indices from test set
        test_indices = torch.where(data.test_mask)[0]
        extra_size = len(candidate_idx)
        extra_indices = test_indices[:extra_size].tolist()
        
        # Create nodes_dict
        nodes_dict = {
            'shared': shared_idx,
            'candidate': candidate_idx,
            'independent': independent_idx,
            'extra': extra_indices,
            'val': torch.where(data.val_mask)[0].tolist(),
            'test': torch.where(data.test_mask)[0].tolist()
        }
        
        # Train models
        model_f, model_g, f_val_acc, g_val_acc = train_models(
            args=args,
            data=data,
            shared_idx=shared_idx,
            candidate_idx=candidate_idx,
            independent_idx=independent_idx,
            device=device,
            logger=logger,
            output_dir=None
        )
        
        # Calculate memorization scores
        node_scores = calculate_node_memorization_score(
            model_f=model_f,
            model_g=model_g,
            data=data,
            nodes_dict=nodes_dict,
            device=device,
            logger=logger,
            num_passes=args.num_passes
        )
        
        # Store results
        results.append({
            'dataset': dataset_name,
            'homophily': float(data.homophily),
            'informativeness': float(data.informativeness),
            'percent_memorized': node_scores['candidate']['percentage_above_threshold'],
            'avg_memorization': node_scores['candidate']['avg_score'],
            'num_memorized': node_scores['candidate']['nodes_above_threshold'],
            'total_nodes': len(node_scores['candidate']['mem_scores']),
            'f_val_acc': float(f_val_acc),
            'g_val_acc': float(g_val_acc)
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create visualization
    plot_path = os.path.join(log_dir, f'memorization_analysis_{timestamp}.png')
    create_visualization(results_df, plot_path, args)
    
    # Save results
    results_df.to_csv(os.path.join(log_dir, 'results.csv'), index=False)
    
    # Calculate correlations
    correlations = results_df[['homophily', 'informativeness', 'percent_memorized']].corr()
    correlations.to_csv(os.path.join(log_dir, 'correlations.csv'))
    
    logger.info("\nAnalysis complete!")
    logger.info(f"Results saved to: {log_dir}")
    logger.info(f"Visualization saved as: {plot_path}")

if __name__ == '__main__':
    main()