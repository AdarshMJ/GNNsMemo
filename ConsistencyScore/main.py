import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import LargestConnectedComponents, RandomNodeSplit
import os
import logging
from datetime import datetime
from model import SimpleGCN, GATv2, GraphConv, AsymGNN
import numpy as np
import random
from utils import (select_candidate_nodes, 
                  compute_memorization_score, save_memorization_results, prepare_dropped_data,
                  plot_memorization_scores, plot_aggregate_results)
import scipy.sparse as sp
import pandas as pd
import copy

def setup_logging(args):
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create log filename with timestamp
    log_filename = f'logs/training_{args.dataset}_{args.model.lower()}_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    # Log all arguments
    logging.info('Training with following parameters:')
    for arg, value in vars(args).items():
        logging.info(f'{arg}: {value}')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[mask] == data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
    return acc

def update_masks_after_node_drop(data, drop_node_list):
    """Update masks and node indices after dropping nodes."""
    device = data.x.device
    num_nodes = data.num_nodes
    remaining_nodes = sorted(list(set(range(num_nodes)) - set(drop_node_list)))
    
    # Create new masks with correct size
    new_train_mask = torch.zeros(len(remaining_nodes), dtype=torch.bool, device=device)
    new_val_mask = torch.zeros(len(remaining_nodes), dtype=torch.bool, device=device)
    new_test_mask = torch.zeros(len(remaining_nodes), dtype=torch.bool, device=device)
    new_y = torch.zeros(len(remaining_nodes), dtype=data.y.dtype, device=device)
    
    # Map old indices to new indices
    for new_idx, old_idx in enumerate(remaining_nodes):
        if data.train_mask[old_idx]:
            new_train_mask[new_idx] = True
        if data.val_mask[old_idx]:
            new_val_mask[new_idx] = True
        if data.test_mask[old_idx]:
            new_test_mask[new_idx] = True
        new_y[new_idx] = data.y[old_idx]
    
    return new_train_mask, new_val_mask, new_test_mask, new_y

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def main():
    parser = argparse.ArgumentParser(description='Train GNN models for node classification')
    parser.add_argument('--model', type=str, required=True, help='Model type (GCN, GATv2, GraphConv, AsymGNN)')
    parser.add_argument('--dataset', type=str, required=True, choices=['Cora', 'Citeseer', 'PubMed'], 
                       help='Dataset name')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--device', type=str, default='mps', choices=['cuda', 'cpu', 'mps'],
                       help='Device to run on')
    parser.add_argument('--edge_drop_rate', type=float, default=0.0,
                       help='Fraction of edges to randomly drop for data augmentation (0.0 means no edge dropping)')
    parser.add_argument('--num_candidates', type=int, default=100,
                       help='Number of candidate nodes to select and analyze for memorization')
    parser.add_argument('--num_passes', type=int, default=1,
                       help='Number of forward passes for memorization score calculation')
    parser.add_argument('--memorization_threshold', type=float, default=0.5,
                       help='Threshold for strong memorization')

    args = parser.parse_args()
    
    # Add device initialization after argument parsing
    device = get_device()
    logging.info(f'Using device: {device}')
    
    # Predefined seeds for reproducibility
    seeds = [3164711608, 894959334, 2487307261, 3349051410, 493067366]
    #seeds = [3164711608]
    # Storage for results
    all_test_acc = []
    all_memorization_results = []
    candidate_nodes = None
    
    for seed in seeds:
        # Set seed before any operations
        set_seed(seed)
        
        # Setup logging for each seed
        logger = logging.getLogger()
        logger.handlers = []  # Clear existing handlers
        
        log_filename = f'logs/training_{args.dataset}_{args.model.lower()}_{seed}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        logging.info(f'\n{"-"*40}')
        logging.info(f'Starting training with seed: {seed}')
        logging.info(f'{"-"*40}\n')
        
        # Reload fresh dataset for each seed
        transform = RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2)
        dataset = Planetoid(root='data', name=args.dataset, transform=transform)
        data = dataset[0].to(device)
        
        # Apply largest connected components transform
        lcc_transform = LargestConnectedComponents()
        data = lcc_transform(data)
        
        # Log dataset statistics
        num_train = int(data.train_mask.sum())
        num_val = int(data.val_mask.sum())
        num_test = int(data.test_mask.sum())
        total_nodes = data.num_nodes
        
        logging.info(f'Dataset Statistics:')
        logging.info(f'Total number of nodes: {total_nodes}')
        logging.info(f'Number of training nodes: {num_train} ({num_train/total_nodes:.2%})')
        logging.info(f'Number of validation nodes: {num_val} ({num_val/total_nodes:.2%})')
        logging.info(f'Number of test nodes: {num_test} ({num_test/total_nodes:.2%})')
        
        # Log initial edge statistics
        num_edges = data.edge_index.size(1) // 2  # Divide by 2 since edges are bidirectional
        logging.info(f'Initial number of edges: {num_edges}')

        # Apply edge dropping if specified
        if args.edge_drop_rate > 0:
            logging.info(f'Applying edge dropping with rate={args.edge_drop_rate}')
            
            # Convert to sparse matrix format for augmentation
            edge_index = data.edge_index.cpu()
            adj = torch.zeros((data.num_nodes, data.num_nodes))
            adj[edge_index[0], edge_index[1]] = 1
            adj = sp.csr_matrix(adj.numpy())
            
            # Convert to numpy array for edge counting
            adj_dense = adj.todense()
            initial_edges = np.sum(adj_dense) // 2
            
            # Apply edge dropping
            aug_adj = aug_random_edge(adj, args.edge_drop_rate)
            
            # Count remaining edges
            remaining_edges = np.sum(aug_adj.todense()) // 2
            edges_dropped = initial_edges - remaining_edges
            
            logging.info(f'Edge Augmentation Statistics:')
            logging.info(f'Initial edges: {initial_edges}')
            logging.info(f'Edges dropped: {edges_dropped}')
            logging.info(f'Remaining edges: {remaining_edges}')
            logging.info(f'Actual drop percentage: {(edges_dropped/initial_edges)*100:.2f}%')
            
            # Convert back to edge_index format
            aug_adj_dense = torch.tensor(aug_adj.todense())
            edge_index = torch.nonzero(aug_adj_dense).t().to(device)
            data.edge_index = edge_index
        
        # Select candidate nodes (only in first seed)
        if candidate_nodes is None:
            candidate_nodes = select_candidate_nodes(data, args.num_candidates, seed=seed)
            logging.info(f'Selected {len(candidate_nodes)} candidate nodes for memorization analysis')
        
        # Train model f (with all nodes)
        model_f = create_model(args, dataset).to(device)
        model_f = train_model(model_f, data, args)
        
        # Calculate and store test accuracy
        test_acc = evaluate(model_f, data, data.test_mask)
        all_test_acc.append(test_acc)
        logging.info(f'Test accuracy for seed {seed}: {test_acc:.4f}')
        
        # Train model g (without candidate nodes)
        data_without_candidates, _ = prepare_dropped_data(data, candidate_nodes)
        model_g = create_model(args, dataset).to(device)
        model_g = train_model(model_g, data_without_candidates, args)
        
        # Compute memorization scores
        results_df = compute_memorization_score(
            model_f, model_g, data, candidate_nodes, args.num_passes
        )
        results_df['seed'] = seed
        all_memorization_results.append(results_df)
        
        # Save results and create visualizations
        save_path = save_memorization_results(results_df, args.dataset, args.model, seed)
        plot_path = plot_memorization_scores(results_df, args.dataset, args.model, seed)
        logging.info(f'Saved memorization results to {save_path}')
        logging.info(f'Saved memorization plots to {plot_path}')
        
        # Log memorization statistics
        strong_mem = (results_df['mem_score'] > args.memorization_threshold).mean()
        avg_mem = results_df['mem_score'].mean()
        logging.info(f'Seed {seed} memorization statistics:')
        logging.info(f'Average memorization score: {avg_mem:.4f}')
        logging.info(f'Proportion of strong memorization: {strong_mem:.4f}')
    
    # Calculate final statistics
    if all_test_acc:  # Only calculate if we have test accuracies
        mean_acc = np.mean(all_test_acc)
        std_acc = np.std(all_test_acc)
        ci = 1.96 * (std_acc / np.sqrt(len(seeds)))  # 95% CI
        
        # Final report
        logging.info('\nFinal Report:')
        logging.info(f'Seeds used: {seeds}')
        logging.info(f'Mean Test Accuracy: {mean_acc:.4f}')
        logging.info(f'Standard Deviation: {std_acc:.4f}')
        logging.info(f'95% Confidence Interval: ±{ci:.4f}')
        logging.info(f'Accuracy Range: [{mean_acc-ci:.4f} - {mean_acc+ci:.4f}]')
    
    # Aggregate memorization results if available
    if all_memorization_results:
        all_results_df = pd.concat(all_memorization_results, ignore_index=True)
        final_results = all_results_df.groupby('node_idx').agg({
            'mem_score': ['mean', 'std'],
            'true_label': 'first',
            'pred_f': lambda x: x.mode()[0],
            'pred_g': lambda x: x.mode()[0],
            'conf_f': 'mean',
            'conf_g': 'mean'
        }).reset_index()
        
        # Save final results and plots
        final_save_path = save_memorization_results(final_results, args.dataset, args.model, 'aggregate')
        final_plot_path = plot_aggregate_results(final_results, args.dataset, args.model, 
                                               threshold=args.memorization_threshold)
        
        logging.info('\nFinal Memorization Analysis:')
        logging.info(f'Results saved to: {final_save_path}')
        logging.info(f'Plots saved to: {final_plot_path}')
        logging.info(f'Average memorization score: {final_results["mem_score"]["mean"].mean():.4f}')
        logging.info(f'Std of memorization scores: {final_results["mem_score"]["std"].mean():.4f}')
        strong_mem_prop = (final_results["mem_score"]["mean"] > args.memorization_threshold).mean()
        logging.info(f'Proportion of nodes with strong memorization: {strong_mem_prop:.4f}')

def train_model(model, data, args):
    """Helper function to train a model with given data"""
    device = data.x.device
    model = model.to(device)  # Ensure model is on the correct device
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        val_acc = evaluate(model, data, data.val_mask)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def create_model(args, dataset):
    """Helper function to create a model instance"""
    model_name = args.model.lower()
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    
    if model_name == 'gcn':
        return SimpleGCN(num_features, num_classes, args.hidden_dim, args.num_layers)
    elif model_name == 'gatv2':
        return GATv2(num_features, args.hidden_dim, num_classes, args.num_layers)
    elif model_name == 'graphconv':
        return GraphConv(num_features, num_classes, args.hidden_dim, args.num_layers)
    elif model_name == 'asymgnn':
        return AsymGNN(num_features, num_classes, args.hidden_dim, args.num_layers)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

if __name__ == '__main__':
    main()