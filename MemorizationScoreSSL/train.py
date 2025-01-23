# train.py
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import os
import os.path as osp
from memscore import MemorizationScorer, plot_alignment_memorization, GraphAugmentor
import argparse
import time
import pandas as pd
from datetime import datetime
import numpy as np
import copy
from tqdm import tqdm


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)
    
    def encode(self, x, edge_index):
        return self.forward(x, edge_index)

def log_graph_stats(data, augmented_edges, node_idx, results, log_file, args):
    """Log average statistics about the graph augmentations for each node"""
    original_edges = data.edge_index.shape[1]
    
    # Compute average stats across all augmentations
    avg_edges_removed = 0
    avg_edges_added = 0
    avg_edge_change_ratio = 0
    
    for aug_edge_index in augmented_edges:
        aug_edges = aug_edge_index.shape[1]
        edges_removed = original_edges - len(set(map(tuple, data.edge_index.t().tolist())) & 
                                          set(map(tuple, aug_edge_index.t().tolist())))
        edges_added = aug_edges - (original_edges - edges_removed)
        
        avg_edges_removed += edges_removed
        avg_edges_added += edges_added
        avg_edge_change_ratio += (aug_edges - original_edges) / original_edges
    
    num_augs = len(augmented_edges)
    stats = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': args.dataset,
        'node_idx': node_idx,
        'original_nodes': data.num_nodes,
        'original_edges': original_edges,
        'avg_edges_removed': avg_edges_removed / num_augs,
        'avg_edges_added': avg_edges_added / num_augs,
        'avg_edge_change_ratio': avg_edge_change_ratio / num_augs,
        'f_alignment': results['f_alignment'],
        'g_alignment': results['g_alignment'],
        'memorization_score': results['memorization'],
        'f_test_acc': results.get('f_test_acc', float('nan')),
        'g_test_acc': results.get('g_test_acc', float('nan'))
    }
    
    # Convert to DataFrame and append to CSV
    df = pd.DataFrame([stats])
    df.to_csv(log_file, mode='a', header=not osp.exists(log_file), index=False)
    
    return df

def create_node_splits(data, train_ratio=0.8, candidate_ratio=0.1):
    """Create node splits following the paper's strategy, respecting original train/test splits"""
    # Only consider training nodes for our splits
    train_nodes = torch.where(data.train_mask)[0]
    num_train_nodes = len(train_nodes)
    
    # Shuffle training nodes
    shuffled_train_nodes = train_nodes[torch.randperm(num_train_nodes)]
    
    # Calculate split sizes from training nodes only
    num_shared = int(num_train_nodes * train_ratio)  # 80% of train nodes
    num_candidates = int(num_train_nodes * candidate_ratio)  # 10% of train nodes
    
    # Create splits
    shared_nodes = shuffled_train_nodes[:num_shared]  # S_S: shared between f and g
    candidate_nodes = shuffled_train_nodes[num_shared:num_shared+num_candidates]  # S_C
    independent_nodes = shuffled_train_nodes[num_shared+num_candidates:num_shared+2*num_candidates]  # S_I
    
    print(f"\nSplit statistics (from training nodes only):")
    print(f"Total training nodes: {num_train_nodes}")
    print(f"Shared nodes (S_S): {len(shared_nodes)}")
    print(f"Candidate nodes (S_C): {len(candidate_nodes)}")
    print(f"Independent nodes (S_I): {len(independent_nodes)}")
    print(f"Test nodes (unchanged): {torch.sum(data.test_mask).item()}")
    
    return shared_nodes, candidate_nodes, independent_nodes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'CiteSeer', 'PubMed'],
                        help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--out_channels', type=int, default=32)
    parser.add_argument('--model_type', type=str, default='gcn', 
                        choices=['encoder', 'gcn'],
                        help='Type of model to use for memorization scoring')
    parser.add_argument('--edge_perturb_ratio', type=float, default=0.1,
                        help='Ratio of edges to perturb in graph augmentation')
    parser.add_argument('--shared_ratio', type=float, default=0.45,
                        help='Ratio of shared nodes (S_S)')
    parser.add_argument('--candidate_ratio', type=float, default=0.35,
                        help='Ratio of candidate nodes (S_C)')
    parser.add_argument('--seeds', nargs='+', type=int, 
                       default=[3164711608, 894959334, 2487307261, 3349051410, 493067366],
                       help='Random seeds for multiple runs')
    args = parser.parse_args()

    # Validate ratios
    if args.shared_ratio + args.candidate_ratio >= 1.0:
        raise ValueError("shared_ratio + candidate_ratio must be less than 1.0 to leave room for independent nodes")

    # Create log directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Store results for all seeds
    all_seed_results = []
    all_seed_stats = []
    
    # Run for each seed
    for seed in args.seeds:
        print(f"\n=== Running with seed {seed} ===")
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Set device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        # Create seed-specific log file
        log_file = osp.join(log_dir, f'memorization_stats_{args.dataset}_seed{seed}_{timestamp}.csv')
        
        # Load dataset with both transforms
        transform = T.Compose([
            T.NormalizeFeatures(),
            T.LargestConnectedComponents(),
        ])
        
        # Load dataset with both transforms
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'Planetoid')
        dataset = Planetoid(path, args.dataset, transform=transform)
        data = dataset[0].to(device)
        
        # Apply random node split
        transform2 = T.RandomNodeSplit(num_val=0, num_test=1000)  # This will make ~985 train nodes
        data = transform2(data)
        
        print(f"\nSplit statistics after RandomNodeSplit:")
        print(f"Number of training nodes: {data.train_mask.sum().item()}")
        print(f"Number of test nodes: {data.test_mask.sum().item()}")
        
        print(f"\nGraph stats after preprocessing:")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.edge_index.shape[1]}")
        
        # Create node splits from training nodes only
        shared_nodes, candidate_nodes, independent_nodes = create_node_splits(
            data, 
            train_ratio=args.shared_ratio, 
            candidate_ratio=args.candidate_ratio
        )
        shared_nodes = shared_nodes.to(device)
        candidate_nodes = candidate_nodes.to(device)
        independent_nodes = independent_nodes.to(device)
        
        # Create masks for f and g models
        f_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        g_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
        
        # Set training masks
        f_train_mask[torch.cat([shared_nodes, candidate_nodes])] = True
        g_train_mask[torch.cat([shared_nodes, independent_nodes])] = True
        
        # Initialize memorization scorer
        print("\nComputing memorization scores...")
        if args.model_type == 'encoder':
            scorer = MemorizationScorer(
                model_type='encoder',
                model_class=GCNEncoder,
                in_channels=dataset.num_features,
                out_channels=args.out_channels,
                device=device,
                num_epochs=args.epochs
            )
        else:  # GCN
            scorer = MemorizationScorer(
                model_type='gcn',
                in_channels=dataset.num_features,
                hidden_channels=args.hidden_channels,
                num_classes=dataset.num_classes,
                device=device,
                num_epochs=args.epochs
            )

        # After initializing the scorer
        if args.model_type == 'encoder':
            print("\nModel Configuration:")
            print("-------------------")
            print(f"Type: Encoder (GCN-based)")
            print(f"Input dimensions: {dataset.num_features}")
            print(f"Embedding dimensions: {args.out_channels}")
            print(f"Training with reconstruction loss")
            print(f"Number of epochs: {args.epochs}")
            print(f"Learning rate: {args.lr}")
        else:  # GCN
            print("\nModel Configuration:")
            print("-------------------")
            print(f"Type: GCN (Node Classification)")
            print(f"Input dimensions: {dataset.num_features}")
            print(f"Hidden dimensions: {args.hidden_channels}")
            print(f"Output dimensions: {dataset.num_classes} (num classes)")
            print(f"Training with cross-entropy loss")
            print(f"Number of epochs: {args.epochs}")
            print(f"Learning rate: {args.lr}")

        print("\nMemorization Measurement:")
        print("------------------------")
        print(f"Edge perturbation ratio: {args.edge_perturb_ratio}")
        print(f"Number of augmentations per node: 5")
        #print(f"Node split ratios: 50% shared, 30% candidates, 20% independent")

        # Create augmentor
        augmentor = GraphAugmentor(edge_perturb_ratio=args.edge_perturb_ratio)
        
        # Compute memorization scores only for candidate nodes
        all_results = []
        
        # Add progress bar for node processing
        pbar = tqdm(candidate_nodes, desc='Processing nodes')
        for node_idx in pbar:
            # Update progress bar description
            pbar.set_postfix({'node': int(node_idx)})
            
            # Train models f and g with their respective masks
            data_f = copy.deepcopy(data)
            data_g = copy.deepcopy(data)
            data_f.train_mask = f_train_mask
            data_g.train_mask = g_train_mask
            
            # Get node indices from masks
            f_train_nodes = torch.where(data_f.train_mask)[0]
            g_train_nodes = torch.where(data_g.train_mask)[0]
            
            aug_edges = augmentor.generate_edge_augmentations(data.edge_index)
            result = scorer.compute_scores_with_splits(data, node_idx, f_train_nodes, g_train_nodes)
            
            stats_df = log_graph_stats(data, aug_edges, node_idx, result, log_file, args)
            all_results.append(result)
        
        # Plot results
        print("\nPlotting results...")
        plot_alignment_memorization(all_results, save_path=f'memorization_{args.dataset}_seed{seed}.png')
        
        # Print summary statistics
        mem_scores = [r['memorization'] for r in all_results]
        print(f"\nMemorization Statistics:")
        print(f"Mean memorization score: {sum(mem_scores)/len(mem_scores):.4f}")
        print(f"Max memorization score: {max(mem_scores):.4f}")
        print(f"Min memorization score: {min(mem_scores):.4f}")
        
        # Save summary statistics
        summary_stats = {
            'dataset': args.dataset,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_nodes': data.num_nodes,
            'num_edges': data.edge_index.shape[1],
            'num_features': dataset.num_features,
            'mean_memorization': sum(mem_scores)/len(mem_scores),
            #'max_memorization': max(mem_scores),
            #'min_memorization': min(mem_scores),
            #'std_memorization': np.std(mem_scores.cpu())
        }
        
        summary_file = osp.join(log_dir, f'summary_stats_{args.dataset}_seed{seed}_{timestamp}.csv')
        pd.DataFrame([summary_stats]).to_csv(summary_file, index=False)
        print(f"\nSummary statistics saved to {summary_file}")
        
        # Store results for this seed
        seed_results = {
            'seed': seed,
            'results': all_results,
            'mem_scores': mem_scores
        }
        all_seed_results.append(seed_results)
        
        # Compute and store statistics for this seed
        seed_stats = {
            'dataset': args.dataset,
            'seed': seed,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_nodes': data.num_nodes,
            'num_edges': data.edge_index.shape[1],
            'num_features': dataset.num_features,
            'mean_memorization': sum(mem_scores)/len(mem_scores),
            'max_memorization': max(mem_scores),
            'min_memorization': min(mem_scores),
            'std_memorization': np.std(mem_scores)
        }
        all_seed_stats.append(seed_stats)
    
    # After processing all seeds, compute average results
    avg_results = []
    
    # Get all unique node indices
    all_node_indices = set()
    for seed_result in all_seed_results:
        for result in seed_result['results']:
            all_node_indices.add(result['node_idx'])
    
    # Compute averages for each node
    for node_idx in all_node_indices:
        node_results = []
        for seed_result in all_seed_results:
            # Find result for this node in this seed
            for result in seed_result['results']:
                if result['node_idx'] == node_idx:
                    node_results.append(result)
                    break
        
        # Compute average scores for this node
        avg_result = {
            'node_idx': node_idx,
            'f_alignment': np.mean([r['f_alignment'] for r in node_results]),
            'g_alignment': np.mean([r['g_alignment'] for r in node_results]),
            'memorization': np.mean([r['memorization'] for r in node_results]),
            'f_test_acc': np.mean([r['f_test_acc'] for r in node_results]),
            'g_test_acc': np.mean([r['g_test_acc'] for r in node_results])
        }
        avg_results.append(avg_result)
    
    # Plot average results
    plot_alignment_memorization(avg_results, 
                              save_path=f'memorization_{args.dataset}_averaged.png')
    
    # Print final statistics
    print("\n=== Final Statistics (Averaged across all seeds) ===")
    mean_mem_scores = [stats['mean_memorization'] for stats in all_seed_stats]
    max_mem_scores = [stats['max_memorization'] for stats in all_seed_stats]
    min_mem_scores = [stats['min_memorization'] for stats in all_seed_stats]
    std_mem_scores = [stats['std_memorization'] for stats in all_seed_stats]
    
    print(f"Mean memorization score: {np.mean(mean_mem_scores):.4f} ± {np.std(mean_mem_scores):.4f}")
    print(f"Max memorization score: {np.mean(max_mem_scores):.4f} ± {np.std(max_mem_scores):.4f}")
    print(f"Min memorization score: {np.mean(min_mem_scores):.4f} ± {np.std(min_mem_scores):.4f}")
    print(f"Average std deviation: {np.mean(std_mem_scores):.4f}")
    
    # Save final summary statistics
    final_stats = {
        'dataset': args.dataset,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_seeds': len(args.seeds),
        'mean_memorization': np.mean(mean_mem_scores),
        'std_memorization': np.std(mean_mem_scores),
        'max_memorization': np.mean(max_mem_scores),
        'min_memorization': np.mean(min_mem_scores),
        'avg_std_deviation': np.mean(std_mem_scores)
    }
    
    summary_file = osp.join(log_dir, f'final_summary_stats_{args.dataset}_{timestamp}.csv')
    pd.DataFrame([final_stats]).to_csv(summary_file, index=False)
    print(f"\nFinal summary statistics saved to {summary_file}")

if __name__ == "__main__":
    main()