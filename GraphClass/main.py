import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from typing import List, Tuple, Dict
import os
import logging
from model import SimpleGCN, GATv2, GraphConv
import numpy as np
import random
from datetime import datetime
import sys
from memorization import split_dataset, calculate_memorization_score, plot_memorization_analysis, plot_aggregate_memorization_analysis
from augmentation import NodeDropping

# Add seeds list and helper functions at the top
RANDOM_SEEDS = [3164711608, 894959334, 2487307261, 3349051410, 493067366]

def create_seed_directory(base_dir: str, seed: int, timestamp: str) -> str:
    """Create a directory for the specific seed run"""
    seed_dir = os.path.join(base_dir, f'seed_{seed}_{timestamp}')
    os.makedirs(seed_dir, exist_ok=True)
    return seed_dir

def aggregate_metrics(seed_metrics: List[Dict]) -> Dict:
    """Aggregate metrics across multiple seeds"""
    aggregated = {}
    for key in seed_metrics[0].keys():
        values = [m[key] for m in seed_metrics]
        aggregated[key] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    return aggregated

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.x, data.edge_index, data.batch).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

def setup_logging(args, seed_dir):
    # Modified to use seed_dir instead of current directory
    logs_dir = os.path.join(seed_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_log_filename = os.path.join(logs_dir, f'{args.dataset}_{args.model}_{timestamp}.log')
    aug_log_filename = os.path.join(logs_dir, f'{args.dataset}_{args.model}_{timestamp}_augmentation.log')
    
    main_logger = logging.getLogger(f'training_{timestamp}')  # Make logger unique
    main_logger.setLevel(logging.INFO)
    main_logger.handlers = []  # Clear any existing handlers
    
    aug_logger = logging.getLogger(f'augmentation_{timestamp}')  # Make logger unique
    aug_logger.setLevel(logging.INFO)
    aug_logger.handlers = []  # Clear any existing handlers
    
    train_file_handler = logging.FileHandler(train_log_filename)
    train_stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    train_file_handler.setFormatter(formatter)
    train_stream_handler.setFormatter(formatter)
    main_logger.addHandler(train_file_handler)
    main_logger.addHandler(train_stream_handler)
    
    aug_file_handler = logging.FileHandler(aug_log_filename)
    aug_file_handler.setFormatter(formatter)
    aug_logger.addHandler(aug_file_handler)
    
    return timestamp, main_logger, aug_logger

def create_model(args, dataset):
    if args.model.lower() == 'gcn':
        return SimpleGCN(dataset.num_features, dataset.num_classes, 
                         args.hidden_dim, args.num_layers)
    elif args.model.lower() == 'gatv2':
        return GATv2(dataset.num_features, args.hidden_dim, 
                     dataset.num_classes, args.num_layers)
    elif args.model.lower() == 'graphconv':
        return GraphConv(dataset.num_features, dataset.num_classes, 
                         args.hidden_dim, args.num_layers)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

def run_single_seed(args, seed, base_timestamp):
    """Run the experiment for a single seed"""
    set_seed(seed)
    
    # Create seed-specific directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    seed_dir = create_seed_directory(current_dir, seed, base_timestamp)
    
    # Setup logging for this seed
    timestamp, main_logger, aug_logger = setup_logging(args, seed_dir)
    
    main_logger.info(f"{'='*50}")
    main_logger.info(f"Running experiment with seed: {seed}")
    main_logger.info(f"{'='*50}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main_logger.info(f"Device: {device}")
    
    # Load dataset
    dataset = TUDataset(root='data/TUDataset', name=args.dataset)
    
    # Log dataset statistics
    main_logger.info(f"\n{'='*50}")
    main_logger.info("Dataset Statistics:")
    main_logger.info(f"{'='*50}")
    main_logger.info(f"Dataset: {args.dataset}")
    main_logger.info(f"Number of graphs: {len(dataset)}")
    main_logger.info(f"Number of features: {dataset.num_features}")
    main_logger.info(f"Number of classes: {dataset.num_classes}")
    
    # Split dataset into train, validation, and test
    shared_idx, candidate_idx, independent_idx, test_idx = split_dataset(dataset)
    
    # Create datasets for models f and g
    model_f_dataset = [dataset[i] for i in shared_idx + candidate_idx]
    model_g_dataset = [dataset[i] for i in shared_idx + independent_idx]
    candidate_graphs = [dataset[i] for i in candidate_idx]
    shared_graphs = [dataset[i] for i in shared_idx]
    independent_graphs = [dataset[i] for i in independent_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    main_logger.info(f"\nDataset splits:")
    main_logger.info(f"Total graphs: {len(dataset)}")
    main_logger.info(f"Training set ({len(shared_idx) + len(candidate_idx) + len(independent_idx)} graphs):")
    main_logger.info(f"  - Shared set: {len(shared_idx)} graphs")
    main_logger.info(f"  - Candidate set: {len(candidate_idx)} graphs")
    main_logger.info(f"  - Independent set: {len(independent_idx)} graphs")
    main_logger.info(f"Test set: {len(test_idx)} graphs")
    
    # Create data loaders
    f_loader = DataLoader(model_f_dataset, batch_size=args.batch_size, shuffle=True)
    g_loader = DataLoader(model_g_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    main_logger.info(f"\nSplit sizes:")
    main_logger.info(f"Model F training corpus: {len(model_f_dataset)}")
    main_logger.info(f"Model G training corpus: {len(model_g_dataset)}")
    main_logger.info(f"Test graphs: {len(test_dataset)}")
    
    # Initialize models f and g
    model_f = create_model(args, dataset)
    model_g = create_model(args, dataset)
    model_f, model_g = model_f.to(device), model_g.to(device)
    
    # Log model architecture
    main_logger.info(f"\n{'='*50}")
    main_logger.info("Model Architecture:")
    main_logger.info(f"{'='*50}")
    main_logger.info(model_f)
    main_logger.info(f"\nTotal number of parameters (model f): {sum(p.numel() for p in model_f.parameters())}")
    main_logger.info(model_g)
    main_logger.info(f"\nTotal number of parameters (model g): {sum(p.numel() for p in model_g.parameters())}")
    
    # Initialize optimizers
    opt_f = torch.optim.Adam(model_f.parameters(), lr=args.lr)
    opt_g = torch.optim.Adam(model_g.parameters(), lr=args.lr)
    
    # Initialize augmentor with only the required parameters
    augmentor = NodeDropping(
        drop_rate=args.drop_rate,
        num_augmentations=args.num_augmentations
    )
    
    # Training loop
    best_val_acc = 0
    best_test_acc = 0
    best_train_acc = 0
    final_epoch = 0
    
    main_logger.info(f"\n{'='*50}")
    main_logger.info("Training Progress:")
    main_logger.info(f"{'='*50}")
    
    for epoch in range(args.epochs):
        # Train and evaluate model f
        f_loss = train(model_f, f_loader, opt_f, device)
        f_acc = test(model_f, test_loader, device)
        
        # Train and evaluate model g
        g_loss = train(model_g, g_loader, opt_g, device)
        g_acc = test(model_g, test_loader, device)
        
        if (epoch + 1) % 10 == 0:
            mem_score, f_scores, g_scores, individual_scores = calculate_memorization_score(
                model_f, model_g, candidate_graphs, augmentor, device
            )
            main_logger.info(f'Epoch: {epoch+1}')
            main_logger.info(f'Model f - Loss: {f_loss:.4f}, Test Acc: {f_acc:.4f}')
            main_logger.info(f'Model g - Loss: {g_loss:.4f}, Test Acc: {g_acc:.4f}')
            main_logger.info(f'Memorization Score: {mem_score:.4f}')
    
    # Final evaluation and plotting
    final_mem_score, f_scores, g_scores, individual_scores = calculate_memorization_score(
        model_f, model_g, candidate_graphs, augmentor, device
    )
    
    # Get final metrics for this seed
    final_metrics = {
        'model_f_acc': test(model_f, test_loader, device),
        'model_g_acc': test(model_g, test_loader, device),
        'memorization_score': final_mem_score
    }
    
    # Create plots directory inside seed directory
    plots_dir = os.path.join(seed_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    plot_path = os.path.join(plots_dir, f'{args.dataset}_{args.model}_{timestamp}_memorization.png')
    plot_memorization_analysis(f_scores, g_scores, individual_scores, plot_path,
                             candidate_graphs=candidate_graphs,
                             shared_graphs=shared_graphs,
                             independent_graphs=independent_graphs,
                             model_f=model_f,
                             model_g=model_g,
                             augmentor=augmentor,
                             device=device)
    
    main_logger.info(f'\nFinal Results for seed {seed}:')
    main_logger.info(f'Model f Test Accuracy: {final_metrics["model_f_acc"]:.4f}')
    main_logger.info(f'Model g Test Accuracy: {final_metrics["model_g_acc"]:.4f}')
    main_logger.info(f'Final Memorization Score: {final_metrics["memorization_score"]:.4f}')
    main_logger.info(f'Memorization plot saved to: {plot_path}')
    
    return final_metrics

def main():
    parser = argparse.ArgumentParser(description='Train GNN models for graph classification')
    parser.add_argument('--model', type=str, required=True, help='Model type (GCN, GATv2, GraphConv)')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'NCI1', 'ENZYMES'],
                       help='Dataset name')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--drop_rate', type=float, default=0.2, 
                       help='Node dropping rate for augmentation')
    parser.add_argument('--num_augmentations', type=int, default=3,
                       help='Number of augmentations per graph')
    
    args = parser.parse_args()
    
    # Create a timestamp for the entire multi-seed experiment
    base_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Run experiments for all seeds
    all_metrics = []
    for seed in RANDOM_SEEDS:
        seed_metrics = run_single_seed(args, seed, base_timestamp)
        all_metrics.append(seed_metrics)
    
    # Aggregate metrics across seeds
    aggregated_metrics = aggregate_metrics(all_metrics)
    
    # Create main results directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_results_dir = os.path.join(current_dir, f'results_{base_timestamp}')
    os.makedirs(main_results_dir, exist_ok=True)
    
    # Save aggregated results and create aggregate plot
    aggregate_plot_path = os.path.join(main_results_dir, f'{args.dataset}_{args.model}_aggregate_memorization.png')
    plot_aggregate_memorization_analysis(all_metrics, aggregate_plot_path)
    
    with open(os.path.join(main_results_dir, 'aggregated_results.txt'), 'w') as f:
        f.write(f"Aggregated Results Across {len(RANDOM_SEEDS)} Seeds:\n")
        f.write(f"{'='*50}\n")
        for metric, values in aggregated_metrics.items():
            f.write(f"{metric}:\n")
            f.write(f"  Mean: {values['mean']:.4f}\n")
            f.write(f"  Std:  {values['std']:.4f}\n")
        f.write(f"\nIndividual Seed Results:\n")
        for seed, metrics in zip(RANDOM_SEEDS, all_metrics):
            f.write(f"\nSeed {seed}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
        f.write(f"\nAggregate plot saved to: {aggregate_plot_path}\n")

if __name__ == '__main__':
    main()