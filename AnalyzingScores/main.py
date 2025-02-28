import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, Actor
from torch_geometric.transforms import RandomNodeSplit, LargestConnectedComponents
from torch_geometric.transforms import Compose
import os
import logging
from model import NodeGCN, NodeGAT, NodeGraphConv
import numpy as np
import random
from datetime import datetime
import sys
from memorization import calculate_node_memorization_score, plot_node_memorization_analysis
from augmentation import NodeAugmentor
from analysis import run_post_hoc_analysis

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(42)
def train(model, x, edge_index, train_mask, y, optimizer, device):
    model.train()
    optimizer.zero_grad()
    out = model(x.to(device), edge_index.to(device))
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, x, edge_index, mask, y, device):
    model.eval()
    with torch.no_grad():
        out = model(x.to(device), edge_index.to(device))
        pred = out[mask].max(1)[1]
        correct = pred.eq(y[mask]).sum().item()
        total = mask.sum().item()
    return correct / total

def setup_logging(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('results', args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup main logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # File handler
    aug_rate = args.feature_flip_rate if args.augmentation_type in ['flip', 'both'] else args.noise_std
    log_file = os.path.join(log_dir, f'{args.dataset}_{args.embedding_layer}emblayer{args.augmentation_type}_{aug_rate}_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    

    # warning_log_file = os.path.join(log_dir, f'{args.dataset}_{args.embedding_layer}emblayer_warnings_{timestamp}.log')
    # warning_file_handler = logging.FileHandler(warning_log_file)
    # warning_file_handler.setLevel(logging.WARNING)
    # warning_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # logger.addHandler(warning_file_handler)

    return logger, log_dir, timestamp

def load_dataset(args, seed):
    transforms = Compose([
        LargestConnectedComponents(),
        RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2)
    ])
    
    if args.dataset.lower() in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root='data', name=args.dataset, transform=transforms)
    elif args.dataset.lower() in ['computers', 'photo']:
        dataset = Amazon(root='data', name=args.dataset, transform=transforms)
    elif args.dataset.lower() == 'actor':
        dataset = Actor(root='data/Actor', transform=transforms)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    dataset._data_list = None
    return dataset

def get_model(model_type, num_features, num_classes, hidden_dim, num_layers, gat_heads=4):
    """Create a new model instance based on specified type"""
    if model_type.lower() == 'gcn':
        return NodeGCN(num_features, num_classes, hidden_dim, num_layers)
    elif model_type.lower() == 'gat':
        return NodeGAT(num_features, num_classes, hidden_dim, num_layers, heads=gat_heads)
    elif model_type.lower() == 'graphconv':
        return NodeGraphConv(num_features, num_classes, hidden_dim, num_layers)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def verify_no_data_leakage(shared_idx, candidate_idx, independent_idx, f_train_mask, g_train_mask, logger):
    """Verify there is no direct overlap between candidate and independent sets"""
    # Convert to sets for easy comparison
    candidate_set = set(candidate_idx)
    independent_set = set(independent_idx)
    
    # Check: No overlap between candidate and independent sets
    overlap = candidate_set.intersection(independent_set)
    if overlap:
        raise ValueError(f"Data leakage detected! Found {len(overlap)} nodes in both candidate and independent sets")
    
    logger.info("\nData Leakage Check:")
    logger.info(f"✓ No overlap between candidate and independent sets")

def train_models(args, data, dataset, shared_idx, candidate_idx, independent_idx, device, logger):
    """Train model f on shared+candidate nodes and model g on shared+independent nodes"""
    
    # Create training masks for both models
    f_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    g_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    
    # Set True for respective training nodes
    f_train_mask[shared_idx + candidate_idx] = True
    g_train_mask[shared_idx + independent_idx] = True
    
    # Verify no data leakage before training
    verify_no_data_leakage(shared_idx, candidate_idx, independent_idx, f_train_mask, g_train_mask, logger)
    
    # Print training set information
    logger.info("\nTraining Set Information:")
    logger.info(f"Model f training on {f_train_mask.sum().item()} nodes:")
    logger.info(f"- {len(shared_idx)} shared nodes: {shared_idx[:5]}...")
    logger.info(f"- {len(candidate_idx)} candidate nodes: {candidate_idx[:5]}...")
    
    logger.info(f"\nModel g training on {g_train_mask.sum().item()} nodes:")
    logger.info(f"- {len(shared_idx)} shared nodes: {shared_idx[:5]}...")
    logger.info(f"- {len(independent_idx)} independent nodes: {independent_idx[:5]}...")
    
    # Get number of classes from dataset
    num_classes = dataset.num_classes
    
    # Lists to store models and their accuracies
    f_models = []
    g_models = []
    f_accs = []
    g_accs = []
    
    # Seeds for multiple training runs
    training_seeds = [42, 123, 456]
    
    logger.info("\nModel Architecture Details:")
    logger.info(f"Model Type: {args.model_type.upper()}")
    logger.info(f"Input Features: {data.num_features}")
    logger.info(f"Hidden Dimensions: {args.hidden_dim}")
    logger.info(f"Number of Layers: {args.num_layers}")
    if args.model_type == 'gat':
        logger.info(f"Number of Attention Heads: {args.gat_heads}")
    logger.info(f"Output Classes: {num_classes}")
    logger.info(f"Training with seeds: {training_seeds}")
    
    # Train multiple models with different seeds
    for seed in training_seeds:
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        logger.info(f"\nTraining with seed {seed}")
        
        # Initialize models
        model_f = get_model(args.model_type, data.num_features, num_classes, 
                           args.hidden_dim, args.num_layers, args.gat_heads).to(device)
        model_g = get_model(args.model_type, data.num_features, num_classes, 
                           args.hidden_dim, args.num_layers, args.gat_heads).to(device)
        
        # Optimizers with different learning rates for accuracy matching
        opt_f = torch.optim.Adam(model_f.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        opt_g = torch.optim.Adam(model_g.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        best_f_acc = 0
        best_g_acc = 0
        best_f_state = None
        best_g_state = None
        
        # Training loop with dynamic learning rate adjustment
        for epoch in range(args.epochs):
            # Train model f (shared + candidate)
            f_loss = train(model_f, data.x, data.edge_index, f_train_mask, data.y, opt_f, device)
            f_test_acc = test(model_f, data.x, data.edge_index, data.test_mask, data.y, device)
            
            # Train model g (shared + independent)
            g_loss = train(model_g, data.x, data.edge_index, g_train_mask, data.y, opt_g, device)
            g_test_acc = test(model_g, data.x, data.edge_index, data.test_mask, data.y, device)
            
            # Save best models
            if f_test_acc > best_f_acc:
                best_f_acc = f_test_acc
                best_f_state = model_f.state_dict()
            
            if g_test_acc > best_g_acc:
                best_g_acc = g_test_acc
                best_g_state = model_g.state_dict()
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(f'Seed {seed}, Epoch {epoch+1}/{args.epochs}:')
                logger.info(f'Model f - Loss: {f_loss:.4f}, Test Acc: {f_test_acc:.4f}')
                logger.info(f'Model g - Loss: {g_loss:.4f}, Test Acc: {g_test_acc:.4f}')
        
        # Load best states
        model_f.load_state_dict(best_f_state)
        model_g.load_state_dict(best_g_state)
        
        # Store models and accuracies
        f_models.append(model_f.state_dict())
        g_models.append(model_g.state_dict())
        f_accs.append(best_f_acc)
        g_accs.append(best_g_acc)
        
        logger.info(f"\nSeed {seed} Results:")
        logger.info(f"Best Model f Test Accuracy: {best_f_acc:.4f}")
        logger.info(f"Best Model g Test Accuracy: {best_g_acc:.4f}")
    
    # Calculate average accuracies
    avg_f_acc = np.mean(f_accs)
    avg_g_acc = np.mean(g_accs)
    std_f_acc = np.std(f_accs)
    std_g_acc = np.std(g_accs)
    
    logger.info("\nOverall Results:")
    logger.info(f"Model f - Avg Test Acc: {avg_f_acc:.4f} ± {std_f_acc:.4f}")
    logger.info(f"Model g - Avg Test Acc: {avg_g_acc:.4f} ± {std_g_acc:.4f}")
    
    # Create ensemble models (take the one closest to mean accuracy)
    f_best_idx = np.argmin(np.abs(np.array(f_accs) - avg_f_acc))
    g_best_idx = np.argmin(np.abs(np.array(g_accs) - avg_g_acc))
    
    # Save best models
    torch.save(f_models[f_best_idx], 'f_model.pt')
    torch.save(g_models[g_best_idx], 'g_model.pt')
    logger.info("\nSaved models closest to mean accuracy:")
    logger.info(f"Model f - Test Acc: {f_accs[f_best_idx]:.4f}")
    logger.info(f"Model g - Test Acc: {g_accs[g_best_idx]:.4f}")
    
    # Return the models with accuracies closest to the mean
    model_f = get_model(args.model_type, data.num_features, num_classes, 
                       args.hidden_dim, args.num_layers, args.gat_heads).to(device)
    model_g = get_model(args.model_type, data.num_features, num_classes, 
                       args.hidden_dim, args.num_layers, args.gat_heads).to(device)
    
    model_f.load_state_dict(f_models[f_best_idx])
    model_g.load_state_dict(g_models[g_best_idx])
    
    return model_f, model_g, f_accs[f_best_idx], g_accs[g_best_idx]

def get_node_splits(data, train_val_mask, swap_candidate_independent=False):
    """
    Create node splits without shuffling to preserve natural ordering.
    
    Args:
        data: PyG data object
        train_val_mask: Mask for train+val nodes
        swap_candidate_independent: If True, swap the roles of candidate and independent nodes
    """
    # Get train+val indices in their original order
    train_val_indices = torch.where(train_val_mask)[0]
    
    # Calculate sizes
    num_nodes = len(train_val_indices)
    shared_size = int(0.50 * num_nodes)
    remaining = num_nodes - shared_size
    split_size = remaining // 2
    
    # Split indices sequentially without shuffling
    shared_idx = train_val_indices[:shared_size].tolist()
    original_candidate_idx = train_val_indices[shared_size:shared_size + split_size].tolist()
    original_independent_idx = train_val_indices[shared_size + split_size:shared_size + split_size * 2].tolist()
    
    # Return swapped indices if requested
    if swap_candidate_independent:
        return shared_idx, original_independent_idx, original_candidate_idx
    else:
        return shared_idx, original_candidate_idx, original_independent_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'Actor'])
    parser.add_argument('--model_type', type=str, default='gcn',
                       choices=['gcn', 'gat', 'graphconv'],
                       help='Type of GNN model to use')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--gat_heads', type=int, default=8,
                       help='Number of attention heads for GAT')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--augmentation_type', type=str, default='flip',
                       choices=['flip', 'noise', 'both'])
    parser.add_argument('--feature_flip_rate', type=float, default=0.1)
    parser.add_argument('--noise_std', type=float, default=0.1)
    parser.add_argument('--num_augmentations', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--embedding_layer', type=int, default=0,
                       help='Which layer to use for embeddings (0-based index, None means last hidden layer)')
    parser.add_argument('--swap_nodes', action='store_true', 
                       help='Swap candidate and independent nodes')
    args = parser.parse_args()
    
    # Setup
    logger, log_dir, timestamp = setup_logging(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    dataset = load_dataset(args, seed=42)
    data = dataset[0].to(device)
    
    # Log augmentation configuration
    logger.info("\nAugmentation Configuration:")
    logger.info(f"Augmentation Type: {args.augmentation_type}")
    logger.info(f"Number of Augmentations: {args.num_augmentations}")
    if args.augmentation_type in ['flip', 'both']:
        logger.info(f"Feature Flip Rate: {args.feature_flip_rate}")
    if args.augmentation_type in ['noise', 'both']:
        logger.info(f"Noise Standard Deviation: {args.noise_std}")
        
    # Log dataset information
    logger.info(f"\nDataset Information:")
    logger.info(f"Dataset Name: {args.dataset}")
    logger.info(f"Number of Nodes: {data.num_nodes}")
    logger.info(f"Number of Edges: {data.edge_index.size(1)}")
    logger.info(f"Number of Features: {data.num_features}")
    logger.info(f"Number of Classes: {dataset.num_classes}")
    
    # Combine train and validation masks
    #train_val_mask = data.train_mask | data.val_mask
    train_val_mask = data.train_mask
    
    # Get node splits
    shared_idx, candidate_idx, independent_idx = get_node_splits(
        data, train_val_mask, swap_candidate_independent=args.swap_nodes
    )
    
    # Get extra indices from test set
    test_indices = torch.where(data.test_mask)[0]
    extra_size = len(candidate_idx)
    extra_indices = test_indices[:extra_size].tolist()  # Take first extra_size test indices

    logger.info("\nPartition Statistics:")
    if args.swap_nodes:
        logger.info("Note: Candidate and Independent nodes have been swapped!")
        logger.info("Original independent nodes are now being used as candidate nodes")
        logger.info("Original candidate nodes are now being used as independent nodes")
    logger.info(f"Total train+val nodes: {train_val_mask.sum().item()}")
    logger.info(f"Shared: {len(shared_idx)} nodes")
    logger.info(f"Candidate: {len(candidate_idx)} nodes")
    logger.info(f"Independent: {len(independent_idx)} nodes")
    logger.info(f"Extra test nodes: {len(extra_indices)} nodes")
    logger.info(f"Test set: {data.test_mask.sum().item()} nodes")
    
    # Phase 1: Train and save models
    model_f, model_g, f_test_acc, g_test_acc = train_models(
        args, data, dataset, shared_idx, candidate_idx, independent_idx, device, logger
    )
    
    # Phase 2: Load saved models
    logger.info("\nLoading saved models...")
    # Initialize new model instances
    model_f_eval = get_model(args.model_type, data.num_features, dataset.num_classes, 
                            args.hidden_dim, args.num_layers, args.gat_heads).to(device)
    model_g_eval = get_model(args.model_type, data.num_features, dataset.num_classes, 
                            args.hidden_dim, args.num_layers, args.gat_heads).to(device)
    
    # Load saved states
    model_f_eval.load_state_dict(torch.load('f_model.pt', weights_only=True))
    model_g_eval.load_state_dict(torch.load('g_model.pt', weights_only=True))
    
    logger.info("Models loaded successfully")
    
    # Set models to eval mode
    model_f_eval.eval()
    model_g_eval.eval()
    
    # Phase 3: Calculate memorization scores
    logger.info("\nCalculating memorization scores...")
    layer_str = f"Layer {args.embedding_layer}" if args.embedding_layer is not None else "Last hidden layer"
    logger.info(f"\nUsing embeddings from: {layer_str}")
    augmentor = NodeAugmentor(
        augmentation_type=args.augmentation_type,
        feature_flip_rate=args.feature_flip_rate,
        noise_std=args.noise_std,
        num_augmentations=args.num_augmentations
    )
    
    # Create dictionary of node types
    nodes_dict = {
        'shared': shared_idx,
        'candidate': candidate_idx,
        'independent': independent_idx,
        'extra': extra_indices
    }
    
    # Calculate memorization scores for all node types
    node_scores = calculate_node_memorization_score(
        model_f=model_f_eval,
        model_g=model_g_eval,
        nodes_dict=nodes_dict,
        x=data.x,
        edge_index=data.edge_index,
        augmentor=augmentor,
        device=device,
        embedding_layer=args.embedding_layer,  # Pass the embedding layer argument
        logger=logger
    )
    
    # Log which layer was used for embeddings
 
    
    # Calculate and log average scores for each node type
    for node_type, scores_dict in node_scores.items():
        logger.info(f"Average memorization score for {node_type} nodes: {scores_dict['avg_score']:.4f}")
    
    # Phase 4: Create basic memorization score visualization
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    aug_rate = args.feature_flip_rate if args.augmentation_type in ['flip', 'both'] else args.noise_std
    plot_filename = f'{args.dataset}_{args.embedding_layer}emblayer{args.augmentation_type}_{aug_rate}_{timestamp}.png'
    plot_path = os.path.join(log_dir, plot_filename)
    
    plot_node_memorization_analysis(
        node_scores=node_scores,
        save_path=plot_path,
        title_suffix=f"Dataset: {args.dataset}\nf_acc={f_test_acc:.3f}, g_acc={g_test_acc:.3f}",
        node_types_to_plot=['shared', 'candidate','extra']
    )
    logger.info(f"Basic memorization plot saved to: {plot_path}")
    
    # Phase 5: Run post-hoc analyses
    logger.info("\nRunning post-hoc analyses...")
    aug_rate = args.feature_flip_rate if args.augmentation_type in ['flip', 'both'] else args.noise_std
    analysis_dir = os.path.join(log_dir, f'Analysis_{args.dataset}_{args.embedding_layer}emblayer_{args.augmentation_type}_{aug_rate}_{timestamp}')
    analysis_results = run_post_hoc_analysis(
        edge_index=data.edge_index,
        nodes_dict=nodes_dict,
        memorization_scores=node_scores,
        output_dir=analysis_dir,
        node_labels=data.y  # Pass node labels for NLI analysis
    )
    
    # Log analysis results
    logger.info("\nAnalysis Results:")
    logger.info(f"- Visualization saved to: {analysis_results['visualization_path']}")
    logger.info(f"- Score clustering correlation: {analysis_results['clustering']['correlation']:.3f}")
    logger.info(f"- Score clustering p-value: {analysis_results['clustering']['p_value']:.3f}")
    
    # Log NLI analysis results if available
    if 'nli_analysis' in analysis_results:
        logger.info("\nNode Label Informativeness Analysis:")
        for node_type, results in analysis_results['nli_analysis'].items():
            if node_type != 'correlations':
                if 'avg_nli' in results:
                    logger.info(f"\n{node_type} nodes:")
                    logger.info(f"  Average NLI score: {results['avg_nli']:.3f}")
        
        logger.info("\nNLI-Memorization Correlations:")
        for node_type, corr in analysis_results['nli_analysis']['correlations'].items():
            logger.info(f"{node_type}:")
            logger.info(f"  Correlation: {corr['correlation']:.3f}")
            logger.info(f"  P-value: {corr['p_value']:.3f}")

    # Log structural property results
    logger.info("\nStructural Properties Analysis:")
    for node_type, metrics in analysis_results['structural_properties'].items():
        logger.info(f"\n{node_type} nodes:")
        for metric, stats in metrics.items():
            logger.info(f"  {metric}:")
            logger.info(f"    Mean: {stats['mean']:.3f}")
            logger.info(f"    Std:  {stats['std']:.3f}")
            logger.info(f"    Median: {stats['median']:.3f}")
    
    # Log statistical test results
    logger.info("\nStatistical Test Results:")
    for comparison, stats in analysis_results['statistical_tests']['pairwise_ttests'].items():
        logger.info(f"\n{comparison}:")
        logger.info(f"  t-statistic: {stats['t_statistic']:.3f}")
        logger.info(f"  p-value: {stats['p_value']:.3f}")
        logger.info(f"  effect size (Cohen's d): {stats['effect_size']:.3f}")
        logger.info(f"  mean difference: {stats['mean_diff']:.3f}")
    
    # Log distance effects
    avg_scores = analysis_results['distance_effects']['avg_scores']
    logger.info("\nMemorization score decay with distance from candidate nodes:")
    for dist, score in avg_scores.items():
        if not np.isnan(score):
            logger.info(f"  Distance {dist}: {score:.3f}")

if __name__ == '__main__':
    main()