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
from memorization import split_train_nodes, calculate_node_memorization_score, plot_node_memorization_analysis
from augmentation import NodeAugmentor

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
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

def train_models(args, data, dataset, shared_idx, candidate_idx, independent_idx, device, logger):
    """Train model f on shared+candidate nodes and model g on shared+independent nodes"""
    
    # Create masks for training
    def create_mask(idx, num_nodes):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = True
        return mask
    
    model_f_mask = create_mask(shared_idx + candidate_idx, data.num_nodes)
    model_g_mask = create_mask(shared_idx + independent_idx, data.num_nodes)
    
    # Get number of classes from dataset
    num_classes = dataset.num_classes
    
    # Initialize models
    model_f = NodeGCN(data.num_features, num_classes, args.hidden_dim, args.num_layers).to(device)
    model_g = NodeGCN(data.num_features, num_classes, args.hidden_dim, args.num_layers).to(device)
    
    # Optimizers
    opt_f = torch.optim.Adam(model_f.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_g = torch.optim.Adam(model_g.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    logger.info("Starting model training...")
    
    # Training loop
    for epoch in range(args.epochs):
        # Train model f (shared + candidate)
        f_loss = train(model_f, data.x, data.edge_index, model_f_mask, data.y, opt_f, device)
        f_test_acc = test(model_f, data.x, data.edge_index, data.test_mask, data.y, device)
        
        # Train model g (shared + independent)
        g_loss = train(model_g, data.x, data.edge_index, model_g_mask, data.y, opt_g, device)
        g_test_acc = test(model_g, data.x, data.edge_index, data.test_mask, data.y, device)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch {epoch+1}/{args.epochs}:')
            logger.info(f'Model f - Loss: {f_loss:.4f}, Test Acc: {f_test_acc:.4f}')
            logger.info(f'Model g - Loss: {g_loss:.4f}, Test Acc: {g_test_acc:.4f}')
    
    # Save models
    torch.save(model_f.state_dict(), 'f_model.pt')
    torch.save(model_g.state_dict(), 'g_model.pt')
    logger.info("Models saved as f_model.pt and g_model.pt")
    
    return model_f, model_g, f_test_acc, g_test_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photo', 'Actor'])
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--augmentation_type', type=str, default='flip',
                       choices=['flip', 'noise', 'both'])
    parser.add_argument('--feature_flip_rate', type=float, default=0.5)
    parser.add_argument('--noise_std', type=float, default=0.1)
    parser.add_argument('--num_augmentations', type=int, default=5)
    args = parser.parse_args()
    
    # Setup
    logger, log_dir, timestamp = setup_logging(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    dataset = load_dataset(args, seed=42)
    data = dataset[0].to(device)
    
    # Merge train and validation sets
    train_val_mask = data.train_mask | data.val_mask
    train_val_indices = torch.where(train_val_mask)[0].tolist()
    
    # Split into shared, candidate, and independent nodes
    shared_idx, candidate_idx, independent_idx = split_train_nodes(train_val_indices)
    logger.info(f"Split sizes - Shared: {len(shared_idx)}, Candidate: {len(candidate_idx)}, Independent: {len(independent_idx)}")
    
    # Phase 1: Train and save models
    model_f, model_g, f_test_acc, g_test_acc = train_models(
        args, data, dataset, shared_idx, candidate_idx, independent_idx, device, logger
    )
    
    # Phase 2: Load saved models
    logger.info("\nLoading saved models...")
    # Initialize new model instances
    model_f_eval = NodeGCN(data.num_features, dataset.num_classes, args.hidden_dim, args.num_layers).to(device)
    model_g_eval = NodeGCN(data.num_features, dataset.num_classes, args.hidden_dim, args.num_layers).to(device)
    
    # Load saved states
    model_f_eval.load_state_dict(torch.load('f_model.pt'))
    model_g_eval.load_state_dict(torch.load('g_model.pt'))
    
    logger.info("Models loaded successfully")
    
    # Set models to eval mode
    model_f_eval.eval()
    model_g_eval.eval()
    
    # Phase 3: Calculate memorization scores
    logger.info("\nCalculating memorization scores...")
    augmentor = NodeAugmentor(
        augmentation_type=args.augmentation_type,
        feature_flip_rate=args.feature_flip_rate,
        noise_std=args.noise_std,
        num_augmentations=args.num_augmentations
    )
    
    memorization_score, f_scores, g_scores, normalized_scores = calculate_node_memorization_score(
        model_f=model_f_eval,  # Use loaded models
        model_g=model_g_eval,  # Use loaded models
        candidate_nodes=candidate_idx,
        x=data.x,
        edge_index=data.edge_index,
        augmentor=augmentor,
        device=device,
        logger=logger
    )
    
    logger.info(f"Average memorization score: {memorization_score:.4f}")
    
    # Phase 4: Create visualization
    logger.info("\nGenerating visualization...")
    plot_path = os.path.join(log_dir, f'memorization_{args.dataset}_{timestamp}.png')
    plot_node_memorization_analysis(
        f_scores=f_scores,
        g_scores=g_scores,
        mem_scores=normalized_scores,
        save_path=plot_path,
        title_suffix=f"Dataset: {args.dataset}\nf_acc={f_test_acc:.3f}, g_acc={g_test_acc:.3f}"
    )
    logger.info(f"Plot saved to: {plot_path}")

if __name__ == '__main__':
    main()