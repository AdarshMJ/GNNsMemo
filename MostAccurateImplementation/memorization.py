import torch
import numpy as np
from typing import List, Tuple, Dict
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.utils import k_hop_subgraph
from scipy import stats
from tqdm import tqdm

def split_train_nodes(train_val_indices: List[int], seed: int = None) -> Tuple[List[int], List[int], List[int]]:
    """
    Split the combined train+val nodes into shared, candidate, and independent sets
    Args:
        train_val_indices: List of node indices from combined train and validation sets
        seed: Random seed for splitting
    Returns:
        Tuple of (shared_indices, candidate_indices, independent_indices)
    """
    # Create a new RandomState for this split
    rng = np.random.RandomState(seed)
    
    # Shuffle the indices using the random state
    indices = train_val_indices.copy()
    rng.shuffle(indices)
    
    # Split into shared (50%), candidate (25%), independent (25%)
    num_nodes = len(indices)
    shared_size = int(0.5 * num_nodes)
    candidate_size = int(0.25 * num_nodes)
    
    shared_indices = indices[:shared_size]
    candidate_indices = indices[shared_size:shared_size + candidate_size]
    independent_indices = indices[shared_size + candidate_size:]
    
    return shared_indices, candidate_indices, independent_indices

def calculate_node_memorization_score(
    model_f, model_g, 
    candidate_nodes: List[int],
    x: torch.Tensor,
    edge_index: torch.Tensor,
    augmentor,
    device: torch.device,
    logger=None
) -> Tuple[float, List[float], List[float], List[float]]:
    """Calculate memorization scores for nodes using the reference implementation logic"""
    model_f.eval()
    model_g.eval()
    
    prob_b = []  # Store model g distances
    prob_a = []  # Store model f distances
    
    # Add tqdm progress bar
    pbar = tqdm(candidate_nodes, desc="Calculating memorization scores")
    
    for node_idx in pbar:
        # Get augmented versions of the node's neighborhood
        aug_data, aug_stats = augmentor(node_idx, x, edge_index)
        
        dist_b = []  # Store g model distances for this node
        dist_a = []  # Store f model distances for this node
        
        # Get original embeddings
        with torch.no_grad():
            # Get model g original embedding
            _, emb_g_orig = model_g(x.to(device), edge_index.to(device), return_node_emb=True)
            emb_g_orig = emb_g_orig[node_idx:node_idx+1]
            emb_g_orig = F.normalize(emb_g_orig, p=2, dim=1)
            
            # Get model f original embedding
            _, emb_f_orig = model_f(x.to(device), edge_index.to(device), return_node_emb=True)
            emb_f_orig = emb_f_orig[node_idx:node_idx+1]
            emb_f_orig = F.normalize(emb_f_orig, p=2, dim=1)
        
        # Process each augmentation
        for aug_data_i in aug_data:
            x_aug = aug_data_i['x'].to(device)
            edge_index_aug = aug_data_i['edge_index'].to(device)
            
            with torch.no_grad():
                # Get model g augmented embedding
                _, emb_g_aug = model_g(x_aug, edge_index_aug, return_node_emb=True)
                emb_g_aug = emb_g_aug[node_idx:node_idx+1]
                emb_g_aug = F.normalize(emb_g_aug, p=2, dim=1)
                
                # Get model f augmented embedding
                _, emb_f_aug = model_f(x_aug, edge_index_aug, return_node_emb=True)
                emb_f_aug = emb_f_aug[node_idx:node_idx+1]
                emb_f_aug = F.normalize(emb_f_aug, p=2, dim=1)
                
                # Calculate L2 distances
                g_dist = torch.norm(emb_g_orig - emb_g_aug, p=2).item()
                f_dist = torch.norm(emb_f_orig - emb_f_aug, p=2).item()
                
                dist_b.append(g_dist)
                dist_a.append(f_dist)
        
        # Average distances across augmentations for this node
        avg_dist_b = np.mean(dist_b)
        avg_dist_a = np.mean(dist_a)
        
        prob_b.append(avg_dist_b)
        prob_a.append(avg_dist_a)
        
        # Update progress bar with current scores
        pbar.set_postfix({
            'g_dist': f'{avg_dist_b:.4f}',
            'f_dist': f'{avg_dist_a:.4f}'
        })
    
    # Convert to numpy arrays
    prob_b = np.array(prob_b)
    prob_a = np.array(prob_a)
    
    # Calculate difference scores
    diff_prob = prob_b - prob_a
    
    # Apply min-max normalization to differences
    maximum = max(diff_prob)
    minimum = min(diff_prob)
    vrange = maximum - minimum
    diff_prob = diff_prob/vrange  # Normalize
    memorization_score = np.mean(diff_prob)
    
    if logger:
        logger.info(f"Average memorization score is: {memorization_score:.4f}")
    
    return memorization_score, prob_a.tolist(), prob_b.tolist(), diff_prob.tolist()

def plot_node_memorization_analysis(
    f_scores: List[float],
    g_scores: List[float],
    mem_scores: List[float],
    save_path: str,
    title_suffix="",
    normalization: str = 'minmax'
):
    """Plot node memorization analysis results"""
    plt.figure(figsize=(15, 6))
    
    # Debug info
    print(f"Number of scores: {len(mem_scores)}")
    print(f"Score range: {min(mem_scores):.4f} to {max(mem_scores):.4f}")
    print(f"Mean score: {np.mean(mem_scores):.4f}")
    
    # Plot 1: Scatter plot with y=x line
    plt.subplot(1, 2, 1)
    
    # Add y=x line first (so it appears behind the points)
    min_val = min(min(f_scores), min(g_scores))
    max_val = max(max(f_scores), max(g_scores))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
    
    # Plot scatter points
    scatter = plt.scatter(f_scores, g_scores, c=mem_scores, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Memorization Score')
    plt.xlabel('Model f Alignment Loss')
    plt.ylabel('Model g Alignment Loss')
    plt.title('Alignment Loss Comparison')
    plt.legend()
    
    # Plot 2: Histogram
    plt.subplot(1, 2, 2)
    plt.hist(mem_scores, bins=20, edgecolor='black')
    plt.xlabel('Memorization Score')
    plt.ylabel('Frequency')
    plt.title('Memorization Score Distribution')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()