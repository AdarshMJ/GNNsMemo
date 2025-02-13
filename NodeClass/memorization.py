import torch
import numpy as np
from typing import List, Tuple, Dict
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.utils import k_hop_subgraph
from scipy import stats

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
    logger=None,
    normalization: str = 'ratio'  # 'ratio' or 'minmax'
) -> Tuple[float, List[float], List[float], List[float]]:
    """Calculate memorization scores for nodes using specified normalization"""
    mem_scores = []
    f_scores = []
    g_scores = []
    diff_scores = []  # To store g_score - f_score before normalization
    
    for node_idx in candidate_nodes:
        # Get augmented versions of the node's neighborhood
        aug_data, aug_stats = augmentor(node_idx, x, edge_index)
        
        if logger:
            logger.info(f"\n{'='*50}")
            logger.info(f"Augmentation Details for Node {node_idx}")
            logger.info(f"{'='*50}")
            
            for aug_idx, stats in enumerate(aug_stats):
                logger.info(f"\nAugmentation {aug_idx + 1}:")
                logger.info(f"Augmentation types: {', '.join(stats['augmentation_types'])}")
                
                # Log flip stats if present
                if 'flip_stats' in stats:
                    flip_stats = stats['flip_stats']
                    logger.info("\nFeature Flip Statistics:")
                    logger.info(f"Number of features flipped: {flip_stats['num_features_flipped']}")
                    logger.info(f"Flip rate used: {flip_stats['flip_rate_used']}")
                    logger.info("\nFeature changes:")
                    for change in flip_stats['feature_changes']:
                        logger.info(f"Index {change['index']}: {change['original']} -> {change['new']}")
                
                # Log noise stats if present
                if 'noise_stats' in stats:
                    noise_stats = stats['noise_stats']
                    logger.info("\nGaussian Noise Statistics:")
                    logger.info(f"Noise std: {noise_stats['noise_std']}")
                    logger.info(f"Max noise magnitude: {noise_stats['max_noise_magnitude']:.4f}")
                    logger.info(f"Average noise magnitude: {noise_stats['avg_noise_magnitude']:.4f}")
                    logger.info("\nSignificant feature changes:")
                    for change in noise_stats['feature_changes'][:5]:  # Show first 5 changes
                        logger.info(f"Index {change['index']}: {change['original']:.4f} -> {change['new']:.4f} (noise: {change['noise']:.4f})")
                    if len(noise_stats['feature_changes']) > 5:
                        logger.info(f"... and {len(noise_stats['feature_changes']) - 5} more changes")
                
                # Log general node statistics
                node_stats = stats['node_feature_stats']
                logger.info(f"\nNode feature statistics:")
                logger.info(f"Total features: {node_stats['total_features']}")
                logger.info(f"Number of 1s before: {node_stats['total_ones_before']}")
                logger.info(f"Number of 1s after: {node_stats['total_ones_after']}")
        
        # Get original embedding (no augmentation)
        with torch.no_grad():
            _, emb_f_orig = model_f(x.to(device), edge_index.to(device), return_node_emb=True)
            _, emb_g_orig = model_g(x.to(device), edge_index.to(device), return_node_emb=True)
            
            # Get embeddings for the target node
            emb_f_orig = emb_f_orig[node_idx:node_idx+1]
            emb_g_orig = emb_g_orig[node_idx:node_idx+1]
            
            # Normalize original embeddings
            emb_f_orig = F.normalize(emb_f_orig, p=2, dim=1)
            emb_g_orig = F.normalize(emb_g_orig, p=2, dim=1)
        
        f_dists = []
        g_dists = []
        
        # Compare each augmentation to original
        for aug_data_i in aug_data:
            x_aug = aug_data_i['x'].to(device)
            edge_index_aug = aug_data_i['edge_index'].to(device)
            
            with torch.no_grad():
                _, emb_f_aug = model_f(x_aug, edge_index_aug, return_node_emb=True)
                _, emb_g_aug = model_g(x_aug, edge_index_aug, return_node_emb=True)
                
                # Get embeddings for the target node
                emb_f_aug = emb_f_aug[node_idx:node_idx+1]
                emb_g_aug = emb_g_aug[node_idx:node_idx+1]
            
                # Normalize augmented embeddings
                emb_f_aug = F.normalize(emb_f_aug, p=2, dim=1)
                emb_g_aug = F.normalize(emb_g_aug, p=2, dim=1)
                
                # Calculate distances from original
                f_dist = torch.norm(emb_f_aug - emb_f_orig, p=2).item()
                g_dist = torch.norm(emb_g_aug - emb_g_orig, p=2).item()
                
                f_dists.append(f_dist)
                g_dists.append(g_dist)
        
        # Average distances from original for each model
        f_score = np.mean(f_dists) if f_dists else 0.0
        g_score = np.mean(g_dists) if g_dists else 0.0
        
        f_scores.append(f_score)
        g_scores.append(g_score)
        diff_scores.append(g_score - f_score)
    
    # Apply normalization based on specified method
    if normalization == 'ratio':
        # Original ratio-based normalization (-1 to 1)
        mem_scores = [(g - f) / (g + f + 1e-8) for f, g in zip(f_scores, g_scores)]
        final_mem_score = np.mean(mem_scores)
    else:  # minmax normalization like vision implementation
        # Min-max normalization (0 to 1)
        maximum = max(diff_scores)
        minimum = min(diff_scores)
        vrange = maximum - minimum + 1e-8  # avoid division by zero
        mem_scores = [(score - minimum) / vrange for score in diff_scores]
        final_mem_score = np.mean(mem_scores)
    
    return final_mem_score, f_scores, g_scores, mem_scores

def plot_node_memorization_analysis(
    f_scores: List[float],
    g_scores: List[float],
    mem_scores: List[float],
    save_path: str,
    shared_scores: List[float] = None,
    independent_scores: List[float] = None,
    title_suffix="",
    normalization: str = 'ratio'
):
    """Plot node memorization analysis results with detailed visualizations"""
    plt.figure(figsize=(15, 6))  # Adjusted figure size for 2 plots
    
    # Plot 1: Scatter plot of alignment scores
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(f_scores, g_scores, c=mem_scores, 
                         cmap='viridis', alpha=0.4, s=20)
    
    # Add y=x line
    max_val = max(max(f_scores), max(g_scores))
    min_val = min(min(f_scores), min(g_scores))
    plt.plot([min_val, max_val], [min_val, max_val], 
             'k--', alpha=0.5, label='Equal Alignment')
    
    plt.colorbar(scatter, label=f'Memorization Score ({normalization} norm)')
    plt.xlabel('Model f Alignment Loss')
    plt.ylabel('Model g Alignment Loss')
    plt.title(f'Memorization Scores {title_suffix}')
    plt.legend()
    
    # Plot 2: Memorization score histogram
    plt.subplot(1, 2, 2)
    if shared_scores is not None and independent_scores is not None:
        # Calculate common bins for all scores
        all_scores = np.concatenate([mem_scores, shared_scores, independent_scores])
        min_score, max_score = min(all_scores), max(all_scores)
        bins = np.linspace(min_score, max_score, 30)  # 30 bins across range
        
        # Plot histograms
        plt.hist(mem_scores, bins=bins, alpha=0.6, label=f'Candidate ({len(mem_scores)} nodes)', 
                color='blue', density=False)
        plt.hist(shared_scores, bins=bins, alpha=0.6, label=f'Shared ({len(shared_scores)} nodes)', 
                color='green', density=False)
        plt.hist(independent_scores, bins=bins, alpha=0.6, label=f'Independent ({len(independent_scores)} nodes)', 
                color='red', density=False)
    
    # Add vertical line at 0 for ratio normalization or 0.5 for minmax
    if normalization == 'ratio':
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    else:
        plt.axvline(x=0.5, color='k', linestyle='--', alpha=0.5)
    
    plt.xlabel(f'Memorization Score ({normalization} norm)')
    plt.ylabel('Frequency')
    plt.title(f'Memorization Score Distribution\nHistogram shows raw counts {title_suffix}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()