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
    nodes_dict: Dict[str, List[int]],  # Dictionary of node types to their indices
    x: torch.Tensor,
    edge_index: torch.Tensor,
    augmentor,
    device: torch.device,
    logger=None
) -> Dict[str, List[float]]:
    """Calculate memorization scores for different types of nodes"""
    model_f.eval()
    model_g.eval()
    
    results = {}
    
    for node_type, nodes in nodes_dict.items():
        prob_b = []  # Store model g distances
        prob_a = []  # Store model f distances
        
        # Add tqdm progress bar
        pbar = tqdm(nodes, desc=f"Calculating memorization scores for {node_type} nodes")
        
        for node_idx in pbar:
            # Get augmented versions of the node's neighborhood
            aug_data, aug_stats = augmentor(node_idx, x, edge_index)
            
            dist_b = []  # Store g model distances for this node
            dist_a = []  # Store f model distances for this node
        
            
            # Process each augmentation
            for aug_data_i in aug_data:
                x_aug = aug_data_i['x'].to(device)
                edge_index_aug = aug_data_i['edge_index'].to(device)
                
                with torch.no_grad():
                    _, emb_g_orig = model_g(x.to(device), edge_index.to(device), return_node_emb=True)
                    emb_g_orig = emb_g_orig[node_idx:node_idx+1]
                    emb_g_orig = F.normalize(emb_g_orig, p=2, dim=1)

                    _, emb_f_orig = model_f(x.to(device), edge_index.to(device), return_node_emb=True)
                    emb_f_orig = emb_f_orig[node_idx:node_idx+1]
                    emb_f_orig = F.normalize(emb_f_orig, p=2, dim=1)
                    _, emb_g_aug = model_g(x_aug, edge_index_aug, return_node_emb=True)
                    emb_g_aug = emb_g_aug[node_idx:node_idx+1]
                    emb_g_aug = F.normalize(emb_g_aug, p=2, dim=1)
                    
                    _, emb_f_aug = model_f(x_aug, edge_index_aug, return_node_emb=True)
                    emb_f_aug = emb_f_aug[node_idx:node_idx+1]
                    emb_f_aug = F.normalize(emb_f_aug, p=2, dim=1)
                    
                    g_dist = torch.norm(emb_g_orig - emb_g_aug, p=2).item()
                    f_dist = torch.norm(emb_f_orig - emb_f_aug, p=2).item()
                    
                    dist_b.append(g_dist)
                    dist_a.append(f_dist)
            
            avg_dist_b = np.mean(dist_b)
            avg_dist_a = np.mean(dist_a)
            
            prob_b.append(avg_dist_b)
            prob_a.append(avg_dist_a)
            
            pbar.set_postfix({
                'g_dist': f'{avg_dist_b:.4f}',
                'f_dist': f'{avg_dist_a:.4f}'
            })
        
        # Convert to numpy arrays
        prob_b = np.array(prob_b)
        prob_a = np.array(prob_a)
        
        # Calculate difference scores
        diff_prob = prob_b - prob_a
        maximum = max(diff_prob)
        minimum = min(diff_prob)
        vrange = maximum - minimum
        diff_prob = diff_prob/vrange
        
        # Store additional metrics for plotting
        avg_score = np.mean(diff_prob)
        results[node_type] = {
            'mem_scores': diff_prob.tolist(),
            'f_distances': prob_a.tolist(),
            'g_distances': prob_b.tolist(),
            'avg_score': avg_score
        }
        
        if logger:
            logger.info(f"Average memorization score for {node_type} nodes: {avg_score:.4f}")
    
    return results

def plot_node_memorization_analysis(
    node_scores: Dict[str, Dict],
    save_path: str,
    title_suffix="",
):
    """Plot node memorization analysis results"""
    plt.figure(figsize=(15, 6))
    
    # Plot 1: Model comparison scatter plot (only for candidate nodes)
    plt.subplot(1, 2, 1)
    
    # Get candidate node data
    if 'candidate' in node_scores:
        f_distances = node_scores['candidate']['f_distances']
        g_distances = node_scores['candidate']['g_distances']
        mem_scores = node_scores['candidate']['mem_scores']
        
        # Add y=x line in red
        min_val = min(min(f_distances), min(g_distances))
        max_val = max(max(f_distances), max(g_distances))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='y=x')
        
        # Create scatter plot with viridis colormap
        scatter = plt.scatter(f_distances, g_distances, 
                            c=mem_scores, cmap='viridis', 
                            alpha=0.6, s=50)
        plt.colorbar(scatter, label='Memorization Score')
    
    plt.xlabel('Model f Distance')
    plt.ylabel('Model g Distance')
    plt.title('Alignment Distance Comparison (Candidate Nodes)')
    plt.legend()
    
    # Plot 2: Histogram for all node types
    plt.subplot(1, 2, 2)
    
    colors = {'shared': 'blue', 'candidate': 'red', 'independent': 'green', 'extra': 'purple'}
    
    for node_type, scores_dict in node_scores.items():
        scores = scores_dict['mem_scores']
        mean_score = scores_dict['avg_score']  # Use pre-calculated average
        plt.hist(scores, bins=20, alpha=0.5, 
                label=f'{node_type} nodes (mean={mean_score:.3f})',
                color=colors[node_type], edgecolor='black')
    
    plt.xlabel('Memorization Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Memorization Scores by Node Type')
    plt.legend()
    
    # Save plot with high DPI for better quality
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()