import torch
import numpy as np
from typing import List, Tuple, Dict
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.utils import k_hop_subgraph
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def verify_node_access(node_idx: int, node_type: str, nodes_dict: Dict[str, List[int]]):
    """Verify that node belongs to the correct set"""
    if node_idx not in nodes_dict[node_type]:
        raise ValueError(f"Node {node_idx} is being processed as {node_type} but not found in that set!")

def calculate_node_memorization_score(
    model_f, model_g, 
    nodes_dict: Dict[str, List[int]],  # Dictionary of node types to their indices
    x: torch.Tensor,
    edge_index: torch.Tensor,
    augmentor,
    device: torch.device,
    embedding_layer: int = None,  # New parameter to specify which layer's embeddings to use
    logger=None
) -> Dict[str, List[float]]:
    """
    Calculate memorization scores for different types of nodes
    
    Args:
        model_f: First model trained on shared+candidate nodes
        model_g: Second model trained on shared+independent nodes
        nodes_dict: Dictionary mapping node types to their indices
        x: Node feature matrix
        edge_index: Edge index matrix
        augmentor: Node augmentation object
        device: Device to run computations on
        embedding_layer: Which layer's embeddings to use (0-based index, None means last hidden layer)
        logger: Logger object for printing progress
    """
    model_f.eval()
    model_g.eval()
    
    # Verify no overlap between candidate and independent sets
    candidate_set = set(nodes_dict['candidate'])
    independent_set = set(nodes_dict['independent'])
    if candidate_set.intersection(independent_set):
        raise ValueError("Data leakage detected! Overlap between candidate and independent sets")
    
    results = {}
    
    for node_type, nodes in nodes_dict.items():
        prob_b = []  # Store model g distances
        prob_a = []  # Store model f distances
        
        # Add tqdm progress bar
        pbar = tqdm(nodes, desc=f"Calculating memorization scores for {node_type} nodes")
        
        for node_idx in pbar:
            # Verify node access is valid
            verify_node_access(node_idx, node_type, nodes_dict)
            
            # Get augmented versions of the node's neighborhood
            aug_data, aug_stats = augmentor(node_idx, x, edge_index)
            
            dist_b = []  # Store g model distances for this node
            dist_a = []  # Store f model distances for this node
        
            
            # Process each augmentation
            for aug_data_i in aug_data:
                x_aug = aug_data_i['x'].to(device)
                edge_index_aug = aug_data_i['edge_index'].to(device)
                model_f.eval()
                model_g.eval()
    
                with torch.no_grad():
                    # Get embeddings from specified layer for model g
                    _, emb_g_orig = model_g(x.to(device), edge_index.to(device), 
                                          return_node_emb=True, embedding_layer=embedding_layer)
                    emb_g_orig = emb_g_orig[node_idx:node_idx+1]
                    #emb_g_orig = F.normalize(emb_g_orig, p=2, dim=1)
                    
                    # Get embeddings from specified layer for model f
                    _, emb_f_orig = model_f(x.to(device), edge_index.to(device), 
                                          return_node_emb=True, embedding_layer=embedding_layer)
                    emb_f_orig = emb_f_orig[node_idx:node_idx+1]
                    #emb_f_orig = F.normalize(emb_f_orig, p=2, dim=1)
                    
                    # Get embeddings for augmented input
                    _, emb_g_aug = model_g(x_aug, edge_index_aug, 
                                         return_node_emb=True, embedding_layer=embedding_layer)
                    emb_g_aug = emb_g_aug[node_idx:node_idx+1]
                    #emb_g_aug = F.normalize(emb_g_aug, p=2, dim=1)
                    
                    _, emb_f_aug = model_f(x_aug, edge_index_aug, 
                                         return_node_emb=True, embedding_layer=embedding_layer)
                    emb_f_aug = emb_f_aug[node_idx:node_idx+1]
                    #emb_f_aug = F.normalize(emb_f_aug, p=2, dim=1)
                    
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
            'avg_score': avg_score,
            'embedding_layer': embedding_layer  # Store which layer was used
        }
        
        if logger:
            logger.info(f"Average memorization score for {node_type} nodes (layer {embedding_layer}): {avg_score:.4f}")
    
    return results

def plot_node_memorization_analysis(
    node_scores: Dict[str, Dict],
    save_path: str,
    title_suffix="",
    node_types_to_plot: List[str] = None  # Optional list of node types to plot
):
    """
    Plot node memorization analysis results
    Args:
        node_scores: Dictionary containing scores for each node type
        save_path: Path to save the plot
        title_suffix: Additional text to add to plot titles
        node_types_to_plot: List of node types to include in histogram (e.g., ['shared', 'candidate'])
                          If None, all node types will be plotted
    """
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
    
    # Plot 2: Histogram for specified node types
    plt.subplot(1, 2, 2)

    # Color and label definitions
    colors = {'candidate': 'blue', 'independent': 'orange', 'extra': 'green', 'shared': 'red'}
    labels = {'candidate': '$S_C$', 'independent': '$S_I$', 'extra': '$S_E$', 'shared': '$S_S$'}

    # If no specific types are provided, plot all available types
    if node_types_to_plot is None:
        node_types_to_plot = list(node_scores.keys())

    # Plot KDE for each specified node type
    for node_type in node_types_to_plot:
        if node_type in node_scores:
            scores = node_scores[node_type]['mem_scores']
            mean_score = node_scores[node_type]['avg_score']
            
            # Use seaborn's kdeplot for smooth density estimation
            sns.kdeplot(
                scores, 
                fill=True, 
                color=colors[node_type],
                alpha=0.5,
                label=labels[node_type],
                common_norm=False,  # Each distribution normalized to its own peak
                bw_adjust=0.8,      # Adjust bandwidth for smoothness
            )

    # Set up plot appearance
    plt.xlabel('Memorization Score')
    plt.ylabel('Density')
    title = 'Distribution of Memorization Scores'
    if node_types_to_plot != list(node_scores.keys()):
        title += f" ({', '.join(node_types_to_plot)})"
    plt.title(title)
    plt.xlim(-1.0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    # Save the complete figure with both subplots
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()