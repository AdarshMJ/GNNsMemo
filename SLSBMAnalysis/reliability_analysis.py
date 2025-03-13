import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
import pandas as pd
from scipy.stats import mannwhitneyu

def kd_retention(model, data: Data, noise_level: float = 0.1, device=None):
    """
    Calculate reliability scores using KD retention through entropy differences
    Args:
        model: The GNN model to evaluate
        data: PyG Data object containing graph data
        noise_level: Standard deviation of Gaussian noise (default: 0.1)
        device: Device to run computations on
    Returns:
        delta_entropy: Normalized entropy differences as reliability scores
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    # Get original predictions and entropies
    with torch.no_grad():
        out_teacher = model(data.x.to(device), data.edge_index.to(device))
        data_teacher = F.softmax(out_teacher, dim=-1).cpu().numpy()
        weight_t = np.array([entropy(dt) for dt in data_teacher])
        
        # Add Gaussian noise to features
        feats_noise = copy.deepcopy(data.x)
        feats_noise += torch.randn_like(feats_noise) * noise_level
        data_noise = Data(x=feats_noise, edge_index=data.edge_index).to(device)
        
        # Get predictions on noisy data
        out_noise = model(data_noise.x, data_noise.edge_index)
        out_noise = F.softmax(out_noise, dim=-1).cpu().numpy()
        
        # Calculate entropy differences
        weight_s = np.abs(np.array([entropy(on) for on in out_noise]) - weight_t)
        delta_entropy = weight_s / np.max(weight_s)
    
    return delta_entropy

def analyze_reliability_vs_memorization(
    model_f,
    model_g,
    data,
    node_scores,
    noise_level: float = 0.1,
    device=None
):
    """
    Analyze relationship between node reliability and memorization scores
    Args:
        model_f: First GNN model
        model_g: Second GNN model
        data: PyG Data object
        node_scores: Dictionary containing memorization scores
        noise_level: Noise level for reliability calculation
        device: Device to run computations on
    Returns:
        Dictionary containing analysis results
    """
    # Calculate reliability scores for both models
    reliability_f = kd_retention(model_f, data, noise_level, device)
    reliability_g = kd_retention(model_g, data, noise_level, device)
    
    results = {}
    
    for node_type, scores in node_scores.items():
        if node_type in ['val', 'test']:
            continue
            
        node_data = scores['raw_data']
        memorized_mask = node_data['mem_score'] > 0.5
        
        # Get reliability scores for nodes
        rel_f = [reliability_f[idx] for idx in node_data['node_idx']]
        rel_g = [reliability_g[idx] for idx in node_data['node_idx']]
        
        # Calculate average reliability
        rel_scores = np.mean([rel_f, rel_g], axis=0)
        
        # Split into memorized vs non-memorized
        mem_rel = rel_scores[memorized_mask]
        non_mem_rel = rel_scores[~memorized_mask]
        
        # Perform statistical test
        if len(mem_rel) > 0 and len(non_mem_rel) > 0:
            stat, pvalue = mannwhitneyu(mem_rel, non_mem_rel, alternative='two-sided')
        else:
            stat, pvalue = None, None
            
        results[node_type] = {
            'reliability_scores': rel_scores,
            'memorized_reliability': mem_rel,
            'non_memorized_reliability': non_mem_rel,
            'stat_test': {
                'statistic': stat,
                'pvalue': pvalue
            }
        }
    
    return results

def plot_reliability_analysis(results, save_path: str):
    """Create visualization comparing reliability scores between memorized and non-memorized nodes"""
    n_types = len(results)
    fig = plt.figure(figsize=(15, 5 * n_types))
    
    colors = {
        'Memorized': '#FF9999',     # Light red
        'Non-memorized': '#66B2FF'  # Light blue
    }
    
    for idx, (node_type, data) in enumerate(results.items(), 1):
        plt.subplot(n_types, 1, idx)
        
        mem_scores = data['memorized_reliability']
        non_mem_scores = data['non_memorized_reliability']
        
        # Create violin plots
        plot_data = [mem_scores, non_mem_scores]
        
        # Add box plots
        bp = plt.boxplot(plot_data, positions=[1, 2], widths=0.6,
                        patch_artist=True, showfliers=False)
        
        # Customize box plots
        for box, color in zip(bp['boxes'], colors.values()):
            box.set(facecolor=color, alpha=0.8)
        
        # Add individual points
        for i, (scores, pos) in enumerate(zip([mem_scores, non_mem_scores], [1, 2])):
            if len(scores) > 0:
                plt.scatter([pos] * len(scores), scores,
                          alpha=0.4, color=list(colors.values())[i],
                          s=30, zorder=3)
        
        # Add statistical annotation
        if data['stat_test']['pvalue'] is not None:
            pvalue = data['stat_test']['pvalue']
            significance = ''
            if pvalue < 0.001:
                significance = '***'
            elif pvalue < 0.01:
                significance = '**'
            elif pvalue < 0.05:
                significance = '*'
            
            if significance:
                max_y = max(np.max(mem_scores), np.max(non_mem_scores))
                y_pos = max_y + 0.05
                plt.plot([1, 1, 2, 2], [y_pos, y_pos + 0.02, y_pos + 0.02, y_pos],
                        color='black', lw=1.5)
                plt.text(1.5, y_pos + 0.03, significance,
                        ha='center', va='bottom')
        
        plt.title(f'{node_type.capitalize()} Nodes - Reliability Score Distribution\n' +
                 f'(Memorized: n={len(mem_scores)}, μ={np.mean(mem_scores):.3f} | ' +
                 f'Non-memorized: n={len(non_mem_scores)}, μ={np.mean(non_mem_scores):.3f})')
        plt.ylabel('Reliability Score (1 - Normalized Entropy Difference)')
        plt.xticks([1, 2], ['Memorized', 'Non-memorized'])
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        if data['stat_test']['pvalue'] is not None:
            plt.text(0.98, 0.02, f'p-value: {data["stat_test"]["pvalue"]:.2e}',
                    transform=plt.gca().transAxes,
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.suptitle('Reliability Analysis: Comparing Memorized vs Non-memorized Nodes\n' +
                 'Higher scores indicate better reliability under noise perturbations',
                 fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return pd.DataFrame({
        'Node Type': list(results.keys()),
        'Mean Memorized Reliability': [np.mean(data['memorized_reliability']) for data in results.values()],
        'Mean Non-memorized Reliability': [np.mean(data['non_memorized_reliability']) for data in results.values()],
        'P-value': [data['stat_test']['pvalue'] for data in results.values()]
    })