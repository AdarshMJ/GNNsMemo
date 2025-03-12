import torch
import torch.nn.functional as F
import numpy as np
import copy
from scipy.stats import entropy, mannwhitneyu
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
from torch_geometric.data import Data

def calculate_delta_entropy(model, data: Data, noise_level: float = 1.0, num_samples: int = 10, device=None):
    """
    Calculate delta entropy (reliability score) for all nodes
    Averages over multiple noise samples to reduce variance
    Args:
        model: The GNN model to evaluate
        data: PyG Data object containing graph data
        noise_level: Standard deviation of Gaussian noise
        num_samples: Number of noise samples to average over
        device: Device to run computations on
    Returns:
        delta_entropy: Average entropy differences across noise samples
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    
    # Get original predictions and their entropies
    with torch.no_grad():
        out_clean = model(data.x.to(device), data.edge_index.to(device))
        probs_clean = F.softmax(out_clean, dim=-1).cpu().numpy()
        entropy_clean = np.array([entropy(p) for p in probs_clean])
        
        # Initialize array to store entropy differences
        all_delta_entropies = []
        
        # Multiple noise samples
        for _ in range(num_samples):
            # Add noise and get new predictions
            feats_noise = copy.deepcopy(data.x)
            feats_noise += torch.randn_like(feats_noise) * noise_level
            data_noise = Data(x=feats_noise, edge_index=data.edge_index).to(device)
            
            out_noise = model(data_noise.x, data_noise.edge_index)
            probs_noise = F.softmax(out_noise, dim=-1).cpu().numpy()
            entropy_noise = np.array([entropy(p) for p in probs_noise])
            
            # Calculate entropy difference
            delta_entropy = np.abs(entropy_noise - entropy_clean)
            all_delta_entropies.append(delta_entropy)
        
        # Average across samples
        avg_delta_entropy = np.mean(all_delta_entropies, axis=0)
        std_delta_entropy = np.std(all_delta_entropies, axis=0)
        
    return avg_delta_entropy, std_delta_entropy

def analyze_memorization_reliability(model_f, model_g, data, node_scores, noise_level=0.1, num_samples=10, device=None):
    """
    Analyze relationship between memorization scores and reliability (delta entropy)
    Uses model-specific delta entropy scores based on training sets
    """
    # Calculate delta entropy for both models
    delta_entropy_f, std_f = calculate_delta_entropy(model_f, data, noise_level, num_samples, device)
    delta_entropy_g, std_g = calculate_delta_entropy(model_g, data, noise_level, num_samples, device)
    
    results = {}
    for node_type, scores in node_scores.items():
        if node_type in ['val', 'test']:
            continue
            
        node_data = scores['raw_data']
        memorized_mask = node_data['mem_score'] > 0.5
        indices = node_data['node_idx'].values
        
        # Select appropriate delta entropy scores based on node type
        if node_type == 'candidate':
            node_entropy_scores = delta_entropy_f[indices]
            node_entropy_stds = std_f[indices]
        elif node_type == 'independent':
            node_entropy_scores = delta_entropy_g[indices]
            node_entropy_stds = std_g[indices]
        else:
            node_entropy_scores = (delta_entropy_f[indices] + delta_entropy_g[indices]) / 2
            node_entropy_stds = np.sqrt((std_f[indices]**2 + std_g[indices]**2) / 4)
        
        # Split into memorized vs non-memorized
        mem_entropy = node_entropy_scores[memorized_mask]
        mem_stds = node_entropy_stds[memorized_mask]
        non_mem_entropy = node_entropy_scores[~memorized_mask]
        non_mem_stds = node_entropy_stds[~memorized_mask]
        
        # Calculate correlation only if we have data
        if len(node_entropy_scores) > 0:
            correlation = np.corrcoef(node_data['mem_score'].values, node_entropy_scores)[0,1]
        else:
            correlation = np.nan

        # Perform statistical test only if both groups have data
        if len(mem_entropy) > 0 and len(non_mem_entropy) > 0:
            stat, pvalue = mannwhitneyu(mem_entropy, non_mem_entropy, alternative='two-sided')
        else:
            stat, pvalue = None, None
            
        results[node_type] = {
            'delta_entropy': node_entropy_scores,
            'delta_entropy_std': node_entropy_stds,
            'memorized_entropy': mem_entropy,
            'memorized_entropy_std': mem_stds,
            'non_memorized_entropy': non_mem_entropy,
            'non_memorized_entropy_std': non_mem_stds,
            'memorization_scores': node_data['mem_score'].values,
            'stat_test': {'statistic': stat, 'pvalue': pvalue},
            'correlation': correlation
        }
        
    return results

def plot_memorization_reliability_analysis(results, save_path: str):
    """Create visualizations for memorization vs reliability analysis"""
    n_types = len(results)
    fig = plt.figure(figsize=(15, 5 * n_types))
    
    colors = {
        'Memorized': '#FF9999',     # Light red
        'Non-memorized': '#66B2FF'  # Light blue
    }
    
    for idx, (node_type, data) in enumerate(results.items(), 1):
        # Create two subplots for each node type
        
        # 1. Scatter plot showing correlation
        plt.subplot(n_types, 2, 2*idx-1)
        scatter = plt.scatter(data['memorization_scores'], 
                            data['delta_entropy'],
                            c=data['memorization_scores'] > 0.5,
                            cmap=ListedColormap([colors['Non-memorized'], colors['Memorized']]),
                            alpha=0.6)
        
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Memorization Threshold')
        plt.xlabel('Memorization Score')
        plt.ylabel('Delta Entropy (Higher = Less Reliable)')
        
        model_type = 'Model f' if node_type == 'candidate' else \
                    'Model g' if node_type == 'independent' else \
                    'Models f+g Average'
        
        plt.title(f'{node_type.capitalize()} Nodes - Memorization vs Reliability ({model_type})\n' +
                 f'Correlation: {np.corrcoef(data["memorization_scores"], data["delta_entropy"])[0,1]:.3f}')
        plt.grid(True, alpha=0.3)
        plt.legend(handles=scatter.legend_elements()[0], 
                  labels=['Non-memorized', 'Memorized'])
        
        # 2. Box plot comparing distributions
        plt.subplot(n_types, 2, 2*idx)
        plot_data = [data['memorized_entropy'], data['non_memorized_entropy']]
        
        bp = plt.boxplot(plot_data, positions=[1, 2], widths=0.6,
                        patch_artist=True, showfliers=False)
        
        # Customize box plots
        for box, color in zip(bp['boxes'], colors.values()):
            box.set(facecolor=color, alpha=0.8)
            
        # Add individual points
        for i, (scores, pos) in enumerate(zip([data['memorized_entropy'], 
                                             data['non_memorized_entropy']], [1, 2])):
            if len(scores) > 0:
                plt.scatter([pos] * len(scores), scores,
                          alpha=0.4, color=list(colors.values())[i],
                          s=30, zorder=3)
        
        # Add statistical annotation if available
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
                max_y = max(np.max(data['memorized_entropy']), 
                          np.max(data['non_memorized_entropy']))
                y_pos = max_y + 0.05
                plt.plot([1, 1, 2, 2], [y_pos, y_pos + 0.02, y_pos + 0.02, y_pos],
                        color='black', lw=1.5)
                plt.text(1.5, y_pos + 0.03, significance,
                        ha='center', va='bottom')
        
        plt.title(f'{node_type.capitalize()} Nodes - Reliability Distribution ({model_type})\n' +
                 f'(Memorized: n={len(data["memorized_entropy"])}, ' +
                 f'μ={np.mean(data["memorized_entropy"]):.3f} | ' +
                 f'Non-memorized: n={len(data["non_memorized_entropy"])}, ' +
                 f'μ={np.mean(data["non_memorized_entropy"]):.3f})')
        plt.ylabel('Delta Entropy (Higher = Less Reliable)')
        plt.xticks([1, 2], ['Memorized', 'Non-memorized'])
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        if data['stat_test']['pvalue'] is not None:
            plt.text(0.98, 0.02, f'p-value: {data["stat_test"]["pvalue"]:.2e}',
                    transform=plt.gca().transAxes,
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.suptitle('Memorization vs Reliability Analysis\n' +
                 'Higher Delta Entropy indicates less reliable predictions under noise',
                 fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary DataFrame with better handling of edge cases
    summary_data = []
    for node_type, data in results.items():
        mem_entropy = data['memorized_entropy']
        non_mem_entropy = data['non_memorized_entropy']
        
        model_type = 'Model f' if node_type == 'candidate' else \
                    'Model g' if node_type == 'independent' else \
                    'Models f+g Average'

        # Handle edge cases gracefully
        mem_mean = np.mean(mem_entropy) if len(mem_entropy) > 0 else "No memorized nodes"
        non_mem_mean = np.mean(non_mem_entropy) if len(non_mem_entropy) > 0 else "No non-memorized nodes"
        
        # Format correlation and p-value with appropriate messages
        correlation = data['correlation']
        pvalue = data['stat_test']['pvalue']
        
        corr_str = f"{correlation:.3f}" if not np.isnan(correlation) else "N/A"
        pvalue_str = f"{pvalue:.3e}" if pvalue is not None else "N/A (insufficient data)"
        
        summary_data.append({
            'Node Type': node_type,
            'Model Used': model_type,
            'Memorized Nodes Mean Entropy': mem_mean,
            'Non-memorized Nodes Mean Entropy': non_mem_mean,
            'Memorized Nodes Count': len(mem_entropy),
            'Non-memorized Nodes Count': len(non_mem_entropy),
            'Correlation (Mem Score vs Entropy)': corr_str,
            'P-value': pvalue_str
        })
    
    return pd.DataFrame(summary_data)