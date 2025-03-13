import argparse
import torch
import os
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

# Import existing functions from main.py and main_sbm.py
from main import (set_seed, train_models, verify_no_data_leakage, 
                 get_node_splits, setup_logging, get_model,
                 calculate_node_memorization_score, test)

def load_grid_graphs(data_dir):
    """Load all graphs from the grid dataset."""
    with open(os.path.join(data_dir, "sbm_list.pkl"), "rb") as f:
        graphs = pickle.load(f)
    return graphs

def analyze_correlations(results_df, logger):
    """Analyze correlations between graph properties and memorization."""
    correlations = {}
    
    # Variables to analyze
    variables = {
        'homophily': 'Graph Homophily',
        'informativeness': 'Label Informativeness',
        'percent_memorized': 'Memorization Rate',
        'f_val_acc': 'Model F Validation Accuracy',
        'g_val_acc': 'Model G Validation Accuracy'
    }
    
    logger.info("\n=== Correlation Analysis ===")
    
    # Calculate correlations between all pairs
    for var1 in variables:
        for var2 in variables:
            if var1 >= var2:  # Skip duplicates and self-correlations
                continue
                
            # Calculate Spearman correlation
            rho, p_value = spearmanr(results_df[var1], results_df[var2])
            correlations[f'spearman_{var1}_{var2}'] = (rho, p_value)
            
            # Calculate Pearson correlation
            r, p_value_pearson = pearsonr(results_df[var1], results_df[var2])
            correlations[f'pearson_{var1}_{var2}'] = (r, p_value_pearson)
            
            logger.info(f"\n{variables[var1]} vs {variables[var2]}:")
            logger.info(f"Spearman correlation: ρ = {rho:.3f} (p = {p_value:.3e})")
            logger.info(f"Pearson correlation: r = {r:.3f} (p = {p_value_pearson:.3e})")
            
            # Interpret correlation strength
            strength = "strong" if abs(rho) > 0.7 else "moderate" if abs(rho) > 0.4 else "weak"
            logger.info(f"Correlation strength: {strength}")
    
    return correlations

def create_correlation_heatmap(results_df, save_path):
    """Create a heatmap of correlations between variables."""
    variables = ['homophily', 'informativeness', 'percent_memorized', 
                'f_val_acc', 'g_val_acc']
    
    # Calculate correlation matrix
    corr_matrix = results_df[variables].corr(method='spearman')
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu', vmin=-1, vmax=1, 
                center=0, fmt='.2f', square=True)
    
    # Customize labels
    labels = ['Homophily', 'Informativeness', 'Memorization', 
              'Model F Acc', 'Model G Acc']
    plt.xticks(np.arange(len(labels)) + 0.5, labels, rotation=45, ha='right')
    plt.yticks(np.arange(len(labels)) + 0.5, labels, rotation=0)
    
    plt.title('Spearman Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_memorization_heatmap(results_df, save_path, args):
    """
    Create a 2D heatmap of memorization percentage vs homophily and informativeness.
    
    Args:
        results_df: DataFrame with columns ['homophily', 'informativeness', 'percent_memorized']
        save_path: Path to save the visualization
        args: Script arguments
    """
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(
        results_df['homophily'],
        results_df['informativeness'],
        c=results_df['percent_memorized'],
        cmap='viridis',
        s=100,  # marker size
        alpha=0.7
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Memorization Rate (%)', fontsize=12)
    
    # Customize plot
    plt.xlabel('Graph Homophily', fontsize=12)
    plt.ylabel('Label Informativeness', fontsize=12)
    plt.title(f'Memorization Analysis Grid\nModel: {args.model_type.upper()}', fontsize=14)
    
    # Add annotations for each point
    if len(results_df) <= 50:  # Only annotate if not too crowded
        for idx, row in results_df.iterrows():
            plt.annotate(
                f"{row['percent_memorized']:.1f}%",
                (row['homophily'], row['informativeness']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.8
            )
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing grid graphs')
    parser.add_argument('--model_type', type=str, default='gcn',
                       choices=['gcn', 'gat', 'graphconv'])
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--gat_heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_passes', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='results/grid_analysis')
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'grid_analysis_{args.model_type}_{timestamp}'
    log_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    logger = logging.getLogger('grid_analysis')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(log_dir, 'analysis.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load all graphs
    logger.info(f"Loading graphs from {args.data_dir}")
    graphs = load_grid_graphs(args.data_dir)
    logger.info(f"Loaded {len(graphs)} graphs")
    
    # Process each graph
    results = []
    for idx, data in enumerate(tqdm(graphs, desc="Processing graphs")):
        logger.info(f"\nProcessing graph {idx + 1}/{len(graphs)}")
        logger.info(f"Homophily: {data.homophily:.4f}")
        logger.info(f"Informativeness: {data.informativeness:.4f}")
        
        data = data.to(device)
        
        # Get node splits
        shared_idx, candidate_idx, independent_idx = get_node_splits(
            data, data.train_mask, swap_candidate_independent=False
        )
        
        # Get extra indices from test set
        test_indices = torch.where(data.test_mask)[0]
        extra_size = len(candidate_idx)
        extra_indices = test_indices[:extra_size].tolist()
        
        # Create nodes_dict
        nodes_dict = {
            'shared': shared_idx,
            'candidate': candidate_idx,
            'independent': independent_idx,
            'extra': extra_indices,
            'val': torch.where(data.val_mask)[0].tolist(),
            'test': torch.where(data.test_mask)[0].tolist()
        }
        
        # Train models
        model_f, model_g, f_val_acc, g_val_acc = train_models(
            args=args,
            data=data,
            shared_idx=shared_idx,
            candidate_idx=candidate_idx,
            independent_idx=independent_idx,
            device=device,
            logger=logger,
            output_dir=None
        )
        
        # Calculate memorization scores
        node_scores = calculate_node_memorization_score(
            model_f=model_f,
            model_g=model_g,
            data=data,
            nodes_dict=nodes_dict,
            device=device,
            logger=logger,
            num_passes=args.num_passes
        )
        
        # Store results for candidate nodes
        results.append({
            'homophily': float(data.homophily),
            'informativeness': float(data.informativeness),
            'percent_memorized': node_scores['candidate']['percentage_above_threshold'],
            'avg_memorization': node_scores['candidate']['avg_score'],
            'num_memorized': node_scores['candidate']['nodes_above_threshold'],
            'total_nodes': len(node_scores['candidate']['mem_scores']),
            'f_val_acc': float(f_val_acc),
            'g_val_acc': float(g_val_acc)
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Perform correlation analysis
    correlations = analyze_correlations(results_df, logger)
    
    # Create correlation heatmap
    heatmap_path = os.path.join(log_dir, 'correlation_heatmap.png')
    create_correlation_heatmap(results_df, heatmap_path)
    logger.info(f"\nCorrelation heatmap saved to: {heatmap_path}")
    
    # Save correlation results
    corr_df = pd.DataFrame({
        'correlation_type': [k for k in correlations.keys()],
        'value': [v[0] for v in correlations.values()],
        'p_value': [v[1] for v in correlations.values()]
    })
    corr_df.to_csv(os.path.join(log_dir, 'correlations.csv'), index=False)
    
    # Create and save visualization
    plot_path = os.path.join(log_dir, 'memorization_heatmap.png')
    create_memorization_heatmap(results_df, plot_path, args)
    
    # Save results
    results_df.to_csv(os.path.join(log_dir, 'grid_results.csv'), index=False)
    
    logger.info("\nAnalysis complete!")
    logger.info(f"Results saved to: {log_dir}")
    logger.info(f"Visualization saved as: {plot_path}")

if __name__ == '__main__':
    main()
