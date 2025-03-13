import argparse
import torch
from torch_geometric.transforms import RandomNodeSplit, LargestConnectedComponents
from torch_geometric.transforms import Compose
import os
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from generate_homophily import HomophilySBMDataset

# Import existing functions from main.py
from main import (set_seed, train_models, verify_no_data_leakage, 
                 get_node_splits, setup_logging, get_model,
                 calculate_node_memorization_score, test)

def analyze_sbm_graphs(args, logger):
    """Analyze all SBM graphs and collect results"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create base output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the SBM dataset based on fixed parameter
    if args.fixed_informativeness is not None:
        dataset = HomophilySBMDataset(
            root=args.dataset_path,
            informativeness=args.fixed_informativeness,
            n_graphs=args.n_graphs
        )
        logger.info(f"Dataset contains {len(dataset)} graphs with fixed informativeness {args.fixed_informativeness}")
    else:
        dataset = HomophilySBMDataset(
            root=args.dataset_path,
            homophily=args.fixed_homophily,
            n_graphs=args.n_graphs
        )
        logger.info(f"Dataset contains {len(dataset)} graphs with fixed homophily {args.fixed_homophily}")
    
    # Container for results
    results = []
    
    # Add console handler for real-time logging
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    # Process each graph
    for idx in range(len(dataset)):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing Graph {idx} of {len(dataset)-1}")
        logger.info(f"{'='*80}")
        
        data = dataset[idx].to(device)
        
        # Get homophily value, handling both tensor and float types
        h_adj = data.homophily.item() if hasattr(data.homophily, 'item') else data.homophily
        
        # Get label informativeness if available, handling both tensor and float types
        informativeness = None
        if hasattr(data, 'label_informativeness'):
            informativeness = (data.label_informativeness.item() 
                              if hasattr(data.label_informativeness, 'item') 
                              else data.label_informativeness)
        elif hasattr(data, 'informativeness'):
            informativeness = (data.informativeness.item() 
                              if hasattr(data.informativeness, 'item') 
                              else data.informativeness)
        
        # Log graph details
        logger.info("\nGraph Details:")
        logger.info(f"Adjusted Homophily: {h_adj:.3f}")
        if informativeness is not None:
            logger.info(f"Label Informativeness: {informativeness:.3f}")
        logger.info(f"Number of Nodes: {data.num_nodes}")
        logger.info(f"Number of Edges: {data.edge_index.size(1) // 2}")
        
        # Get node splits and log partition details
        shared_idx, candidate_idx, independent_idx = get_node_splits(
            data, data.train_mask, swap_candidate_independent=args.swap_nodes
        )
        
        test_indices = torch.where(data.test_mask)[0]
        extra_size = len(candidate_idx)
        extra_indices = test_indices[:extra_size].tolist()
        
        # Log partition statistics
        logger.info("\nNode Partition Statistics:")
        logger.info(f"Training nodes total: {data.train_mask.sum().item()}")
        logger.info(f"- Shared nodes: {len(shared_idx)} ({len(shared_idx)/data.num_nodes*100:.1f}%)")
        logger.info(f"- Candidate nodes: {len(candidate_idx)} ({len(candidate_idx)/data.num_nodes*100:.1f}%)")
        logger.info(f"- Independent nodes: {len(independent_idx)} ({len(independent_idx)/data.num_nodes*100:.1f}%)")
        logger.info(f"Validation nodes: {data.val_mask.sum().item()} ({data.val_mask.sum().item()/data.num_nodes*100:.1f}%)")
        logger.info(f"Test nodes: {data.test_mask.sum().item()} ({data.test_mask.sum().item()/data.num_nodes*100:.1f}%)")
        logger.info(f"Extra nodes: {len(extra_indices)} ({len(extra_indices)/data.num_nodes*100:.1f}%)")
        
        # Create nodes_dict
        nodes_dict = {
            'shared': shared_idx,
            'candidate': candidate_idx,
            'independent': independent_idx,
            'extra': extra_indices,
            'val': torch.where(data.val_mask)[0].tolist(),
            'test': torch.where(data.test_mask)[0].tolist()
        }
        
        # Verify no data leakage
        verify_no_data_leakage(shared_idx, candidate_idx, independent_idx, logger)
        
        logger.info("\nTraining Details:")
        logger.info("-" * 50)
        logger.info(f"Training Model f (shared + candidate nodes)")
        logger.info(f"Training Model g (shared + independent nodes)")
        logger.info(f"Number of epochs: {args.epochs}")
        logger.info(f"Learning rate: {args.lr}")
        logger.info(f"Hidden dimensions: {args.hidden_dim}")
        logger.info(f"Number of layers: {args.num_layers}")
        logger.info("-" * 50)
        
        # Train models with detailed logging
        logger.info("\nTraining Models...")
        model_f, model_g, f_val_acc, g_val_acc = train_models(
            args=args,
            data=data,
            shared_idx=shared_idx,
            candidate_idx=candidate_idx,
            independent_idx=independent_idx,
            device=device,
            logger=logger,
            output_dir=None  # Remove graph-specific output directory
        )
        
        # Calculate final test accuracies
        model_f.eval()
        model_g.eval()
        with torch.no_grad():
            f_test_acc = test(model_f, data.x, data.edge_index, data.test_mask, data.y, device)
            g_test_acc = test(model_g, data.x, data.edge_index, data.test_mask, data.y, device)
        
        logger.info("\nFinal Model Performance:")
        logger.info(f"Model f - Val Acc: {f_val_acc:.4f}, Test Acc: {f_test_acc:.4f}")
        logger.info(f"Model g - Val Acc: {g_val_acc:.4f}, Test Acc: {g_test_acc:.4f}")
        
        # Calculate memorization scores
        logger.info("\nMemorization Analysis:")
        logger.info("-" * 50)
        node_scores = calculate_node_memorization_score(
            model_f=model_f,
            model_g=model_g,
            data=data,
            nodes_dict=nodes_dict,
            device=device,
            logger=logger,
            num_passes=args.num_passes
        )
        
        # Detailed memorization statistics
        logger.info("\nDetailed Memorization Statistics:")
        logger.info("-" * 50)
        for node_type, scores in node_scores.items():
            logger.info(f"\n{node_type.capitalize()} Nodes Analysis:")
            logger.info(f"  Total nodes: {len(scores['mem_scores'])}")
            logger.info(f"  Average memorization score: {scores['avg_score']:.4f}")
            logger.info(f"  Memorized nodes (score > 0.5): {scores['nodes_above_threshold']} ({scores['percentage_above_threshold']:.1f}%)")
            logger.info(f"  Average f confidence: {np.mean(scores['f_confidences']):.4f}")
            logger.info(f"  Average g confidence: {np.mean(scores['g_confidences']):.4f}")
            
            # Distribution statistics
            mem_scores = np.array(scores['mem_scores'])
            logger.info(f"  Score distribution:")
            logger.info(f"    Min: {np.min(mem_scores):.4f}")
            logger.info(f"    Max: {np.max(mem_scores):.4f}")
            logger.info(f"    Median: {np.median(mem_scores):.4f}")
            logger.info(f"    Std: {np.std(mem_scores):.4f}")
            
            # Score brackets
            brackets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
            logger.info(f"  Score distribution by brackets:")
            for low, high in brackets:
                count = np.sum((mem_scores >= low) & (mem_scores < high))
                percentage = (count / len(mem_scores)) * 100
                logger.info(f"    {low:.1f}-{high:.1f}: {count} nodes ({percentage:.1f}%)")
        
        logger.info("\nSummary:")
        logger.info(f"Adjusted Homophily: {h_adj:.3f}")
        logger.info(f"Label Informativeness: {informativeness:.3f}")
        logger.info(f"Model f - Val Acc: {f_val_acc:.4f}, Test Acc: {f_test_acc:.4f}")
        logger.info(f"Model g - Val Acc: {g_val_acc:.4f}, Test Acc: {g_test_acc:.4f}")
        logger.info(f"Candidate nodes memorized: {node_scores['candidate']['nodes_above_threshold']}/{len(node_scores['candidate']['mem_scores'])} ({node_scores['candidate']['percentage_above_threshold']:.1f}%)")
        
        logger.info(f"\nCompleted processing Graph {idx}")
        logger.info("="*80 + "\n")
        
        # Collect results for candidate nodes
        for node_type, scores in node_scores.items():
            if node_type == 'candidate':
                results.append({
                    'graph_idx': idx,
                    'homophily': h_adj,
                    'informativeness': informativeness if informativeness is not None else np.nan,
                    'avg_memorization': scores['avg_score'],
                    'num_memorized': scores['nodes_above_threshold'],
                    'total_nodes': len(scores['mem_scores']),
                    'percent_memorized': scores['percentage_above_threshold'],
                    'f_val_acc': float(f_val_acc),  # Ensure float conversion
                    'g_val_acc': float(g_val_acc),  # Ensure float conversion
                    'f_test_acc': float(f_test_acc),  # Ensure float conversion
                    'g_test_acc': float(g_test_acc)  # Ensure float conversion
                })
    
    # Remove console handler before returning
    logger.removeHandler(console_handler)
    
    # Convert results to DataFrame and ensure accuracy values are properly formatted
    results_df = pd.DataFrame(results)
    for col in ['f_val_acc', 'g_val_acc', 'f_test_acc', 'g_test_acc']:
        if col in results_df.columns:
            results_df[col] = results_df[col].astype(float)
    
    return results_df  # Return DataFrame instead of list

def plot_informativeness_vs_memorized_count(results_df, save_path):
    """
    Plot the relationship between label informativeness and number of candidate nodes
    with memorization score > 0.5
    """
    # Sort dataframe by informativeness for better visualization
    df = results_df.sort_values('informativeness')
    
    plt.figure(figsize=(12, 8))
    
    # Scatter plot with size proportional to number of memorized nodes
    scatter = plt.scatter(
        df['informativeness'], 
        df['num_memorized'],
        s=80,  # Base point size
        c=df['percent_memorized'],  # Color by memorization percentage instead of informativeness
        cmap='viridis',  # Colormap
        alpha=0.8,
    )
    
    # Label points with graph indices
    for i, row in df.iterrows():
        plt.annotate(
            f"{int(row['graph_idx'])}", 
            (row['informativeness'], row['num_memorized']),
            textcoords="offset points",
            xytext=(0,7),
            ha='center',
            fontsize=9,
        )
    
    # Connect points with a line
    plt.plot(df['informativeness'], df['num_memorized'], '-o', alpha=0.4, color='gray')
    
    # Add horizontal reference line at 0
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Label Informativeness', fontsize=14)
    plt.ylabel('Number of Memorized Nodes\n(Memorization Score > 0.5)', fontsize=14)
    plt.title(f'Effect of Label Informativeness on Node Memorization\n(Fixed Homophily = {df["homophily"].iloc[0]:.2f})', 
              fontsize=16)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add colorbar with memorization percentage label
    cbar = plt.colorbar(scatter)
    cbar.set_label('Memorization Rate (%)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def plot_homophily_vs_memorized_count(results_df, save_path):
    """
    Plot the relationship between homophily and number of candidate nodes
    with memorization score > 0.5 (for fixed informativeness datasets)
    """
    # Sort dataframe by homophily for better visualization
    df = results_df.sort_values('homophily')
    
    plt.figure(figsize=(12, 8))
    
    # Scatter plot with size proportional to number of memorized nodes
    scatter = plt.scatter(
        df['homophily'], 
        df['num_memorized'],
        s=80,  # Base point size
        c=df['percent_memorized'],  # Color by memorization percentage
        cmap='viridis',  # Colormap
        alpha=0.8,
    )
    
    # Label points with graph indices
    for i, row in df.iterrows():
        plt.annotate(
            f"{int(row['graph_idx'])}", 
            (row['homophily'], row['num_memorized']),
            textcoords="offset points",
            xytext=(0,7),
            ha='center',
            fontsize=9,
        )
    
    # Connect points with a line
    plt.plot(df['homophily'], df['num_memorized'], '-o', alpha=0.4, color='gray')
    
    # Add horizontal reference line at 0
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('Graph Homophily', fontsize=14)
    plt.ylabel('Number of Memorized Nodes\n(Memorization Score > 0.5)', fontsize=14)
    plt.title(f'Effect of Homophily on Node Memorization\n(Fixed Informativeness = {df["informativeness"].iloc[0]:.2f})', 
              fontsize=16)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add colorbar with memorization percentage label
    cbar = plt.colorbar(scatter)
    cbar.set_label('Memorization Rate (%)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def plot_memorization_statistics(results_df, save_path, fixed_param='homophily'):
    """
    Create a visualization showing memorization statistics across the dataset.
    
    Args:
        results_df: DataFrame with results
        save_path: Path to save the plot
        fixed_param: Which parameter is fixed ('homophily' or 'informativeness')
    """
    # Determine the x-axis parameter (the one that varies)
    if fixed_param == 'homophily':
        x_param = 'informativeness'
        fixed_value = results_df['homophily'].iloc[0]
        title_prefix = f'Fixed Homophily = {fixed_value:.2f}'
    else:
        x_param = 'homophily'
        fixed_value = results_df['informativeness'].iloc[0]
        title_prefix = f'Fixed Informativeness = {fixed_value:.2f}'
    
    df = results_df.sort_values(x_param)
    
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 14), height_ratios=[3, 2])
    
    # Plot 1: Number of memorized nodes vs x_param
    axs[0].bar(
        np.arange(len(df)), 
        df['num_memorized'], 
        width=0.6,
        color=[plt.cm.viridis(x) for x in np.linspace(0, 1, len(df))],
        alpha=0.8
    )
    
    # Add labels on top of each bar
    for i, v in enumerate(df['num_memorized']):
        axs[0].text(i, v + 1, f"{int(v)}", ha='center', fontsize=10)
    
    # Add x_param values as x-tick labels
    axs[0].set_xticks(np.arange(len(df)))
    axs[0].set_xticklabels([f"{x:.3f}" for x in df[x_param]], rotation=45, ha='right')
    
    axs[0].set_xlabel(x_param.title(), fontsize=12)
    axs[0].set_ylabel('Number of Memorized Nodes\n(Score > 0.5)', fontsize=12)
    axs[0].set_title(f'Effect of {x_param.title()} on Node Memorization\n({title_prefix})', 
                   fontsize=14)
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 2: Model performance vs x_param
    x = np.arange(len(df))
    width = 0.2
    
    # Plot model accuracies
    axs[1].bar(x - width*1.5, df['f_val_acc']*100, width, label='Model f - Val', alpha=0.7, color='skyblue')
    axs[1].bar(x - width*0.5, df['g_val_acc']*100, width, label='Model g - Val', alpha=0.7, color='lightgreen')
    axs[1].bar(x + width*0.5, df['f_test_acc']*100, width, label='Model f - Test', alpha=0.7, color='orange')
    axs[1].bar(x + width*1.5, df['g_test_acc']*100, width, label='Model g - Test', alpha=0.7, color='salmon')
    
    # Add memorization percentage as a line
    ax2 = axs[1].twinx()
    ax2.plot(x, df['percent_memorized'], 'o-', color='purple', linewidth=2, label='% Memorized')
    ax2.set_ylabel('Memorization (%)', fontsize=12, color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    
    # Set labels and legend
    axs[1].set_xlabel(x_param.title(), fontsize=12)
    axs[1].set_ylabel('Model Accuracy (%)', fontsize=12)
    axs[1].set_title(f'Model Performance vs {x_param.title()}', fontsize=14)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([f"{i:.3f}" for i in df[x_param]], rotation=45, ha='right')
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Combine both legends
    lines1, labels1 = axs[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs[1].legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='gcn',
                       choices=['gcn', 'gat', 'graphconv'])
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--gat_heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--swap_nodes', action='store_true')
    parser.add_argument('--num_passes', type=int, default=1)
    
    # Graph generation parameters - either homophily or informativeness must be provided
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--fixed_homophily', type=float, default=None,
                      help='Fix homophily at this value and vary label informativeness')
    group.add_argument('--fixed_informativeness', type=float, default=None,
                      help='Fix informativeness at this value and vary homophily')
                      
    parser.add_argument('--n_graphs', type=int, default=10,
                      help='Number of graphs to analyze')
    parser.add_argument('--dataset_path', type=str, default='data/pyg_sbm',
                      help='Path to the directory containing generated SBM datasets')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Ensure either fixed_homophily or fixed_informativeness is provided
    if args.fixed_homophily is None and args.fixed_informativeness is None:
        args.fixed_homophily = 0.5  # Default to homophily 0.5 if neither is specified
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging and output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.fixed_informativeness is not None:
        exp_name = f'fixed_info_{args.fixed_informativeness}'
    else:
        exp_name = f'fixed_h_{args.fixed_homophily}'
    
    base_dir = f'results/sbm_analysis_{args.model_type}_{exp_name}_{timestamp}'
    os.makedirs(base_dir, exist_ok=True)
    args.output_dir = base_dir
    
    logger = logging.getLogger('sbm_analysis')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(base_dir, 'analysis.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    
    # Log script parameters
    logger.info("SBM Analysis Parameters:")
    for arg in vars(args):
        logger.info(f"  {arg}: {getattr(args, arg)}")
    
    # Run analysis
    results_df = analyze_sbm_graphs(args, logger)
    
    # Create and save plots
    if not results_df.empty:
        if args.fixed_informativeness is not None:
            # For fixed informativeness datasets, plot homophily vs memorization
            count_plot_path = os.path.join(base_dir, 'homophily_vs_memorized_count.png')
            plot_homophily_vs_memorized_count(results_df, count_plot_path)
            
            # Create and save the statistics plot
            stats_plot_path = os.path.join(base_dir, 'memorization_statistics.png')
            plot_memorization_statistics(results_df, stats_plot_path, fixed_param='informativeness')
        else:
            # For fixed homophily datasets, plot informativeness vs memorization
            count_plot_path = os.path.join(base_dir, 'informativeness_vs_memorized_count.png')
            plot_informativeness_vs_memorized_count(results_df, count_plot_path)
            
            # Create and save the statistics plot
            stats_plot_path = os.path.join(base_dir, 'memorization_statistics.png')
            plot_memorization_statistics(results_df, stats_plot_path, fixed_param='homophily')
        
        # Save results to CSV
        results_df.to_csv(os.path.join(base_dir, 'results.csv'), index=False)
        
        logger.info("\nAnalysis complete! Results saved to:")
        logger.info(f"- Log file: {os.path.join(base_dir, 'analysis.log')}")
        logger.info(f"- Results CSV: {os.path.join(base_dir, 'results.csv')}")
        logger.info(f"- Memorization count plot: {count_plot_path}")
        logger.info(f"- Statistics plot: {stats_plot_path}")
    else:
        logger.info("\nNo results generated. Analysis failed.")

if __name__ == '__main__':
    main()
