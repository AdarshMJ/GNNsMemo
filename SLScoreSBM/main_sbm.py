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
    
    # Load the SBM dataset
    dataset = HomophilySBMDataset(root='data/SBM')
    
    # Add debug information
    logger.info(f"Dataset contains {len(dataset)} graphs")
    logger.info(f"Homophily levels: {[dataset[i].homophily.item() for i in range(len(dataset))]}")
    
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
        
        # Create directory for this graph
        graph_dir = os.path.join(args.output_dir, f'graph_{idx}')
        os.makedirs(graph_dir, exist_ok=True)
        
        data = dataset[idx].to(device)
        h_adj = data.homophily.item()
        
        # Get label informativeness if available
        informativeness = data.label_informativeness.item() if hasattr(data, 'label_informativeness') else None
        
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
            output_dir=graph_dir
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
        results_df[col] = results_df[col].astype(float)
    
    return results_df  # Return DataFrame instead of list

def plot_homophily_memorization_relationship(results, save_path):
    """Create plots showing relationship between homophily and memorization"""
    df = pd.DataFrame(results)
    
    # Sort dataframe by homophily for better visualization
    df = df.sort_values('homophily')
    
    # If informativeness is available, create the informativeness plot
    has_informativeness = 'informativeness' in df.columns and not df['informativeness'].isna().all()
    
    if has_informativeness:
        # Create a figure for the focused informativeness plot
        plt.figure(figsize=(10, 8))
        
        # Create the main plot: Homophily vs Informativeness with memorization as color
        scatter = plt.scatter(df['homophily'], df['informativeness'], 
                             c=df['percent_memorized'], cmap='viridis',
                             s=150, alpha=0.7)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Memorization Rate (%)')
        plt.xlabel('Adjusted Homophily', fontsize=14)
        plt.ylabel('Label Informativeness', fontsize=14)
        plt.title('Homophily vs Informativeness\n(color = memorization rate)', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        # Add text labels for each point showing graph index
        for i, row in df.iterrows():
            plt.text(row['homophily'], row['informativeness'], 
                    f"Graph {int(row['graph_idx'])}", 
                    fontsize=9, ha='center', va='bottom')
        
        # Add correlation information
        # corr_info = (f'Correlations:\n' ++
        #             f'- Homophily vs Memorization: {df["homophily"].corr(df["percent_memorized"]):.3f}\n' ++
        #             f'- Informativeness vs Memorization: {df["informativeness"].corr(df["percent_memorized"]):.3f}\n' ++
        #             f'- Homophily vs Informativeness: {df["homophily"].corr(df["informativeness"]):.3f}'))
        # plt.annotate(corr_info, xy=(0.02, 0.02), xycoords='figure fraction', 
        #             fontsize=12, bbox=dict(facecolor='white', alpha=0.8)))
        
        # Save the plot
        plt.tight_layout()
        info_plot_path = save_path.replace('.png', '_informativeness.png')
        plt.savefig(info_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add path info to return value so main function knows about this plot
        df.attrs['info_plot_path'] = info_plot_path
    
    return df

def plot_homophily_vs_memorization_frequency(results, save_path):
    """Create a frequency plot showing homophily vs number of memorized nodes"""
    df = pd.DataFrame(results)
    
    # Sort dataframe by homophily for better visualization
    df = df.sort_values('homophily')
    
    # Create the frequency plot
    plt.figure(figsize=(10, 8))
    
    # Bar plot showing count of memorized nodes per homophily level
    bars = plt.bar(df['homophily'], df['num_memorized'], width=0.05, color='skyblue', alpha=0.7)
    
    # Add a line connecting the tops of the bars to show trend
    plt.plot(df['homophily'], df['num_memorized'], 'ro-', alpha=0.7)
    
    # Add labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    # Calculate correlation
    #correlation = df['homophily'].corr(df['num_memorized'])
    
    # Add title and labels
    plt.title('Frequency of Nodes with Memorization Score > 0.5\nby Adjusted Homophily Level', fontsize=16)
    plt.xlabel('Adjusted Homophily', fontsize=14)
    plt.ylabel('Number of Nodes with Memorization Score > 0.5', fontsize=14)
    
    # Add correlation annotation
    #plt.annotate(f'Correlation: {correlation:.3f}', xy=(0.02, 0.95), xycoords='axes fraction',
     #           fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add percentage on top of each bar (secondary y-axis)
    #ax2 = plt.gca().twinx()
    #ax2.plot(df['homophily'], df['percent_memorized'], 'g--', alpha=0.7)
    #ax2.set_ylabel('Percentage of Memorized Nodes', color='green', fontsize=14)
    #ax2.tick_params(axis='y', colors='green')
    
    # Add grid and tight layout
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
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
    
    args = parser.parse_args()
    
    # Setup logging and output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = f'results/sbm_analysis_{args.model_type}_{timestamp}'
    os.makedirs(base_dir, exist_ok=True)
    args.output_dir = base_dir
    
    logger = logging.getLogger('sbm_analysis')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(base_dir, 'analysis.log'))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    
    # Run analysis
    results = analyze_sbm_graphs(args, logger)
    
    # Create and save plots
    plot_path = os.path.join(base_dir, 'homophily_memorization_relationship.png')
    df = plot_homophily_memorization_relationship(results, plot_path)
    
    # Create and save the frequency plot
    freq_plot_path = os.path.join(base_dir, 'homophily_memorization_frequency.png')
    freq_plot_path = plot_homophily_vs_memorization_frequency(results, freq_plot_path)
    
    # Save results to CSV
    df.to_csv(os.path.join(base_dir, 'results.csv'), index=False)
    
    logger.info("\nAnalysis complete! Results saved to:")
    logger.info(f"- Log file: {os.path.join(base_dir, 'analysis.log')}")
    logger.info(f"- Results CSV: {os.path.join(base_dir, 'results.csv')}")
    logger.info(f"- Summary plot: {plot_path}")
    logger.info(f"- Frequency plot: {freq_plot_path}")
    if hasattr(df, 'attrs') and 'info_plot_path' in df.attrs:
        logger.info(f"- Informativeness plot: {df.attrs['info_plot_path']}")

if __name__ == '__main__':
    main()
