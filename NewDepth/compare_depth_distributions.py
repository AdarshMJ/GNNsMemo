import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparison(original_df, rewired_df):
    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Compare GNN accuracies at different depths
    ax1.plot(original_df['depth'], original_df['gnn_accuracy'], 
             marker='o', label='Original Graph', linewidth=2)
    ax1.plot(rewired_df['depth'], rewired_df['gnn_accuracy'], 
             marker='s', label='Rewired Graph', linewidth=2)
    ax1.set_xlabel('Depth')
    ax1.set_ylabel('GNN Accuracy (%)')
    ax1.set_title('GNN Accuracy vs Depth')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Compare node distribution at different depths
    width = 0.35
    x = original_df['depth']
    ax2.bar(x - width/2, original_df['average_node_count'], 
            width, label='Original Graph')
    ax2.bar(x + width/2, rewired_df['average_node_count'], 
            width, label='Rewired Graph')
    ax2.set_xlabel('Depth')
    ax2.set_ylabel('Average Node Count')
    ax2.set_title('Node Distribution vs Depth')
    ax2.legend()
    ax2.grid(True)

    # Calculate and display overall statistics
    orig_weighted_acc = (original_df['gnn_accuracy'] * 
                        original_df['percentage_of_total']).sum() / 100
    rewired_weighted_acc = (rewired_df['gnn_accuracy'] * 
                           rewired_df['percentage_of_total']).sum() / 100
    
    stats_text = (f'Overall Weighted Accuracies:\n'
                 f'Original Graph: {orig_weighted_acc:.2f}%\n'
                 f'Rewired Graph: {rewired_weighted_acc:.2f}%')
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Update these paths to match your file locations
    original_csv = "/Users/adarshjamadandi/Desktop/Submissions/MemorizationGNNs/Memv2_Nov10/NCResults/CoraOG/aggregate_results_Cora_20241228_091548/depth_distribution_test_set.csv"
    rewired_csv = "/Users/adarshjamadandi/Desktop/Submissions/MemorizationGNNs/Memv2_Nov10/NCResults/CoraAddDel/aggregate_results_Cora_20241228_142527/depth_distribution_test_set.csv"
    
    original_df = pd.read_csv(original_csv)
    rewired_df = pd.read_csv(rewired_csv)
    
    plot_comparison(original_df, rewired_df) 