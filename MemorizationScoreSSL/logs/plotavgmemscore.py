import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def clean_tensor_string(tensor_str):
    """Convert tensor string to float value"""
    try:
        # Remove tensor( and device='cuda:0') parts and convert to float
        value = float(tensor_str.replace('tensor(', '').replace(", device='cuda:0')", ''))
        return value
    except:
        return 0.0

def plot_averaged_results(csv_files, save_path='memorization_averaged_edgeadd.png'):
    """
    Read multiple CSV files and create averaged plot
    """
    # Read and combine all dataframes
    all_dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        # Convert tensor strings to float values
        df['memorization_score'] = df['memorization_score'].apply(clean_tensor_string)
        all_dfs.append(df)
    
    # Get unique node indices
    all_nodes = set()
    for df in all_dfs:
        node_indices = df['node_idx'].apply(clean_tensor_string)
        all_nodes.update(node_indices)
    
    # Compute averages for each node
    avg_results = []
    for node in all_nodes:
        node_rows = []
        for df in all_dfs:
            node_idx_values = df['node_idx'].apply(clean_tensor_string)
            node_row = df[node_idx_values == node]
            if not node_row.empty:
                node_rows.append(node_row.iloc[0])
        
        if node_rows:
            avg_result = {
                'node_idx': node,
                'f_alignment': np.mean([float(row['f_alignment']) for row in node_rows]),
                'g_alignment': np.mean([float(row['g_alignment']) for row in node_rows]),
                'memorization_score': np.mean([float(row['memorization_score']) for row in node_rows])
            }
            avg_results.append(avg_result)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    f_alignments = [r['f_alignment'] for r in avg_results]
    g_alignments = [r['g_alignment'] for r in avg_results]
    mem_scores = [r['memorization_score'] for r in avg_results]
    
    # Create scatter plot
    scatter = plt.scatter(f_alignments, g_alignments, c=mem_scores, 
                         cmap='viridis', alpha=0.6)
    
    # Add y=x line
    min_val = min(min(f_alignments), min(g_alignments))
    max_val = max(max(f_alignments), max(g_alignments))
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', label='y=x')
    
    plt.colorbar(scatter, label='Memorization Score')
    plt.xlabel('Alignment Loss on f')
    plt.ylabel('Alignment Loss on g')
    plt.title('Model Alignment Loss vs. Memorization - Edge Addition (Averaged across seeds)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Print statistics
    mem_scores = np.array(mem_scores)
    print("\n=== Final Statistics (Averaged across all seeds) ===")
    print(f"Mean memorization score: {np.mean(mem_scores):.4f} ± {np.std(mem_scores):.4f}")
    print(f"Max memorization score: {np.max(mem_scores):.4f}")
    print(f"Min memorization score: {np.min(mem_scores):.4f}")
    print(f"Std deviation: {np.std(mem_scores):.4f}")

if __name__ == "__main__":
    # Since we're already in the logs directory, don't include 'logs/' in the path
    csv_files = [f'memadd{i}.csv' for i in range(1, 6)]
    
    # Verify files exist
    existing_files = [f for f in csv_files if os.path.exists(f)]
    
    if not existing_files:
        print("No CSV files found!")
        print("Looking for files: memadd1.csv, memadd2.csv, memadd3.csv, memadd4.csv, memadd5.csv")
        print("Current directory contents:")
        print(os.listdir('.'))  # Print current directory contents
    else:
        print(f"Found {len(existing_files)} CSV files:")
        for f in existing_files:
            print(f"  - {f}")
        plot_averaged_results(existing_files)