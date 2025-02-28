import subprocess
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_experiment(embedding_layer):
    """Run the main.py script with specified embedding layer"""
    # Create output directory
    output_dir = f"results/Cora/Emb{embedding_layer}"
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "python", "main.py",
        "--dataset", "Cora",
        "--model_type", "gcn",
        "--hidden_dim", "32",
        "--num_layers", "4",
        "--augmentation_type", "flip",
        "--feature_flip_rate", "0.09",
        "--num_augmentations", "5",
        "--embedding_layer", str(embedding_layer),
        "--epochs", "100"
    ]
    
    print(f"\nRunning experiment with embedding_layer = {embedding_layer}")
    print("=" * 80)
    
    # Run the command and capture output
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    # Store results
    results = {
        'layer': embedding_layer,
        'memorization_scores': {},
        'nli_scores': {},
        'clustering_correlation': None,
        'clustering_pvalue': None
    }
    
    # Process output in real-time and extract relevant information
    for line in process.stdout:
        print(line.strip())
        
        # Extract memorization scores
        if "Average memorization score for" in line:
            parts = line.strip().split("nodes:")
            node_type = parts[0].split("for ")[-1].strip()
            score = float(parts[1].strip())
            results['memorization_scores'][node_type] = score
        
        # Extract NLI scores
        elif "Average NLI score:" in line:
            parts = line.strip().split("nodes:")
            if len(parts) > 1:
                node_type = parts[0].strip()
                score = float(parts[1].split(':')[1].strip())
                results['nli_scores'][node_type] = score
        
        # Extract clustering correlation
        elif "Score clustering correlation:" in line:
            results['clustering_correlation'] = float(line.strip().split(":")[-1].strip())
        
        # Extract clustering p-value
        elif "Score clustering p-value:" in line:
            results['clustering_pvalue'] = float(line.strip().split(":")[-1].strip())
    
    # Wait for process to complete
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error running experiment for layer {embedding_layer}:")
        print(stderr)
    
    return results

def compare_results(all_results):
    """Create visualizations comparing results across different layers"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join('results', 'layer_comparison', timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame for easier plotting
    rows = []
    for result in all_results:
        # Memorization scores
        for node_type, score in result['memorization_scores'].items():
            rows.append({
                'layer': result['layer'],
                'metric': 'memorization',
                'node_type': node_type,
                'value': score
            })
        
        # NLI scores
        for node_type, score in result['nli_scores'].items():
            rows.append({
                'layer': result['layer'],
                'metric': 'nli',
                'node_type': node_type,
                'value': score
            })
        
        # Clustering correlation
        rows.append({
            'layer': result['layer'],
            'metric': 'clustering_correlation',
            'node_type': 'all',
            'value': result['clustering_correlation']
        })
    
    df = pd.DataFrame(rows)
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Memorization scores by layer and node type
    plt.subplot(2, 2, 1)
    sns.barplot(
        data=df[df['metric'] == 'memorization'],
        x='layer',
        y='value',
        hue='node_type'
    )
    plt.title('Memorization Scores by Layer')
    plt.xlabel('Embedding Layer')
    plt.ylabel('Average Memorization Score')
    
    # Plot 2: NLI scores by layer and node type
    plt.subplot(2, 2, 2)
    sns.barplot(
        data=df[df['metric'] == 'nli'],
        x='layer',
        y='value',
        hue='node_type'
    )
    plt.title('NLI Scores by Layer')
    plt.xlabel('Embedding Layer')
    plt.ylabel('Average NLI Score')
    
    # Plot 3: Clustering correlation by layer
    plt.subplot(2, 2, 3)
    sns.barplot(
        data=df[df['metric'] == 'clustering_correlation'],
        x='layer',
        y='value'
    )
    plt.title('Clustering Correlation by Layer')
    plt.xlabel('Embedding Layer')
    plt.ylabel('Correlation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'layer_comparison.png'))
    plt.close()
    
    # Save numerical results
    results_file = os.path.join(output_dir, 'numerical_results.txt')
    with open(results_file, 'w') as f:
        f.write("Layer Comparison Results\n")
        f.write("=" * 50 + "\n\n")
        
        for layer in sorted(df['layer'].unique()):
            f.write(f"\nResults for Layer {layer}:\n")
            f.write("-" * 50 + "\n")
            
            # Memorization scores
            f.write("\nMemorization Scores:\n")
            layer_mem = df[(df['metric'] == 'memorization') & (df['layer'] == layer)]
            for _, row in layer_mem.iterrows():
                f.write(f"{row['node_type']}: {row['value']:.4f}\n")
            
            # NLI scores
            f.write("\nNLI Scores:\n")
            layer_nli = df[(df['metric'] == 'nli') & (df['layer'] == layer)]
            for _, row in layer_nli.iterrows():
                f.write(f"{row['node_type']}: {row['value']:.4f}\n")
            
            # Clustering correlation
            layer_corr = df[(df['metric'] == 'clustering_correlation') & (df['layer'] == layer)]
            if not layer_corr.empty:
                f.write(f"\nClustering Correlation: {layer_corr['value'].iloc[0]:.4f}\n")
            
            f.write("\n" + "=" * 50 + "\n")
    
    print(f"\nResults saved to {output_dir}")
    return output_dir

def main():
    # Run experiments for each layer
    all_results = []
    for layer in [0, 1, 2]:
        results = run_experiment(layer)
        all_results.append(results)
    
    # Compare and visualize results
    output_dir = compare_results(all_results)
    
    print("\nExperiment completed! Summary of findings:")
    print("=" * 80)
    
    # Print key findings
    for layer, results in enumerate(all_results):
        print(f"\nLayer {layer}:")
        print(f"- Clustering correlation: {results['clustering_correlation']:.3f} (p={results['clustering_pvalue']:.3f})")
        print("\nMemorization scores:")
        for node_type, score in results['memorization_scores'].items():
            print(f"- {node_type}: {score:.3f}")

if __name__ == "__main__":
    main()