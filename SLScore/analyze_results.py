#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

def extract_experiment_info(directory_path):
    """Extract experiment information from the directory name."""
    base_dir = os.path.basename(directory_path)
    
    # Try to parse the directory name format: {model}_{dataset}_{num_layers}_{timestamp}
    parts = base_dir.split('_')
    
    if len(parts) >= 4:
        model_type = parts[0]
        dataset = parts[1]
        num_layers = int(parts[2])
        timestamp = '_'.join(parts[3:])
        
        return {
            'model_type': model_type,
            'dataset': dataset,
            'num_layers': num_layers,
            'timestamp': timestamp,
            'directory': directory_path
        }
    else:
        return None

def read_log_file(directory):
    """Read and parse the log file in the given directory."""
    log_files = glob.glob(os.path.join(directory, '*.log'))
    if not log_files:
        return None
    
    log_file = log_files[0]
    results = {
        'avg_scores': {},
        'nodes_above_threshold': {},
        'statistical_tests': {}
    }
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            # Extract average memorization scores
            if "Average memorization score for" in line:
                parts = line.strip().split("Average memorization score for ")
                if len(parts) > 1:
                    node_type, score = parts[1].split(" nodes: ")
                    results['avg_scores'][node_type] = float(score)
            
            # Extract nodes above threshold
            if "Nodes with score > 0.5:" in line:
                parts = line.strip().split("Nodes with score > 0.5: ")
                if len(parts) > 1:
                    match = re.search(r'(\d+)/(\d+) \(([\d\.]+)%\)', parts[1])
                    if match:
                        count, total, percentage = match.groups()
                        node_type = line.split("for ")[1].split(" nodes")[0]
                        results['nodes_above_threshold'][node_type] = {
                            'count': int(count),
                            'total': int(total),
                            'percentage': float(percentage)
                        }
            
            # Extract t-test results
            if "Candidate vs" in line:
                comparison = line.strip().split("Candidate vs ")[1].strip(':')
                p_value_line = lines[i + 2] if i + 2 < len(lines) else ""
                effect_size_line = lines[i + 3] if i + 3 < len(lines) else ""
                
                if "P-value:" in p_value_line and "Effect size" in effect_size_line:
                    p_value_match = re.search(r'P-value: ([\d\.e-]+)', p_value_line)
                    effect_size_match = re.search(r'Effect size \(Cohen\'s d\): ([\d\.]+)', effect_size_line)
                    
                    if p_value_match and effect_size_match:
                        p_value = float(p_value_match.group(1))
                        effect_size = float(effect_size_match.group(1))
                        results['statistical_tests'][f"candidate_vs_{comparison}"] = {
                            'p_value': p_value,
                            'effect_size': effect_size
                        }
        
        return results
    
    except Exception as e:
        print(f"Error reading log file {log_file}: {e}")
        return None

def aggregate_results(base_directory):
    """Find all experiment directories and aggregate their results."""
    # Find all experiment directories
    experiment_dirs = glob.glob(os.path.join(base_directory, "*_*_*_*"))
    
    all_results = []
    
    for directory in experiment_dirs:
        exp_info = extract_experiment_info(directory)
        if not exp_info:
            continue
        
        log_results = read_log_file(directory)
        if not log_results:
            continue
        
        # Combine experiment info with log results
        combined_results = {**exp_info, **log_results}
        all_results.append(combined_results)
    
    return all_results

def create_layer_comparison_plots(results, output_dir):
    """Create plots comparing results across different numbers of layers."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by dataset
    datasets = set(r['dataset'] for r in results)
    
    # Plot average memorization scores by layer count
    for dataset in datasets:
        dataset_results = [r for r in results if r['dataset'] == dataset]
        
        if not dataset_results:
            continue
        
        # Sort by number of layers
        dataset_results.sort(key=lambda x: x['num_layers'])
        
        # Extract data for plotting
        layers = [r['num_layers'] for r in dataset_results]
        
        # 1. Plot average memorization scores per node type
        plt.figure(figsize=(10, 6))
        node_types = ['candidate', 'shared', 'independent', 'extra']
        markers = ['o', 's', '^', 'x']
        
        for i, node_type in enumerate(node_types):
            scores = [r['avg_scores'].get(node_type, np.nan) for r in dataset_results]
            plt.plot(layers, scores, marker=markers[i], label=node_type, linewidth=2)
        
        plt.xlabel('Number of Layers')
        plt.ylabel('Average Memorization Score')
        plt.title(f'Memorization Scores vs. Model Depth ({dataset})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{dataset}_avg_scores_by_layer.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Plot percentage of nodes above threshold (0.5)
        plt.figure(figsize=(10, 6))
        
        for i, node_type in enumerate(node_types):
            percentages = [r['nodes_above_threshold'].get(node_type, {}).get('percentage', np.nan) 
                          for r in dataset_results]
            plt.plot(layers, percentages, marker=markers[i], label=node_type, linewidth=2)
        
        plt.xlabel('Number of Layers')
        plt.ylabel('Percentage of Nodes with Score > 0.5')
        plt.title(f'Nodes Above Threshold vs. Model Depth ({dataset})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{dataset}_above_threshold_by_layer.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Plot p-values for candidate vs. others
        plt.figure(figsize=(10, 6))
        
        comparison_types = ['shared', 'independent', 'extra']
        
        for i, comp_type in enumerate(comparison_types):
            p_values = [r['statistical_tests'].get(f'candidate_vs_{comp_type}', {}).get('p_value', np.nan) 
                       for r in dataset_results]
            plt.plot(layers, p_values, marker=markers[i], label=f'vs {comp_type}', linewidth=2)
        
        plt.xlabel('Number of Layers')
        plt.ylabel('P-value (t-test)')
        plt.title(f'Statistical Significance vs. Model Depth ({dataset})')
        plt.axhline(y=0.01, color='r', linestyle='--', label='p=0.01')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.yscale('log')  # Use log scale for p-values
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{dataset}_pvalues_by_layer.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Plot effect sizes
        plt.figure(figsize=(10, 6))
        
        for i, comp_type in enumerate(comparison_types):
            effect_sizes = [r['statistical_tests'].get(f'candidate_vs_{comp_type}', {}).get('effect_size', np.nan) 
                          for r in dataset_results]
            plt.plot(layers, effect_sizes, marker=markers[i], label=f'vs {comp_type}', linewidth=2)
        
        plt.xlabel('Number of Layers')
        plt.ylabel('Effect Size (Cohen\'s d)')
        plt.title(f'Effect Size vs. Model Depth ({dataset})')
        plt.axhline(y=0.8, color='r', linestyle='--', label='Large Effect (0.8)')
        plt.axhline(y=0.5, color='orange', linestyle='--', label='Medium Effect (0.5)')
        plt.axhline(y=0.2, color='g', linestyle='--', label='Small Effect (0.2)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{dataset}_effect_sizes_by_layer.png'), dpi=300, bbox_inches='tight')
        plt.close()

def create_dataset_comparison(results, output_dir):
    """Create plots comparing results across different datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by layer count
    layer_counts = set(r['num_layers'] for r in results)
    
    for layers in layer_counts:
        layer_results = [r for r in results if r['num_layers'] == layers]
        
        if not layer_results:
            continue
        
        # Sort by dataset name
        layer_results.sort(key=lambda x: x['dataset'])
        
        # Extract data for plotting
        datasets = [r['dataset'] for r in layer_results]
        
        # 1. Plot average memorization scores per node type
        plt.figure(figsize=(12, 6))
        node_types = ['candidate', 'shared', 'independent', 'extra']
        
        x = np.arange(len(datasets))
        width = 0.2
        
        for i, node_type in enumerate(node_types):
            scores = [r['avg_scores'].get(node_type, np.nan) for r in layer_results]
            plt.bar(x + i*width, scores, width, label=node_type)
        
        plt.xlabel('Dataset')
        plt.ylabel('Average Memorization Score')
        plt.title(f'Memorization Scores Across Datasets ({layers} Layers)')
        plt.xticks(x + width*1.5, datasets)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f'{layers}_layers_avg_scores_by_dataset.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Plot percentage of nodes above threshold (0.5)
        plt.figure(figsize=(12, 6))
        
        for i, node_type in enumerate(node_types):
            percentages = [r['nodes_above_threshold'].get(node_type, {}).get('percentage', np.nan) 
                          for r in layer_results]
            plt.bar(x + i*width, percentages, width, label=node_type)
        
        plt.xlabel('Dataset')
        plt.ylabel('Percentage of Nodes with Score > 0.5')
        plt.title(f'Nodes Above Threshold Across Datasets ({layers} Layers)')
        plt.xticks(x + width*1.5, datasets)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f'{layers}_layers_above_threshold_by_dataset.png'), dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_table(results, output_file):
    """Create a summary table with all key results."""
    
    table_data = []
    
    for result in results:
        row = {
            'Dataset': result['dataset'],
            'Layers': result['num_layers'],
            'Timestamp': result['timestamp']
        }
        
        # Add average scores
        for node_type, score in result.get('avg_scores', {}).items():
            row[f'Avg_{node_type}'] = f"{score:.4f}"
        
        # Add nodes above threshold
        for node_type, data in result.get('nodes_above_threshold', {}).items():
            row[f'Above_0.5_{node_type}'] = f"{data['count']}/{data['total']} ({data['percentage']:.1f}%)"
        
        # Add statistical significance
        for test, data in result.get('statistical_tests', {}).items():
            p_value = data.get('p_value', np.nan)
            effect_size = data.get('effect_size', np.nan)
            row[f'p_{test}'] = f"{p_value:.4f}"
            row[f'effect_{test}'] = f"{effect_size:.4f}"
        
        table_data.append(row)
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(table_data)
    df = df.sort_values(['Dataset', 'Layers']).reset_index(drop=True)
    
    # Save as CSV
    df.to_csv(output_file, index=False)
    
    return df

def main():
    # Base directory containing results
    base_dir = 'results'
    
    # Output directory for analysis
    analysis_dir = 'analysis'
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Aggregate all results
    print("Aggregating results...")
    all_results = aggregate_results(base_dir)
    
    if not all_results:
        print("No results found. Make sure experiments have completed.")
        return
    
    print(f"Found {len(all_results)} experiment results.")
    
    # Create comparison plots
    print("Creating layer comparison plots...")
    create_layer_comparison_plots(all_results, os.path.join(analysis_dir, 'layer_comparison'))
    
    print("Creating dataset comparison plots...")
    create_dataset_comparison(all_results, os.path.join(analysis_dir, 'dataset_comparison'))
    
    # Create summary table
    print("Creating summary table...")
    summary_file = os.path.join(analysis_dir, 'results_summary.csv')
    create_summary_table(all_results, summary_file)
    
    print(f"Analysis complete! Results saved in {analysis_dir}")

if __name__ == "__main__":
    main()