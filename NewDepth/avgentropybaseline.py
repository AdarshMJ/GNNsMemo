import time
import numpy as np
import torch
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import copy
import os
import random
from analysis import *
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
import networkx as nx
from trackDE import *
import re

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


# Global variable for save directory
def get_save_dir(dataset_name, num_epochs, num_layers, model_name, seed):
    """Create and return the save directory path with the new naming convention."""
    save_dir = Path(f'NCResults/{dataset_name}/{dataset_name}_{num_epochs}epochs_{num_layers}layers_{model_name}_seed{seed}')
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir

def train_and_get_results(data, model, optimizer, num_epochs, dataset_name, num_layers, seed, noise_level=1.0):
    """
    Train the model and track metrics.
    
    Args:
        data: Input graph data
        model: The GNN model
        optimizer: The optimizer
        num_epochs: Number of training epochs
        dataset_name: Name of the dataset
        num_layers: Number of layers in the model
        noise_level: Level of noise for entropy calculation (default: 1.0)
    """
    # Get model name from class name
    set_seed(seed)
    model_name = model.__class__.__name__
    
    # Create save_dir with all parameters including model name
    save_dir = get_save_dir(dataset_name, num_epochs, num_layers, model_name, seed)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    avg_testacc_before = []
    avg_trainacc_before = []
    avg_acc_testallsplits_before = []
    avg_acc_trainallsplits_before = []
    avg_valacc_before = []
    avg_acc_valallsplits_before = []
    energy_tracker = EnergyTracker(model, data)
    energy_history = []

    def train(model, optimizer):
        model.train()
        optimizer.zero_grad()  
        out = model(data.x, data.edge_index)          
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()  
        optimizer.step()  
        pred = out.argmax(dim=1)  
        train_correct = pred[train_mask] == data.y[train_mask]  
        train_acc = int(train_correct.sum()) / int(train_mask.sum())  
        return loss

    def val(model):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            val_correct = pred[val_mask] == data.y[val_mask]
            val_acc = int(val_correct.sum()) / int(val_mask.sum())
        return val_acc

    def test(model):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            test_correct = pred[test_mask] == data.y[test_mask]
            test_acc = int(test_correct.sum()) / int(test_mask.sum())
        return test_acc, pred, out

    # Initialize energy tracker (make sure data is on the correct device)
    data = data.to(next(model.parameters()).device)  # Move data to same device as model
    energy_tracker = EnergyTracker(model, data)
    
    for split_idx in range(0, 1):
        model.reset_parameters()
        optimizer = type(optimizer)(model.parameters(), **optimizer.defaults)
        train_mask = data.train_mask[:, split_idx]
        test_mask = data.test_mask[:, split_idx]
        val_mask = data.val_mask[:, split_idx]
        
        # Check for data leakage
        train_nodes = data.train_mask[:, split_idx].nonzero(as_tuple=True)[0].cpu().numpy()
        test_nodes = data.test_mask[:, split_idx].nonzero(as_tuple=True)[0].cpu().numpy()
        val_nodes = data.val_mask[:, split_idx].nonzero(as_tuple=True)[0].cpu().numpy()
        
        if len(np.intersect1d(train_nodes, test_nodes)) > 0:
            print("Warning: Data leakage between train and test sets")
            sys.exit(1)
        if len(np.intersect1d(train_nodes, val_nodes)) > 0:
            print("Warning: Data leakage between train and validation sets")
            sys.exit(1)
        
        # Initialize metric storage
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        test_losses = []
        test_accs = []
        
        print(f"Training for index = {split_idx}")

        # Training loop
        for epoch in tqdm(range(1, num_epochs+1)):
            # Train step
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

            # Track energies every 10 epochs
            if epoch % 10 == 0:
                try:
                    energy_tracker.track_step(epoch)
                except Exception as e:
                    print(f"Error tracking energy at epoch {epoch}: {str(e)}")

            # Calculate and store metrics
            with torch.no_grad():
                # Training metrics
                pred = out.argmax(dim=1)
                train_correct = pred[train_mask] == data.y[train_mask]
                train_acc = float(train_correct.sum()) / int(train_mask.sum())
                train_accs.append(train_acc * 100)
                train_losses.append(float(loss))

                # Validation metrics
                val_out = model(data.x, data.edge_index)
                val_loss = criterion(val_out[val_mask], data.y[val_mask])
                val_pred = val_out.argmax(dim=1)
                val_correct = val_pred[val_mask] == data.y[val_mask]
                val_acc = float(val_correct.sum()) / int(val_mask.sum())
                val_accs.append(val_acc * 100)
                val_losses.append(float(val_loss))

                # Test metrics
                test_out = model(data.x, data.edge_index)
                test_loss = criterion(test_out[test_mask], data.y[test_mask])
                test_pred = test_out.argmax(dim=1)
                test_correct = test_pred[test_mask] == data.y[test_mask]
                test_acc = float(test_correct.sum()) / int(test_mask.sum())
                test_accs.append(test_acc * 100)
                test_losses.append(float(test_loss))

        # Plot training metrics
        plot_training_metrics(
            train_losses, train_accs,
            val_losses, val_accs,
            test_losses, test_accs,
            split_idx,
            save_dir
        )
        # After training, plot energy dynamics
        try:
            energy_tracker.plot_training_dynamics(
                save_dir / f'energy_dynamics_split_{split_idx}_{time.strftime("%Y%m%d_%H%M%S")}.png'
            )
            analyze_model_energies(model, data, save_dir, split_idx)
        except Exception as e:
            print(f"Error plotting energy dynamics: {str(e)}")

        # Final evaluation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            confidences = F.softmax(out, dim=1)
            
            # Calculate delta entropy here
            delta_entropy = kd_retention(model, data, noise_level)
            
            # Calculate final metrics
            train_correct = pred[train_mask] == data.y[train_mask]
            train_acc = int(train_correct.sum()) / int(train_mask.sum())
            avg_trainacc_before.append(train_acc * 100)

            val_acc = val(model)
            avg_valacc_before.append(val_acc * 100)
            
            test_acc, pred, out = test(model)
            avg_testacc_before.append(test_acc * 100)
            
            print(f'Final Training Accuracy: {train_acc*100:.4f}')
            print(f'Final Validation Accuracy: {val_acc*100:.4f}')
            print(f'Final Test Accuracy: {test_acc*100:.4f}\n')

            # Now add the prediction depth analysis
            results_df = analyze_prediction_depth(
                model=model,
                data=data,
                train_mask=train_mask,
                test_mask=test_mask,
                save_dir=save_dir,
                split_idx=split_idx
            )

        # Store average metrics
        avg_acc_trainallsplits_before.append(np.mean(avg_trainacc_before))
        avg_acc_valallsplits_before.append(np.mean(avg_valacc_before))
        avg_acc_testallsplits_before.append(np.mean(avg_testacc_before))

    return (
        avg_acc_testallsplits_before, 
        avg_acc_valallsplits_before, 
        avg_acc_trainallsplits_before,
    )

def plot_training_metrics(losses, accuracies, val_losses, val_accuracies, test_losses, test_accuracies, split_idx, save_dir):
    """Plot training, validation, and test metrics side by side."""
    epochs = range(1, len(losses) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot losses
    ax1.plot(epochs, losses, 'b-', label='Training')
    ax1.plot(epochs, val_losses, 'g-', label='Validation')
    ax1.plot(epochs, test_losses, 'r-', label='Test')
    ax1.set_title(f'Loss vs Epochs (Split {split_idx})', fontsize=20)
    ax1.set_xlabel('Epoch', fontsize=20)
    ax1.set_ylabel('Loss', fontsize=20)
    ax1.tick_params(labelsize=20)
    ax1.legend(fontsize=20)
    
    # Set integer x-axis for epochs
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Plot accuracies
    ax2.plot(epochs, accuracies, 'b-', label='Training')
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation')
    ax2.plot(epochs, test_accuracies, 'r-', label='Test')
    ax2.set_title(f'Accuracy vs Epochs (Split {split_idx})', fontsize=20)
    ax2.set_xlabel('Epoch', fontsize=20)
    ax2.set_ylabel('Accuracy (%)', fontsize=20)
    ax2.tick_params(labelsize=20)
    ax2.legend(fontsize=20)
    
    # Set integer x-axis for epochs and appropriate y-axis for percentages
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(10))  # Set ticks every 10%
    ax2.set_ylim(-5, 105)  # Set y-axis range for percentages
    
    plt.tight_layout()
    plt.savefig(save_dir / f'baselinetraining_metrics_split_{split_idx}_{len(losses)}.png', 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'epoch': epochs,
        'train_loss': losses,
        'train_accuracy': accuracies,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'test_loss': test_losses,
        'test_accuracy': test_accuracies
    })
    metrics_df.to_csv(save_dir / f'baselinetraining_metrics_split_{split_idx}_{len(losses)}.csv', index=False)
    
    # Print final metrics
    print(f"\nFinal metrics (Split {split_idx}):")
    print(f"Training - Loss: {losses[-1]:.4f}, Accuracy: {accuracies[-1]:.2f}%")
    print(f"Validation - Loss: {val_losses[-1]:.4f}, Accuracy: {val_accuracies[-1]:.2f}%")
    print(f"Test - Loss: {test_losses[-1]:.4f}, Accuracy: {test_accuracies[-1]:.2f}%")

def aggregate_results_across_seeds(base_save_dir, dataset_name, num_epochs, num_layers, model_name, seeds):
    """Aggregate results from multiple seeds and create summary visualizations."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    aggregate_dir = Path(f'NCResults/{dataset_name}/aggregate_results_{dataset_name}_{timestamp}')
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    # Initialize dictionaries to store metrics across seeds
    all_metrics = {
        'train': {'accuracy': [], 'loss': []},
        'val': {'accuracy': [], 'loss': []},
        'test': {'accuracy': [], 'loss': []}
    }
    
    all_depth_distributions = []
    all_entropy_accuracies = {'train': [], 'test': []}
    all_gradient_results = []
    
    # Add tracking for Dirichlet energies
    all_layer_energies = []
    all_layer_norms = []

    # Add container for late learner neighborhood results
    all_neighborhood_stats = {'train': [], 'test': []}

    # Collect data from each seed
    for seed in seeds:
        seed_dir = Path(f'NCResults/{dataset_name}/{dataset_name}_{num_epochs}epochs_{num_layers}layers_{model_name}_seed{seed}')
        
        try:
            # Load training metrics
            metrics_files = list(seed_dir.glob('baselinetraining_metrics_split_0_*.csv'))
            if metrics_files:
                metrics_df = pd.read_csv(metrics_files[0])
                
                for metric_type in ['train', 'val', 'test']:
                    all_metrics[metric_type]['accuracy'].append(metrics_df[f'{metric_type}_accuracy'].values)
                    all_metrics[metric_type]['loss'].append(metrics_df[f'{metric_type}_loss'].values)

            # Load depth analysis results
            for set_type in ['train', 'test']:
                depth_files = list(seed_dir.glob(f'full_depth_analysis_{set_type}_split_0_*.csv'))
                if depth_files:
                    depth_df = pd.read_csv(depth_files[0])
                    all_depth_distributions.append((seed, set_type, depth_df))
                
                # Load accuracy vs entropy results
                entropy_files = list(seed_dir.glob(f'gnn_accuracy_vs_entropy_{set_type}_split_0_*.csv'))
                if entropy_files:
                    entropy_df = pd.read_csv(entropy_files[0])
                    all_entropy_accuracies[set_type].append(entropy_df)

            # Load layer energies data
            energy_files = list(seed_dir.glob('layer_energies_split_0_*.csv'))
            if energy_files:
                energy_df = pd.read_csv(energy_files[0])
                all_layer_energies.append(energy_df['dirichlet_energy'].values)
                all_layer_norms.append(energy_df['layer_norm'].values)

            # Load gradient results
            for set_type in ['train', 'test']:
                gradient_files = list(seed_dir.glob(f'gradient_patterns_{set_type}_split_0_*.csv'))
                if gradient_files:
                    gradient_df = pd.read_csv(gradient_files[0])
                    all_gradient_results.append((seed, set_type, gradient_df))

            # Load late learner neighborhood analysis results
            for set_type in ['train', 'test']:
                neighborhood_files = list(seed_dir.glob(f'late_learner_neighborhood_stats_{set_type}_split_0_*.csv'))
                if neighborhood_files:
                    stats_df = pd.read_csv(neighborhood_files[0])
                    all_neighborhood_stats[set_type].append(stats_df)

        except Exception as e:
            print(f"Error processing seed {seed}: {str(e)}")
            continue

    # Only proceed if we have data
    if not all_metrics['train']['accuracy']:
        print("No valid metrics data found!")
        return

    # Create aggregate visualizations
    create_aggregate_training_plot(all_metrics, aggregate_dir)
    if all_depth_distributions:
        create_aggregate_depth_distribution(all_depth_distributions, aggregate_dir)
        create_aggregate_accuracy_depth_plots(all_depth_distributions, aggregate_dir)
        create_aggregate_early_late_learners(all_depth_distributions, aggregate_dir)
    if any(all_entropy_accuracies.values()):
        create_aggregate_entropy_accuracy_plot(all_entropy_accuracies, aggregate_dir)
    
    # Save summary statistics
    summary_stats = calculate_summary_statistics(all_metrics, all_depth_distributions)
    with open(aggregate_dir / 'summary_statistics.txt', 'w') as f:
        f.write(summary_stats)

    # Create aggregate Dirichlet energy plot if we have data
    if all_layer_energies:
        create_aggregate_energy_plot(all_layer_energies, all_layer_norms, aggregate_dir)

    # Create aggregate gradient plots if we have data
    if all_gradient_results:
        for set_type in ['train', 'test']:
            set_results = [df for seed, st, df in all_gradient_results if st == set_type]
            if set_results:
                plot_average_gradient_norms(set_results, aggregate_dir, set_type)

    # Create aggregate visualizations for neighborhood analysis
    #if any(stats for stats in all_neighborhood_stats.values()):
        #create_aggregate_neighborhood_analysis(all_neighborhood_stats, aggregate_dir)

def create_aggregate_training_plot(all_metrics, save_dir):
    """Create aggregate training metrics plot with confidence intervals."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    epochs = range(1, len(all_metrics['train']['accuracy'][0]) + 1)
    colors = {'train': 'blue', 'val': 'green', 'test': 'red'}
    
    # Plot losses
    for metric_type in ['train', 'val', 'test']:
        losses = np.array(all_metrics[metric_type]['loss'])
        mean_loss = np.mean(losses, axis=0)
        std_loss = np.std(losses, axis=0)
        ci_loss = 1.96 * std_loss / np.sqrt(len(losses))
        
        ax1.plot(epochs, mean_loss, color=colors[metric_type], label=f'{metric_type.capitalize()}')
        ax1.fill_between(epochs, mean_loss - ci_loss, mean_loss + ci_loss, 
                        color=colors[metric_type], alpha=0.2)
    
    ax1.set_title('Loss vs Epochs', fontsize=20)
    ax1.set_xlabel('Epoch', fontsize=20)
    ax1.set_ylabel('Loss', fontsize=20)
    ax1.tick_params(labelsize=20)
    ax1.legend(fontsize=20)
    
    # Plot accuracies
    for metric_type in ['train', 'val', 'test']:
        accs = np.array(all_metrics[metric_type]['accuracy'])
        mean_acc = np.mean(accs, axis=0)
        std_acc = np.std(accs, axis=0)
        ci_acc = 1.96 * std_acc / np.sqrt(len(accs))
        
        ax2.plot(epochs, mean_acc, color=colors[metric_type], label=f'{metric_type.capitalize()}')
        ax2.fill_between(epochs, mean_acc - ci_acc, mean_acc + ci_acc,
                        color=colors[metric_type], alpha=0.2)
    
    ax2.set_title('Accuracy vs Epochs', fontsize=20)
    ax2.set_xlabel('Epoch', fontsize=20)
    ax2.set_ylabel('Accuracy (%)', fontsize=20)
    ax2.tick_params(labelsize=20)
    ax2.legend(fontsize=20)
    ax2.set_ylim(-5, 105)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'aggregate_training_metrics.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_aggregate_depth_distribution(all_depth_distributions, save_dir):
    """Create aggregate depth distribution plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    for set_type, ax in zip(['train', 'test'], [ax1, ax2]):
        depth_data = [df['prediction_depth'] for seed, st, df in all_depth_distributions if st == set_type]
        
        # Calculate mean and CI for each depth
        all_depths = np.unique(np.concatenate([d.unique() for d in depth_data]))
        depth_counts = np.zeros((len(depth_data), len(all_depths)))
        
        for i, depths in enumerate(depth_data):
            counts = depths.value_counts()
            for j, depth in enumerate(all_depths):
                depth_counts[i, j] = counts.get(depth, 0)
        
        # Normalize to percentages
        depth_percentages = depth_counts / depth_counts.sum(axis=1, keepdims=True) * 100
        mean_percentages = np.mean(depth_percentages, axis=0)
        ci_percentages = 1.96 * np.std(depth_percentages, axis=0) / np.sqrt(len(depth_data))
        
        ax.bar(all_depths, mean_percentages, yerr=ci_percentages, capsize=5)
        ax.set_title(f'Prediction Depth Distribution ({set_type.capitalize()} Set)', fontsize=20)
        ax.set_xlabel('Layer', fontsize=20)
        ax.set_ylabel('Percentage of Nodes', fontsize=20)
        ax.tick_params(labelsize=20)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'aggregate_depth_distribution.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_aggregate_entropy_accuracy_plot(all_entropy_accuracies, save_dir):
    """Create aggregate entropy vs accuracy plot with detailed x-axis labels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    for set_type, ax in zip(['train', 'test'], [ax1, ax2]):
        entropy_dfs = all_entropy_accuracies[set_type]
        if not entropy_dfs:
            continue
            
        # Get entropy ranges from the first DataFrame
        # Assuming the entropy_range column contains strings like "0.123-0.456 (n=100)"
        entropy_ranges = entropy_dfs[0]['entropy_range'].str.extract(r'([\d.]+)-([\d.]+)')[0].astype(float)
        
        # Combine accuracies from all seeds
        accuracies = np.array([df['accuracy'].values for df in entropy_dfs])
        mean_acc = np.mean(accuracies, axis=0)
        ci_acc = 1.96 * np.std(accuracies, axis=0) / np.sqrt(len(accuracies))
        
        # Calculate average node counts for each bin
        node_counts = np.mean([df['num_nodes'].values for df in entropy_dfs], axis=0)
        
        # Create bar plot for node counts
        bars = ax.bar(range(len(mean_acc)), node_counts, alpha=0.3, color='blue')
        
        # Create twin axis for accuracy
        ax2 = ax.twinx()
        
        # Plot accuracy with error bars
        line = ax2.errorbar(range(len(mean_acc)), mean_acc, yerr=ci_acc,
                          marker='o', capsize=5, capthick=1, elinewidth=1,
                          color='red', linestyle='-', linewidth=2, markersize=8)
        
        # Create x-axis labels with entropy ranges and sample sizes
        x_labels = []
        for i, (df, acc) in enumerate(zip(entropy_dfs[0].itertuples(), mean_acc)):
            entropy_range = df.entropy_range.split('\n')[0]  # Get just the range part
            avg_nodes = int(np.mean([d['num_nodes'].iloc[i] for d in entropy_dfs]))
            x_labels.append(f'{entropy_range}\n(n={avg_nodes})')
        
        # Set labels and title
        ax.set_xticks(range(len(mean_acc)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_xlabel('Delta Entropy Range (average sample size)', fontsize=20)
        ax.set_ylabel('Number of Nodes', color='blue', fontsize=20)
        ax2.set_ylabel('Accuracy', color='red', fontsize=20)
        ax.set_title(f'Accuracy vs Delta Entropy ({set_type.capitalize()} Set)', fontsize=20)
        ax.tick_params(labelsize=20)
        ax2.tick_params(labelsize=20)
        ax.legend(fontsize=20)
        ax2.legend(fontsize=20)
        
        # Add grid for easier reading
        ax2.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend([bars.patches[0]], ['Node Count'], loc='upper left')
        ax2.legend([line], ['Accuracy'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'aggregate_entropy_accuracy.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Also save the numerical data
    summary_df = pd.DataFrame({
        'entropy_range': x_labels,
        'mean_accuracy': mean_acc,
        'confidence_interval': ci_acc,
        'average_node_count': node_counts
    })
    summary_df.to_csv(save_dir / 'aggregate_entropy_accuracy_summary.csv', index=False)

def create_aggregate_accuracy_depth_plots(all_depth_distributions, save_dir):
    """Create aggregate analysis of accuracy vs depth with separate GNN and KNN plots."""
    # Create two separate figures
    fig_gnn, (ax1_gnn, ax2_gnn) = plt.subplots(1, 2, figsize=(20, 8))
    fig_knn, (ax1_knn, ax2_knn) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Initialize dictionaries to store aggregated data
    train_data = {'depths': [], 'node_counts': [], 'gnn_accuracies': [], 'knn_accuracies': []}
    test_data = {'depths': [], 'node_counts': [], 'gnn_accuracies': [], 'knn_accuracies': []}
    
    for set_type, ax_gnn, ax_knn in zip(['train', 'test'], [ax1_gnn, ax2_gnn], [ax1_knn, ax2_knn]):
        depth_dfs = [df for seed, st, df in all_depth_distributions if st == set_type]
        
        if depth_dfs:
            # Calculate statistics for each depth
            all_depths = sorted(depth_dfs[0]['prediction_depth'].unique())
            
            for depth in all_depths:
                # Get node counts and accuracies for this depth across all seeds
                depth_stats = []
                for df in depth_dfs:
                    depth_mask = df['prediction_depth'] == depth
                    depth_stats.append({
                        'node_count': depth_mask.sum(),
                        'gnn_acc': df[depth_mask]['gnn_correct'].mean() if depth_mask.any() else 0,
                        'knn_acc': df[depth_mask]['knn_correct_at_depth'].mean() if depth_mask.any() else 0
                    })
                
                # Calculate means and confidence intervals
                mean_count = np.mean([stats['node_count'] for stats in depth_stats])
                mean_gnn_acc = np.mean([stats['gnn_acc'] for stats in depth_stats]) * 100
                mean_knn_acc = np.mean([stats['knn_acc'] for stats in depth_stats]) * 100
                
                gnn_ci = 1.96 * np.std([stats['gnn_acc'] for stats in depth_stats]) * 100 / np.sqrt(len(depth_stats))
                knn_ci = 1.96 * np.std([stats['knn_acc'] for stats in depth_stats]) * 100 / np.sqrt(len(depth_stats))
                
                # Store data for CSV
                data_dict = train_data if set_type == 'train' else test_data
                data_dict['depths'].append(depth)
                data_dict['node_counts'].append(mean_count)
                data_dict['gnn_accuracies'].append(mean_gnn_acc)
                data_dict['knn_accuracies'].append(mean_knn_acc)
            
            # Create twin axes for node counts and accuracies
            ax_gnn2 = ax_gnn.twinx()
            ax_knn2 = ax_knn.twinx()
            
            # Plot node count bars (blue)
            ax_gnn.bar(data_dict['depths'], data_dict['node_counts'], 
                      alpha=0.3, color='blue', label='Node Count')
            ax_knn.bar(data_dict['depths'], data_dict['node_counts'], 
                      alpha=0.3, color='blue', label='Node Count')
            
            # Plot accuracy lines (red)
            ax_gnn2.errorbar(data_dict['depths'], data_dict['gnn_accuracies'],
                           yerr=[gnn_ci]*len(data_dict['depths']),
                           color='red', marker='o', capsize=5, capthick=1,
                           elinewidth=1, linewidth=2, markersize=8,
                           label='GNN Accuracy')
            ax_knn2.errorbar(data_dict['depths'], data_dict['knn_accuracies'],
                           yerr=[knn_ci]*len(data_dict['depths']),
                           color='red', marker='o', capsize=5, capthick=1,
                           elinewidth=1, linewidth=2, markersize=8,
                           label='KNN Accuracy')
            
            # Customize GNN plot
            ax_gnn.set_xlabel('Prediction Depth', fontsize=20)
            ax_gnn.set_ylabel('Number of Nodes', fontsize=20, color='blue')
            ax_gnn2.set_ylabel('Accuracy (%)', fontsize=20, color='red')
            ax_gnn.set_title(f'GNN: {set_type.capitalize()} Set', fontsize=20)
            ax_gnn.tick_params(labelsize=15)
            ax_gnn2.tick_params(labelsize=15)
            ax_gnn.grid(True, alpha=0.3)
            
            # Set y-axis limits with padding
            ax_gnn2.set_ylim(-5, 105)  # Changed from (0, 100) to (-5, 105)
            
            # Customize KNN plot
            ax_knn.set_xlabel('Prediction Depth', fontsize=20)
            ax_knn.set_ylabel('Number of Nodes', fontsize=20, color='blue')
            ax_knn2.set_ylabel('Accuracy (%)', fontsize=20, color='red')
            ax_knn.set_title(f'KNN: {set_type.capitalize()} Set', fontsize=20)
            ax_knn.tick_params(labelsize=15)
            ax_knn2.tick_params(labelsize=15)
            ax_knn.grid(True, alpha=0.3)
            
            # Set y-axis limits with padding
            ax_knn2.set_ylim(-5, 105)  # Changed from (0, 100) to (-5, 105)
            
            # Add legends with adjusted position
            lines_gnn, labels_gnn = ax_gnn.get_legend_handles_labels()
            lines_gnn2, labels_gnn2 = ax_gnn2.get_legend_handles_labels()
            ax_gnn.legend(lines_gnn + lines_gnn2, labels_gnn + labels_gnn2, 
                        bbox_to_anchor=(1.0, 1.15), loc='upper right', fontsize=15)
            
            lines_knn, labels_knn = ax_knn.get_legend_handles_labels()
            lines_knn2, labels_knn2 = ax_knn2.get_legend_handles_labels()
            ax_knn.legend(lines_knn + lines_knn2, labels_knn + labels_knn2, 
                        bbox_to_anchor=(1.0, 1.15), loc='upper right', fontsize=15)
    
    # Save plots with adjusted padding
    plt.figure(fig_gnn.number)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect parameters for padding
    plt.savefig(save_dir / 'aggregate_gnn_accuracy_depth.png', bbox_inches='tight', dpi=300)
    
    plt.figure(fig_knn.number)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the rect parameters for padding
    plt.savefig(save_dir / 'aggregate_knn_accuracy_depth.png', bbox_inches='tight', dpi=300)
    
    plt.close('all')
    
    # Save node counts and accuracies to CSV files
    for set_type, data in [('train', train_data), ('test', test_data)]:
        df = pd.DataFrame({
            'depth': data['depths'],
            'average_node_count': data['node_counts'],
            'gnn_accuracy': data['gnn_accuracies'],
            'knn_accuracy': data['knn_accuracies']
        })
        
        # Add percentage of total nodes
        total_nodes = df['average_node_count'].sum()
        df['percentage_of_total'] = (df['average_node_count'] / total_nodes) * 100
        
        # Save to CSV
        df.to_csv(save_dir / f'depth_distribution_{set_type}_set.csv', index=False)
        
        # Print summary
        print(f"\n{set_type.capitalize()} Set Depth Distribution:")
        print(df.to_string(float_format=lambda x: '{:.2f}'.format(x)))

def create_aggregate_early_late_learners(all_depth_distributions, save_dir):
    """Create aggregate analysis of early vs late learners."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    for set_type, ax in zip(['train', 'test'], [ax1, ax2]):
        early_depths = []
        late_depths = []
        
        for _, st, df in all_depth_distributions:
            if st == set_type:
                # Just use prediction depths instead of entropy
                early = df[df['prediction_depth'] <= 1]['prediction_depth']
                late = df[df['prediction_depth'] >= 2]['prediction_depth']
                
                early_depths.append(early)
                late_depths.append(late)
        
        if early_depths and late_depths:  # Check if we have data
            # Calculate statistics
            early_counts = [len(e) for e in early_depths]
            late_counts = [len(l) for l in late_depths]
            total_counts = [e + l for e, l in zip(early_counts, late_counts)]
            
            early_percentages = [e/t * 100 for e, t in zip(early_counts, total_counts)]
            late_percentages = [l/t * 100 for l, t in zip(late_counts, total_counts)]
            
            early_mean = np.mean(early_percentages)
            late_mean = np.mean(late_percentages)
            
            early_ci = 1.96 * np.std(early_percentages) / np.sqrt(len(early_percentages))
            late_ci = 1.96 * np.std(late_percentages) / np.sqrt(len(late_percentages))
            
            # Plot
            ax.bar([1, 2], [early_mean, late_mean], yerr=[early_ci, late_ci],
                  capsize=5, color=['lightblue', 'lightgreen'], alpha=0.7)
            
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Early Learners\n(Depth ≤ 1)', 'Late Learners\n(Depth ≥ 2)'])
            ax.set_title(f'Distribution: Early vs Late Learners ({set_type.capitalize()} Set)', 
                        fontsize=20)
            ax.set_xlabel('Learner Type', fontsize=20)
            ax.set_ylabel('Percentage of Nodes (%)', fontsize=20)
            ax.tick_params(labelsize=20)
            
            # Add percentage labels
            ax.text(1, early_mean, f'{early_mean:.1f}%',
                   horizontalalignment='center', verticalalignment='bottom')
            ax.text(2, late_mean, f'{late_mean:.1f}%',
                   horizontalalignment='center', verticalalignment='bottom')
            
            # Set y-axis limits to show percentages properly
            ax.set_ylim(0, 100)
    
    #plt.tight_layout()
    #plt.savefig(save_dir / 'aggregate_early_late_learners.png', bbox_inches='tight', dpi=300)
    #plt.close()

def calculate_summary_statistics(all_metrics, all_depth_distributions):
    """Calculate and format summary statistics across all seeds."""
    stats = []
    
    # Final metrics statistics
    for metric_type in ['train', 'val', 'test']:
        # Convert to numpy array if needed and get the last value
        final_accs = [acc.iloc[-1] if hasattr(acc, 'iloc') else acc[-1] 
                     for acc in all_metrics[metric_type]['accuracy']]
        mean_acc = np.mean(final_accs)
        ci_acc = 1.96 * np.std(final_accs) / np.sqrt(len(final_accs))
        
        stats.append(f"Final {metric_type.capitalize()} Accuracy: {mean_acc:.2f}% ± {ci_acc:.2f}%")
    
    # Early vs Late learner statistics
    for set_type in ['train', 'test']:
        early_counts = []
        late_counts = []
        
        for _, st, df in all_depth_distributions:
            if st == set_type:
                early = len(df[df['prediction_depth'] <= 1])
                late = len(df[df['prediction_depth'] >= 2])
                early_counts.append(early / len(df) * 100)
                late_counts.append(late / len(df) * 100)
        
        if early_counts and late_counts:  # Check if lists are not empty
            early_mean = np.mean(early_counts)
            late_mean = np.mean(late_counts)
            early_ci = 1.96 * np.std(early_counts) / np.sqrt(len(early_counts))
            late_ci = 1.96 * np.std(late_counts) / np.sqrt(len(late_counts))
            
            stats.append(f"\n{set_type.capitalize()} Set:")
            stats.append(f"Early Learners: {early_mean:.2f}% ± {early_ci:.2f}%")
            stats.append(f"Late Learners: {late_mean:.2f}% ± {late_ci:.2f}%")
    
    return '\n'.join(stats)





# Modify the main training function to handle multiple seeds
def run_multiple_seeds(data, model_class, optimizer_class, optimizer_params, num_epochs, dataset_name, num_layers, seeds):
    """Run the entire training process for multiple seeds."""
    all_results = []
    
    for seed in seeds:
        print(f"\nTraining with seed {seed}")
        set_seed(seed)
        
        # Initialize new model and optimizer for each seed
        model = model_class()  # Create new model instance
        optimizer = optimizer_class(model.parameters(), **optimizer_params)  # Create optimizer with parameters
        
        # Train model
        results = train_and_get_results(
            data=data,
            model=model,
            optimizer=optimizer,
            num_epochs=num_epochs,
            dataset_name=dataset_name,
            num_layers=num_layers,
            seed=seed
        )
        all_results.append(results)
    
    # Calculate and print average accuracies across all seeds
    avg_test_acc = np.mean([res[0][0] for res in all_results])
    avg_train_acc = np.mean([res[2][0] for res in all_results])
    test_std = np.std([res[0][0] for res in all_results])
    train_std = np.std([res[2][0] for res in all_results])
    
    print("\n" + "="*50)
    print(f"Final Results Averaged Over {len(seeds)} Seeds:")
    print(f"Average Train Accuracy: {avg_train_acc:.2f}% ± {train_std:.2f}%")
    print(f"Average Test Accuracy: {avg_test_acc:.2f}% ± {test_std:.2f}%")
    print("="*50 + "\n")
    
    # Aggregate results
    aggregate_results_across_seeds(
        base_save_dir='EntropyTracking',
        dataset_name=dataset_name,
        num_epochs=num_epochs,
        num_layers=num_layers,
        model_name=model_class().__class__.__name__,  # Get model name
        seeds=seeds
    )
    
    return all_results

def create_aggregate_energy_plot(all_energies, all_norms, save_dir):
    """Create aggregate plot of Dirichlet energies and norms across layers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Convert to numpy arrays
    energies_array = np.array(all_energies)
    norms_array = np.array(all_norms)
    
    # Calculate means and confidence intervals
    energy_means = np.mean(energies_array, axis=0)
    energy_ci = 1.96 * np.std(energies_array, axis=0) / np.sqrt(len(all_energies))
    
    norm_means = np.mean(norms_array, axis=0)
    norm_ci = 1.96 * np.std(norms_array, axis=0) / np.sqrt(len(all_norms))
    
    layers = range(len(energy_means))
    
    # Plot Dirichlet energies
    ax1.errorbar(layers, energy_means, yerr=energy_ci, 
                marker='o', capsize=5, capthick=1, 
                elinewidth=1, linewidth=2, markersize=8,
                label='Mean Dirichlet Energy')
    ax1.set_title('Average Dirichlet Energy Across Layers', fontsize=20)
    ax1.set_xlabel('Layer', fontsize=20)
    ax1.set_ylabel('Dirichlet Energy', fontsize=20)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=15)
    
    # Plot layer norms
    ax2.errorbar(layers, norm_means, yerr=norm_ci,
                marker='o', capsize=5, capthick=1,
                elinewidth=1, linewidth=2, markersize=8,
                label='Mean Layer Norm')
    ax2.set_title('Average Layer Norms Across Layers', fontsize=20)
    ax2.set_xlabel('Layer', fontsize=20)
    ax2.set_ylabel('Layer Norm', fontsize=20)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=15)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'aggregate_layer_energies.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save numerical results
    results_df = pd.DataFrame({
        'layer': layers,
        'mean_dirichlet_energy': energy_means,
        'energy_ci': energy_ci,
        'mean_layer_norm': norm_means,
        'norm_ci': norm_ci
    })
    results_df.to_csv(save_dir / 'aggregate_layer_energies.csv', index=False)
    
    # Print summary statistics
    print("\nDirichlet Energy Summary:")
    for i, (energy, ci) in enumerate(zip(energy_means, energy_ci)):
        print(f"Layer {i}: {energy:.4f} ± {ci:.4f}")

def plot_average_gradient_norms(gradient_dfs, save_dir, set_type):
    """Create aggregate plot of gradient norms across seeds."""
    print(f"\nCreating aggregate gradient norms plot for {set_type} set...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    try:
        # Get all unique depths from the column names
        depths = sorted([int(col.split('_')[1]) for col in gradient_dfs[0].columns if col.startswith('Depth')])
        
        # Use actual layer parameter names
        layer_names = [
            'convs.0.lin.weight', 'convs.0.lin.bias',
            'convs.1.lin.weight', 'convs.1.lin.bias',
            'convs.2.lin.weight', 'convs.2.lin.bias',
            'convs.3.lin.weight', 'convs.3.lin.bias',
            'convs.4.lin.weight', 'convs.4.lin.bias'
        ]
        num_layers = len(layer_names)
        
        plt.figure(figsize=(15, 6))  # Made wider to accommodate longer names
        
        # Plot line for each depth
        colors = plt.cm.viridis(np.linspace(0, 1, len(depths)))
        for depth, color in zip(depths, colors):
            depth_values = []
            for df in gradient_dfs:
                depth_col = f'Depth_{depth}'
                if depth_col in df.columns:
                    depth_values.append(df[depth_col].values[:num_layers])
            
            depth_values = np.array(depth_values)
            means = np.mean(depth_values, axis=0)
            ci = 1.96 * np.std(depth_values, axis=0) / np.sqrt(len(gradient_dfs))
            
            x = np.arange(num_layers)
            plt.errorbar(x, means, yerr=ci,
                        marker='o', capsize=5, capthick=1,
                        elinewidth=1, linewidth=2, markersize=8,
                        label=f'Depth {depth}',
                        color=color)
        
        plt.title(f'Average Gradient Norms by Prediction Depth ({set_type.capitalize()} Set)', fontsize=15)
        plt.xlabel('Model Parameters', fontsize=12)
        plt.ylabel('Normalized Gradient Norm', fontsize=12)
        
        plt.xticks(range(num_layers), layer_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Use log scale if values span multiple orders of magnitude
        if plt.ylim()[1] > 10 * plt.ylim()[0]:
            plt.yscale('log')
        
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(save_dir / f'aggregate_gradient_norms_{set_type}_{timestamp}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        # Save numerical results
        results = {
            'layer_name': layer_names
        }
        for depth in depths:
            depth_values = []
            for df in gradient_dfs:
                depth_col = f'Depth_{depth}'
                if depth_col in df.columns:
                    depth_values.append(df[depth_col].values[:num_layers])
            depth_values = np.array(depth_values)
            means = np.mean(depth_values, axis=0)
            ci = 1.96 * np.std(depth_values, axis=0) / np.sqrt(len(gradient_dfs))
            results[f'depth_{depth}_mean'] = means
            results[f'depth_{depth}_ci'] = ci
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_dir / f'aggregate_gradient_norms_{set_type}_{timestamp}.csv', index=False)
        
        # Print summary statistics for each depth
        for depth in depths:
            print(f"\nGradient Statistics for Depth {depth}:")
            means = results[f'depth_{depth}_mean']
            print(f"Mean gradient norm: {np.mean(means):.2e}")
            print(f"Max gradient norm:  {np.max(means):.2e}")
            print(f"Min gradient norm:  {np.min(means):.2e}")
        
    except Exception as e:
        print(f"Error in plot_average_gradient_norms for {set_type} set: {str(e)}")
        print("DataFrame shapes:", [df.shape for df in gradient_dfs])
        print("DataFrame columns:", [df.columns.tolist() for df in gradient_dfs])
        print("DataFrame head:", gradient_dfs[0].head())
        raise

def calculate_prediction_depth(model, data, node_idx, k=5):
    """
    Calculate prediction depth based on the new definition:
    - L = l if k-NN at layer (l-1) disagrees with final prediction
    - BUT k-NN at all layers >= l agrees with final prediction
    - L = 0 if all k-NN probes agree with final prediction
    """
    model.eval()
    with torch.no_grad():
        # Get final model prediction
        final_out = model(data.x, data.edge_index)
        final_pred = final_out[node_idx].argmax().item()
        
        # Get embeddings and k-NN predictions at each layer
        embeddings = []
        knn_preds = []
        h = data.x
        
        # Layer 0 (input)
        embeddings.append(h)
        dists = torch.cdist(h[node_idx].unsqueeze(0), h)
        _, indices = dists.topk(k + 1, largest=False)
        knn_pred = torch.mode(data.y[indices[0][1:]])[0].item()
        knn_preds.append(knn_pred)
        
        # Subsequent layers
        for conv in model.convs:
            h = conv(h, data.edge_index)
            embeddings.append(h)
            
            dists = torch.cdist(h[node_idx].unsqueeze(0), h)
            _, indices = dists.topk(k + 1, largest=False)
            knn_pred = torch.mode(data.y[indices[0][1:]])[0].item()
            knn_preds.append(knn_pred)
        
        # If all k-NN predictions match final prediction, depth is 0
        if all(pred == final_pred for pred in knn_preds):
            return 0
            
        # Find first layer where this and all subsequent k-NN preds match final
        for depth in range(1, len(embeddings)):
            if (knn_preds[depth-1] != final_pred and  # Previous layer disagrees
                all(pred == final_pred for pred in knn_preds[depth:])):  # All subsequent agree
                return depth
                
        # If no clear depth found, return final layer
        return len(embeddings) - 1















