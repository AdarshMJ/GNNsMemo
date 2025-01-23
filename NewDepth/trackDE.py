import torch
import torch_geometric as pyg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
import time

def dirichlet_energy(x, edge_index, edge_weights):
    """Calculate Dirichlet energy for given node features."""
    try:
        # Ensure all tensors are on the same device and type
        device = x.device
        edge_index = edge_index.to(device)
        edge_weights = edge_weights.to(device)
        
        # Calculate differences between connected nodes
        diff = x[edge_index[0]] - x[edge_index[1]]  # [E, F]
        
        # Calculate squared differences and weight them
        squared_diff = torch.sum(diff * diff, dim=1)  # [E]
        weighted_diff = edge_weights * squared_diff  # [E]
        
        # Sum all contributions and divide by 2
        energy = torch.sum(weighted_diff) / 2
        
        return float(energy.cpu().item())  # Convert to Python float
    except Exception as e:
        print(f"Error in dirichlet_energy: {e}")
        return 0.0

class EnergyTracker:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.history = defaultdict(list)
        
        # Precompute adjacency matrix and weights
        device = data.x.device
        edge_index = data.edge_index.to(device)
        
        # Use binary adjacency matrix for Dirichlet energy calculation
        A = pyg.utils.to_dense_adj(edge_index)[0]
        self.edge_index, self.edge_weights = pyg.utils.dense_to_sparse(A)
        self.edge_weights = self.edge_weights.to(device)
        
        # Register hooks to capture intermediate activations
        self.activations = []
        def hook_fn(module, input, output):
            self.activations.append(output)
        
        # Register hook for input layer
        self.input_hook = self.model.register_forward_hook(
            lambda m, inp, out: self.activations.append(inp[0]))
        
        # Register hooks for each layer
        for conv in self.model.convs:
            conv.register_forward_hook(hook_fn)
    
    def get_current_values(self):
        """Get current energy and norm values for all layers."""
        self.model.eval()
        self.activations = []  # Clear previous activations
        
        with torch.no_grad():
            # Do a single forward pass
            _ = self.model(self.data.x, self.data.edge_index)
            
            # Calculate energies and norms for all captured activations
            energies = []
            norms = []
            
            for x in self.activations:
                energy = dirichlet_energy(x, self.edge_index, self.edge_weights)
                norm = float(torch.norm(x, p=2).cpu().item())
                energies.append(energy)
                norms.append(norm)
        
        return energies, norms
    
    def track_step(self, epoch):
        """Track energies and norms for current model state."""
        energies, norms = self.get_current_values()
        
        for i, (energy, norm) in enumerate(zip(energies, norms)):
            self.history[f'layer_{i}_energy'].append(energy)
            self.history[f'layer_{i}_norm'].append(norm)
        self.history['epochs'].append(epoch)
    
    def plot_training_dynamics(self, save_path=None):
        """Plot the evolution of energies and norms during training."""
        num_layers = len(self.model.convs)  # +1 for initial layer
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot energies
        for i in range(num_layers):
            values = self.history[f'layer_{i}_energy']
            if any(v != 0 for v in values):  # Only plot if we have non-zero values
                ax1.plot(self.history['epochs'], 
                        values,
                        label=f'Layer {i}',
                        marker='o',
                        markersize=4)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Dirichlet Energy')
        if ax1.get_ylim()[0] > 0:  # Only use log scale if all values are positive
            ax1.set_yscale('log')
        ax1.legend()
        ax1.set_title('Dirichlet Energy During Training')
        ax1.grid(True, alpha=0.3)
        
        # Plot norms
        for i in range(num_layers):
            values = self.history[f'layer_{i}_norm']
            if any(v != 0 for v in values):  # Only plot if we have non-zero values
                ax2.plot(self.history['epochs'],
                        values,
                        label=f'Layer {i}',
                        marker='o',
                        markersize=4)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Layer Norm')
        if ax2.get_ylim()[0] > 0:  # Only use log scale if all values are positive
            ax2.set_yscale('log')
        ax2.legend()
        ax2.set_title('Layer Norms During Training')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            # Save the plot
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
            # Save the numerical data
            save_dir = Path(save_path).parent
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save energy values
            energy_df = pd.DataFrame({
                'epoch': self.history['epochs'],
                **{f'layer_{i}_energy': self.history[f'layer_{i}_energy'] 
                   for i in range(num_layers)}
            })
            energy_df.to_csv(save_dir / f'energy_values_{timestamp}.csv', index=False)
            
            # Save norm values
            norm_df = pd.DataFrame({
                'epoch': self.history['epochs'],
                **{f'layer_{i}_norm': self.history[f'layer_{i}_norm'] 
                   for i in range(num_layers)}
            })
            norm_df.to_csv(save_dir / f'norm_values_{timestamp}.csv', index=False)
        
        plt.close()

def analyze_model_energies(model, data, save_dir, split_idx):
    """Analyze Dirichlet energy and norms for model layers."""
    tracker = EnergyTracker(model, data)
    energies, norms = tracker.get_current_values()
    
    # Create visualization
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(15, 6))
    
    # Plot energies and norms
    layers = range(len(energies))
    plt.plot(layers, energies, 'b-', label='Dirichlet Energy', marker='o')
    plt.plot(layers, norms, 'r--', label='Layer Norm', marker='s')
    
    plt.title('Dirichlet Energy and Layer Norms Across Model Layers')
    plt.xlabel('Layer')
    plt.ylabel('Value')
    if min(energies + norms) > 0:  # Only use log scale if all values are positive
        plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_dir / f'layer_energies_split_{split_idx}_{timestamp}.png',
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Save numerical results
    results_df = pd.DataFrame({
        'layer': layers,
        'dirichlet_energy': energies,
        'layer_norm': norms
    })
    results_df.to_csv(save_dir / f'layer_energies_split_{split_idx}_{timestamp}.csv',
                      index=False)
    
    return results_df