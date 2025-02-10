import torch
import random
from torch_geometric.data import Data
from typing import List, Dict, Tuple

class NodeDropping:
    def __init__(self, drop_rate: float = 0.2, num_augmentations: int = 3):
        self.drop_rate = drop_rate
        self.num_augmentations = num_augmentations
    
    def __call__(self, data: Data) -> Tuple[List[Data], List[Dict]]:
        augmented_graphs = []
        aug_stats = []
        original_nodes = data.x.size(0)
        original_edges = data.edge_index.size(1) // 2  # Divide by 2 for undirected graphs
        
        for aug_idx in range(self.num_augmentations):
            # Calculate number of nodes to drop
            num_nodes = data.x.size(0)
            num_drop = int(num_nodes * self.drop_rate)
            
            # Select nodes to keep
            keep_nodes = sorted(random.sample(range(num_nodes), num_nodes - num_drop))
            
            # Create node mask for the kept nodes
            node_mask = torch.zeros(num_nodes, dtype=torch.bool)
            node_mask[keep_nodes] = 1
            
            # Update edge_index
            row, col = data.edge_index
            edge_mask = node_mask[row] & node_mask[col]
            new_edge_index = data.edge_index[:, edge_mask]
            
            # Remap node indices
            node_idx = torch.zeros(num_nodes, dtype=torch.long)
            node_idx[keep_nodes] = torch.arange(len(keep_nodes))
            new_edge_index = node_idx[new_edge_index]
            
            # Create new graph
            aug_data = Data(
                x=data.x[keep_nodes],
                edge_index=new_edge_index,
                y=data.y
            )
            
            # Collect statistics
            aug_stats.append({
                'original_nodes': original_nodes,
                'original_edges': original_edges,
                'augmented_nodes': aug_data.x.size(0),
                'augmented_edges': aug_data.edge_index.size(1) // 2,
                'drop_rate': self.drop_rate
            })
            
            augmented_graphs.append(aug_data)
        
        return augmented_graphs, aug_stats
