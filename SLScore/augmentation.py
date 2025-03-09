import torch
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from torch_geometric.utils import k_hop_subgraph, remove_self_loops, add_self_loops

class NodeAugmentor:
    def __init__(self, 
                 augmentation_type: str = 'flip',  # 'flip', 'noise', or 'both'
                 feature_flip_rate: float = 0.05,  # Rate of features to flip
                 noise_std: float = 0.1,          # Standard deviation for Gaussian noise
                 num_augmentations: int = 5,      # Number of augmentations
                 mask_edges: bool = True,         # Whether to mask edges
                 mask_hops: int = 1,              # Number of hops to mask around target node
                 logger: Optional[logging.Logger] = None):  # Logger for tracking
        """
        Node augmentation using feature flipping and/or Gaussian noise.
        
        Args:
            augmentation_type: Type of augmentation to use ('flip', 'noise', or 'both')
            feature_flip_rate: Percentage of features to flip (default 5%)
            noise_std: Standard deviation for Gaussian noise (default 0.1)
            num_augmentations: Number of augmented versions to create (default 5)
            mask_edges: Whether to mask edges around target node
            mask_hops: Number of hops to mask (default 1)
            logger: Optional logger for tracking masking statistics
        """
        self.augmentation_type = augmentation_type
        self.feature_flip_rate = feature_flip_rate
        self.noise_std = noise_std
        self.num_augmentations = num_augmentations
        self.mask_edges = mask_edges
        self.mask_hops = mask_hops
        self.logger = logger
    
    def feature_flip(self, x: torch.Tensor, node_idx: int) -> Tuple[torch.Tensor, Dict]:
        """Flip features of target node"""
        aug_x = x.clone()
        num_features = x.size(1)
        num_flip = max(1, int(num_features * self.feature_flip_rate))
        
        # Randomly select features to flip
        flip_indices = np.random.choice(num_features, num_flip, replace=False)
        
        # Store original values
        original_values = aug_x[node_idx, flip_indices].cpu().numpy()
        
        # Flip selected features
        aug_x[node_idx, flip_indices] = 1 - aug_x[node_idx, flip_indices]
        
        # Get new values
        new_values = aug_x[node_idx, flip_indices].cpu().numpy()
        
        stats = {
            'type': 'feature_flip',
            'num_features_flipped': num_flip,
            'flip_rate_used': self.feature_flip_rate,
            'flipped_indices': flip_indices.tolist(),
            'feature_changes': [
                {'index': idx, 'original': orig, 'new': new}
                for idx, orig, new in zip(flip_indices, original_values, new_values)
            ]
        }
        
        return aug_x, stats
    
    def add_gaussian_noise(self, x: torch.Tensor, node_idx: int) -> Tuple[torch.Tensor, Dict]:
        """Add Gaussian noise to target node features"""
        aug_x = x.clone()
        
        # Generate Gaussian noise
        noise = torch.randn_like(aug_x[node_idx]) * self.noise_std
        
        # Store original values
        original_values = aug_x[node_idx].cpu().numpy()
        
        # Add noise
        aug_x[node_idx] = aug_x[node_idx] + noise
        
        # Get new values
        new_values = aug_x[node_idx].cpu().numpy()
        
        stats = {
            'type': 'gaussian_noise',
            'noise_std': self.noise_std,
            'max_noise_magnitude': float(noise.abs().max()),
            'avg_noise_magnitude': float(noise.abs().mean()),
            'feature_changes': [
                {'index': i, 'original': orig, 'new': new, 'noise': float(n)}
                for i, (orig, new, n) in enumerate(zip(original_values, new_values, noise.cpu().numpy()))
                if abs(n) > 1e-6  # Only log significant changes
            ]
        }
        
        return aug_x, stats
    
    def create_edge_mask(self, node_idx: int, edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, Dict]:
        """
        Create a mask for edges to prevent message passing through target node's neighborhood
        
        Args:
            node_idx: Target node index
            edge_index: Edge index tensor
            num_nodes: Total number of nodes in the graph
            
        Returns:
            Masked edge_index tensor and masking statistics
        """
        # Extract k-hop subgraph around the target node
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.mask_hops, edge_index, relabel_nodes=False,
            num_nodes=num_nodes, flow='source_to_target'
        )
        
        # Create a mask for edges to keep
        # We want to mask (exclude) all edges in the k-hop subgraph
        # edge_mask gives us the edges in the original edge_index that are in the subgraph,
        # so we negate it to get edges to keep
        edges_to_keep = ~edge_mask
        
        # Create masked edge index
        masked_edge_index = edge_index[:, edges_to_keep]
        
        # Calculate statistics
        total_edges = edge_index.size(1)
        masked_edges = total_edges - masked_edge_index.size(1)
        nodes_in_subgraph = len(subset)
        
        stats = {
            'total_edges': total_edges,
            'masked_edges': masked_edges,
            'masked_percentage': (masked_edges / total_edges) * 100 if total_edges > 0 else 0,
            'num_nodes_in_khop': nodes_in_subgraph,
            'k_hops': self.mask_hops
        }
        
        # Log masking statistics if logger is available
        if self.logger:
            self.logger.info(f"Edge masking for node {node_idx}:")
            self.logger.info(f"K-hop neighborhood size (k={self.mask_hops}): {nodes_in_subgraph} nodes")
            self.logger.info(f"Total edges in graph: {total_edges}")
            self.logger.info(f"Masked edges: {masked_edges} ({stats['masked_percentage']:.2f}%)")
            self.logger.info(f"Remaining edges after masking: {masked_edge_index.size(1)}")
            
        return masked_edge_index, stats
    
    def __call__(self, node_idx: int, x: torch.Tensor, 
                 edge_index: torch.Tensor) -> Tuple[List[Dict], List[Dict]]:
        """Create augmented versions using specified augmentation type(s)"""
        augmented_data = []
        aug_stats = []
        
        # Get original feature values for logging
        original_features = x[node_idx].cpu().numpy()
        
        # Create edge mask if specified
        masked_edge_index = edge_index
        masking_stats = None
        
        if self.mask_edges:
            masked_edge_index, masking_stats = self.create_edge_mask(
                node_idx, edge_index, x.size(0)
            )
        
        for aug_idx in range(self.num_augmentations):
            aug_x = x.clone()
            combined_stats = {
                'augmentation_index': aug_idx,
                'node_index': node_idx.item() if torch.is_tensor(node_idx) else node_idx,
                'augmentation_types': []
            }
            
            # Apply feature flipping if specified
            if self.augmentation_type in ['flip', 'both']:
                aug_x, flip_stats = self.feature_flip(aug_x, node_idx)
                combined_stats['flip_stats'] = flip_stats
                combined_stats['augmentation_types'].append('flip')
            
            # Apply Gaussian noise if specified
            if self.augmentation_type in ['noise', 'both']:
                aug_x, noise_stats = self.add_gaussian_noise(aug_x, node_idx)
                combined_stats['noise_stats'] = noise_stats
                combined_stats['augmentation_types'].append('noise')
            
            # Add general stats
            combined_stats['node_feature_stats'] = {
                'total_ones_before': int(original_features.sum()),
                'total_ones_after': int(aug_x[node_idx].cpu().numpy().sum()),
                'total_features': len(original_features)
            }
            
            # Add edge masking stats if available
            if masking_stats:
                combined_stats['edge_masking'] = masking_stats
                combined_stats['augmentation_types'].append('edge_masking')
            
            aug_data = {
                'x': aug_x,
                'edge_index': masked_edge_index if self.mask_edges else edge_index
            }
            
            augmented_data.append(aug_data)
            aug_stats.append(combined_stats)
        
        return augmented_data, aug_stats