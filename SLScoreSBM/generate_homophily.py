import torch
from torch_geometric.data import Data, InMemoryDataset
import torch.nn.functional as F
import numpy as np
import os
import networkx as nx
from tqdm import tqdm
from itertools import repeat
from nodeli import li_node  # Import label informativeness function

class HomophilySBMDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(HomophilySBMDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['homophily_sbm_data.pt']

    def len(self):
        return self.slices['x'].size(0) - 1

    def get(self, idx):
        data = Data()
        for key in self.slices.keys():
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    def process(self):
        # Parameters - modified for a more challenging dataset
        n_nodes = 1000  # Total nodes
        n_classes = 4  # Number of classes
        sizes = [250, 250, 250, 250]  # Community sizes
        avg_degree = 10  # Reduced from 10 to make task harder
        feature_dim = 100
        n_graphs = 5
        feature_noise = 0.1
        
        # Create stats file
        stats_file = os.path.join(self.root, 'dataset_statistics.txt')
        with open(stats_file, 'w') as f:
            f.write("Synthetic SBM Dataset Statistics\n")
            f.write("==============================\n\n")
            f.write(f"Number of graphs: {n_graphs}\n")
            f.write(f"Nodes per graph: {n_nodes}\n")
            f.write(f"Number of classes: {n_classes}\n")
            f.write(f"Class sizes: {sizes}\n")
            f.write(f"Target average degree: {avg_degree}\n")
            f.write(f"Feature dimension: {feature_dim}\n\n")
            f.write("Per-Graph Statistics (Homophily and Informativeness):\n")
            f.write("------------------------------------------------\n")
        
        # Generate graphs across homophily spectrum
        homophily_levels = np.linspace(0.0, 0.5, n_graphs)
        
        data_list = []
        for idx, h_adj in enumerate(tqdm(homophily_levels, desc="Generating graphs")):
            # Calculate edge homophily from adjusted homophily
            adjust = 1/n_classes
            h_edge = h_adj * (1 - adjust) + adjust
            
            # Base probabilities from homophily
            p_avg = avg_degree / (n_nodes - 1)
            p_in_base = h_edge * p_avg * n_classes
            p_out_base = (1 - h_edge) * p_avg * n_classes / (n_classes - 1)
            
            # Create structured probability matrix for label informativeness
            # Instead of uniform p_out, create varying off-diagonal probabilities
            # This creates a more informative structure where labels better predict connections
            probs = np.zeros((n_classes, n_classes))
            
            # Fill diagonal (same-class connections)
            np.fill_diagonal(probs, p_in_base)
            
            # Fill off-diagonal with structured probabilities
            # We'll make connections between class i and i+1 more likely than others
            # This creates a ring-like structure in class connectivity
            for i in range(n_classes):
                for j in range(n_classes):
                    if i != j:
                        # Distance in the "class ring" - controls informativeness
                        dist = min((i - j) % n_classes, (j - i) % n_classes)
                        # Closer classes connect more, creating structure
                        factor = 1.0 - (0.5 * dist / (n_classes // 2))
                        probs[i, j] = p_out_base * factor
            
            # Ensure average connection probability stays the same
            # This preserves homophily while varying informativeness
            off_diag_avg = np.sum(probs) - np.trace(probs)
            off_diag_avg /= (n_classes * n_classes - n_classes)
            
            # Rescale to maintain target homophily
            for i in range(n_classes):
                for j in range(n_classes):
                    if i != j:
                        probs[i, j] = p_out_base * probs[i, j] / off_diag_avg
            
            # Ensure probabilities are valid
            probs = np.clip(probs, 0, 1)
            
            # Generate graph
            graph = nx.stochastic_block_model(sizes, probs, seed=42+idx)
            
            # Calculate label informativeness
            labels = np.array([i for i in range(n_classes) for _ in range(sizes[i])])
            informativeness = li_node(graph, labels)
            
            # Convert to edge index
            edge_index = torch.tensor(list(graph.edges())).t().contiguous()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

            # Generate features with noise
            x = torch.randn(n_nodes, feature_dim)
            x += feature_noise * torch.randn_like(x)
            #x = F.normalize(x, p=2, dim=1)
            
            # Generate labels 
            y = torch.tensor(labels)
            
            # Add shuffle to node ordering
            node_perm = torch.randperm(n_nodes)
            x = x[node_perm]
            y = y[node_perm]
            
            # Create masks - standard procedure
            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)

            # For each class
            for c in range(n_classes):
                idx_c = (y == c).nonzero().view(-1)
                n_per_class = idx_c.size(0)
                
                # Randomly shuffle indices
                perm = torch.randperm(n_per_class)
                idx_c = idx_c[perm]
                
                # Split: 60% train, 20% val, 20% test
                n_train = int(0.6 * n_per_class)
                n_val = int(0.2 * n_per_class)
                
                train_idx = idx_c[:n_train]
                val_idx = idx_c[n_train:n_train+n_val]
                test_idx = idx_c[n_train+n_val:]
                
                train_mask[train_idx] = True
                val_mask[val_idx] = True
                test_mask[test_idx] = True
            
            # Calculate actual statistics
            actual_edges = edge_index.shape[1] // 2
            actual_avg_degree = (2 * actual_edges) / n_nodes
            
            # For calculating actual homophily
            edge_list = edge_index.t().numpy()
            same_class_edges = sum(1 for i, j in edge_list[:len(edge_list)//2] 
                                 if y[i] == y[j])
            actual_edge_homophily = same_class_edges / (len(edge_list)//2)
            
            # Log statistics for this graph including informativeness
            with open(stats_file, 'a') as f:
                f.write(f"\nGraph {idx}:\n")
                f.write(f"  - Target adjusted homophily: {h_adj:.3f}\n")
                f.write(f"  - Actual edge homophily: {actual_edge_homophily:.3f}\n")
                f.write(f"  - Label informativeness: {informativeness:.3f}\n")
                f.write(f"  - Number of edges: {actual_edges}\n")
                f.write(f"  - Average degree: {actual_avg_degree:.2f}\n")
            
            # Create PyG Data object with additional statistics
            data = Data(
                x=x,
                edge_index=edge_index,
                y=y,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                homophily=torch.tensor([h_adj]),
                actual_homophily=torch.tensor([actual_edge_homophily]),
                label_informativeness=torch.tensor([informativeness]),
                avg_degree=torch.tensor([actual_avg_degree])
            )
            data_list.append(data)

        # Log final average statistics
        with open(stats_file, 'a') as f:
            avg_actual_homophily = np.mean([d.actual_homophily.item() for d in data_list])
            avg_degree_achieved = np.mean([d.avg_degree.item() for d in data_list])
            f.write("\nAverage Statistics Across All Graphs:\n")
            f.write("----------------------------------\n")
            f.write(f"Average degree achieved: {avg_degree_achieved:.2f}\n")
            f.write(f"Average edge homophily: {avg_actual_homophily:.3f}\n")

        # Use PyG's built-in collate function to process the data and slices
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
