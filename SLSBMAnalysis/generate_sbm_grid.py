import os
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
import argparse
from tqdm import tqdm
import pickle
from scipy import stats
from sklearn.model_selection import train_test_split

def H(*args):
    """Calculate entropy of probability distribution."""
    x = np.array(args)
    x = x[x > 0]  # Avoid log(0)
    return - (x * np.log2(x)).sum()

# Reuse these functions from generate_sbm_pyg.py
def create_sbm_graph(sizes, p_mat, seed=42):
    """Create stochastic block model graph."""
    return nx.stochastic_block_model(sizes, p_mat, seed=seed, directed=False, selfloops=False)

def generate_random_features(num_nodes, feature_dim=100, seed=42):
    """Generate random node features."""
    np.random.seed(seed)
    features = np.random.normal(0, 1, size=(num_nodes, feature_dim))
    return torch.FloatTensor(features)

def generate_correlated_features(graph, community_labels, feature_dim=100, noise_level=0.1, seed=42):
    """Generate community-correlated node features."""
    np.random.seed(seed)
    num_nodes = len(graph.nodes)
    num_communities = len(np.unique(community_labels))
    
    base_vectors = np.random.normal(0, 1, size=(num_communities, feature_dim))
    features = np.zeros((num_nodes, feature_dim))
    
    for i, label in enumerate(community_labels):
        features[i] = base_vectors[label] + np.random.normal(0, noise_level, feature_dim)
    
    return torch.FloatTensor(features)

def create_train_val_test_masks(num_nodes, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Create train/val/test masks."""
    indices = np.arange(num_nodes)
    train_val_idx, test_idx = train_test_split(
        indices, test_size=1.0 - (train_ratio + val_ratio), random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio / (train_ratio + val_ratio), random_state=seed
    )
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    return train_mask, val_mask, test_mask

def nx_to_pyg(nx_graph, features, labels, train_mask, val_mask, test_mask):
    """Convert NetworkX graph to PyG Data object."""
    edges = list(nx_graph.edges)
    edge_index = torch.tensor([[u, v] for u, v in edges] + 
                            [[v, u] for u, v in edges], dtype=torch.long).t()
    
    data = Data(
        x=features,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    return data

def generate_grid_graphs(
    num_homophily_levels=10,
    num_info_levels=10,
    feature_dim=100,
    noise_level=0.1,
    community_size=250,
    degree=10,
    random_features=False,
    min_distance=0.03,
    seed=42
):
    """
    Generate graphs with varying homophily and informativeness levels.
    
    Args:
        num_homophily_levels: Number of homophily levels to sample
        num_info_levels: Number of informativeness levels to sample per homophily
        feature_dim: Dimension of node features
        noise_level: Level of noise to add to node features
        community_size: Size of each community
        degree: Average node degree
        random_features: If True, use random node features
        min_distance: Minimum Euclidean distance between (homophily, informativeness) points
        seed: Random seed
        
    Returns:
        List of PyG Data objects
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set up parameters
    K = degree / community_size
    sizes = [community_size] * 4
    total_nodes = sum(sizes)
    
    # Initialize storage for valid configurations
    valid_configs = []
    homophily_values = []
    info_values = []
    
    # Generate candidate configurations
    with tqdm(total=num_homophily_levels * num_info_levels, desc="Finding valid configurations") as pbar:
        for p_0 in np.linspace(0.26, 0.99, num_homophily_levels):  # p_0 > 0.25 for positive homophily
            for p_1 in np.linspace(0.01, 0.99, num_info_levels):
                p_2 = (1 - p_0 - p_1) / 2
                if p_2 <= 0.01:  # Ensure positive probability
                    continue
                
                homophily = p_0 - 0.25
                informativeness = 1 - H(p_0, p_1, p_2, p_2) / np.log2(4)
                
                # Check distance from existing points
                too_close = False
                for h, i in zip(homophily_values, info_values):
                    if np.sqrt((homophily - h) ** 2 + (informativeness - i) ** 2) < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    valid_configs.append((p_0, p_1, p_2))
                    homophily_values.append(homophily)
                    info_values.append(informativeness)
                
                pbar.update(1)
    
    print(f"Found {len(valid_configs)} valid configurations")
    
    # Generate graphs for each valid configuration
    graphs = []
    
    for idx, (p_0, p_1, p_2) in enumerate(tqdm(valid_configs, desc="Generating graphs")):
        # Create probability matrix
        probs = np.array([
            [p_0, p_1, p_2, p_2],
            [p_1, p_0, p_2, p_2],
            [p_2, p_2, p_0, p_1],
            [p_2, p_2, p_1, p_0],
        ]) * K
        
        # Create graph
        graph_seed = seed + idx
        nx_graph = create_sbm_graph(sizes, probs, seed=graph_seed)
        
        # Generate labels
        labels = np.zeros(total_nodes, dtype=int)
        start_idx = 0
        for comm_id, size in enumerate(sizes):
            labels[start_idx:start_idx + size] = comm_id
            start_idx += size
        
        labels_tensor = torch.LongTensor(labels)
        
        # Generate features
        if random_features:
            features = generate_random_features(total_nodes, feature_dim=feature_dim, seed=graph_seed)
        else:
            features = generate_correlated_features(
                nx_graph, labels, feature_dim=feature_dim, 
                noise_level=noise_level, seed=graph_seed
            )
        
        # Create masks
        train_mask, val_mask, test_mask = create_train_val_test_masks(
            total_nodes, seed=graph_seed
        )
        
        # Convert to PyG Data
        data = nx_to_pyg(
            nx_graph, features, labels_tensor, 
            train_mask, val_mask, test_mask
        )
        
        # Add metadata
        data.homophily = homophily_values[idx]
        data.informativeness = info_values[idx]
        data.p_0 = p_0
        data.p_1 = p_1
        data.p_2 = p_2
        data.random_features = random_features
        
        graphs.append(data)
    
    return graphs, homophily_values, info_values

def save_graphs(graphs, output_dir, name_prefix="graph"):
    """Save PyG Data objects to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual graphs
    for i, data in enumerate(graphs):
        torch.save(data, os.path.join(output_dir, f"{name_prefix}_{i}.pt"))
    
    # Also save as a list for convenience
    with open(os.path.join(output_dir, f"{name_prefix}_list.pkl"), "wb") as f:
        pickle.dump(graphs, f)

def main():
    parser = argparse.ArgumentParser(description='Generate SBM graphs with varying homophily and informativeness')
    parser.add_argument('--num_homophily', type=int, default=10,
                        help='Number of homophily levels to sample')
    parser.add_argument('--num_info', type=int, default=10,
                        help='Number of informativeness levels per homophily')
    parser.add_argument('--feature_dim', type=int, default=100,
                        help='Dimension of node features')
    parser.add_argument('--noise_level', type=float, default=0.1,
                        help='Noise level for node features')
    parser.add_argument('--community_size', type=int, default=250,
                        help='Size of each community')
    parser.add_argument('--degree', type=int, default=10,
                        help='Average node degree')
    parser.add_argument('--random_features', action='store_true',
                        help='Use random node features')
    parser.add_argument('--min_distance', type=float, default=0.03,
                        help='Minimum distance between configurations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='data/sbm_grid',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Generate graphs
    graphs, homophily_values, info_values = generate_grid_graphs(
        num_homophily_levels=args.num_homophily,
        num_info_levels=args.num_info,
        feature_dim=args.feature_dim,
        noise_level=args.noise_level,
        community_size=args.community_size,
        degree=args.degree,
        random_features=args.random_features,
        min_distance=args.min_distance,
        seed=args.seed
    )
    
    # Create output directory with feature type info
    feature_type = "random" if args.random_features else "correlated"
    output_folder = os.path.join(args.output_dir, feature_type)
    
    # Save graphs
    save_graphs(graphs, output_folder, name_prefix="sbm")
    
    print(f"\nGenerated {len(graphs)} graphs")
    print(f"Using {feature_type} features")
    print(f"Homophily range: {min(homophily_values):.4f} - {max(homophily_values):.4f}")
    print(f"Informativeness range: {min(info_values):.4f} - {max(info_values):.4f}")
    print(f"Saved to: {output_folder}")
    
    # Save metadata
    with open(os.path.join(output_folder, "metadata.txt"), "w") as f:
        f.write(f"Number of graphs: {len(graphs)}\n")
        f.write(f"Feature dimension: {args.feature_dim}\n")
        f.write(f"Feature type: {feature_type}\n")
        f.write(f"Noise level: {args.noise_level}\n")
        f.write(f"Community size: {args.community_size}\n")
        f.write(f"Average degree: {args.degree}\n\n")
        
        f.write("Graph properties:\n")
        for i, (h, li) in enumerate(zip(homophily_values, info_values)):
            f.write(f"{i}: Homophily={h:.6f}, Informativeness={li:.6f}\n")

if __name__ == "__main__":
    main()
