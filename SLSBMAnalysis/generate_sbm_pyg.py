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

def create_sbm_graph(sizes, p_mat, seed=42):
    """Create stochastic block model graph."""
    return nx.stochastic_block_model(sizes, p_mat, seed=seed, directed=False, selfloops=False)

def generate_random_features(num_nodes, feature_dim=100, seed=42):
    """
    Generate completely random node features with no correlation to community labels.
    
    Args:
        num_nodes: Number of nodes in the graph
        feature_dim: Dimension of node features
        seed: Random seed
        
    Returns:
        torch.Tensor of shape [num_nodes, feature_dim]
    """
    np.random.seed(seed)
    features = np.random.normal(0, 1, size=(num_nodes, feature_dim))
    return torch.FloatTensor(features)

def generate_correlated_features(graph, community_labels, feature_dim=100, noise_level=0.1, seed=42):
    """
    Generate node features that are correlated with community labels.
    
    Args:
        graph: NetworkX graph
        community_labels: Community labels for each node
        feature_dim: Dimension of node features
        noise_level: Standard deviation of the noise to add
        seed: Random seed
        
    Returns:
        torch.Tensor of shape [num_nodes, feature_dim]
    """
    np.random.seed(seed)
    num_nodes = len(graph.nodes)
    num_communities = len(np.unique(community_labels))
    
    # Create distinct base vectors for each community
    base_vectors = np.random.normal(0, 1, size=(num_communities, feature_dim))
    
    # Assign features based on community membership
    features = np.zeros((num_nodes, feature_dim))
    for i, label in enumerate(community_labels):
        # Add the base vector for this community
        features[i] = base_vectors[label]
        # Add Gaussian noise
        features[i] += np.random.normal(0, noise_level, feature_dim)
    
    return torch.FloatTensor(features)

def create_train_val_test_masks(num_nodes, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Create train/val/test masks with specified ratios."""
    indices = np.arange(num_nodes)
    
    # Split indices into train+val and test
    train_val_idx, test_idx = train_test_split(
        indices, test_size=1.0 - (train_ratio + val_ratio), random_state=seed
    )
    
    # Split train+val into train and val
    train_idx, val_idx = train_test_split(
        train_val_idx, 
        test_size=val_ratio / (train_ratio + val_ratio), 
        random_state=seed
    )
    
    # Create binary masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    return train_mask, val_mask, test_mask

def nx_to_pyg(nx_graph, features, labels, train_mask, val_mask, test_mask):
    """Convert NetworkX graph to PyG Data object."""
    # Create edge_index
    edges = list(nx_graph.edges)
    edge_index = torch.tensor([[u, v] for u, v in edges] + 
                              [[v, u] for u, v in edges], dtype=torch.long).t()
    
    # Create PyG Data object
    data = Data(
        x=features,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    
    return data

def generate_fixed_homophily_graphs(
    target_homophily,
    num_graphs,
    feature_dim=100,
    noise_level=0.1,
    community_size=250,
    degree=10,
    random_features=False,
    seed=42
):
    """
    Generate graphs with fixed homophily but varying informativeness levels.
    
    Args:
        target_homophily: Fixed homophily (modularity) value
        num_graphs: Number of graphs to generate
        feature_dim: Dimension of node features
        noise_level: Level of noise to add to node features
        community_size: Size of each community
        degree: Average node degree
        random_features: If True, use random node features instead of community-correlated ones
        seed: Random seed
        
    Returns:
        List of PyG Data objects and informativeness values
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set up parameters
    K = degree / community_size
    sizes = [community_size] * 4
    total_nodes = sum(sizes)
    
    # Calculate p_0 from target homophily
    p_0 = target_homophily + 0.25
    
    # Generate candidate informativeness levels by varying p_1
    candidates = []
    for p_1 in np.linspace(0.01, 0.99, 100):
        p_2 = (1 - p_0 - p_1) / 2
        if p_2 <= 0:
            continue
        
        # Calculate informativeness
        li = 1 - H(p_0, p_1, p_2, p_2) / np.log2(4)
        candidates.append((li, p_1, p_2))
    
    # Sort by informativeness
    candidates.sort()
    
    if len(candidates) < num_graphs:
        raise ValueError(f"Could only generate {len(candidates)} distinct informativeness levels, " 
                         f"but {num_graphs} were requested.")
    
    # Select evenly spaced informativeness levels
    indices = np.linspace(0, len(candidates) - 1, num_graphs).astype(int)
    selected = [candidates[i] for i in indices]
    
    graphs = []
    info_values = []
    
    for i, (li, p_1, p_2) in enumerate(tqdm(selected, desc=f"Generating graphs (homophily={target_homophily:.2f})")):
        # Create probability matrix
        probs = np.array([
            [p_0, p_1, p_2, p_2],
            [p_1, p_0, p_2, p_2],
            [p_2, p_2, p_0, p_1],
            [p_2, p_2, p_1, p_0],
        ]) * K
        
        # Create graph
        graph_seed = seed + i
        nx_graph = create_sbm_graph(sizes, probs, seed=graph_seed)
        
        # Get node labels (community IDs)
        labels = np.zeros(total_nodes, dtype=int)
        start_idx = 0
        for comm_id, size in enumerate(sizes):
            labels[start_idx:start_idx + size] = comm_id
            start_idx += size
        
        # Convert to PyTorch tensors
        labels_tensor = torch.LongTensor(labels)
        
        # Generate node features based on parameter choice
        if random_features:
            features = generate_random_features(total_nodes, feature_dim=feature_dim, seed=graph_seed)
        else:
            features = generate_correlated_features(
                nx_graph, labels, feature_dim=feature_dim, 
                noise_level=noise_level, seed=graph_seed
            )
        
        # Create train/val/test masks
        train_mask, val_mask, test_mask = create_train_val_test_masks(
            total_nodes, seed=graph_seed
        )
        
        # Convert to PyG Data
        data = nx_to_pyg(
            nx_graph, features, labels_tensor, 
            train_mask, val_mask, test_mask
        )
        
        # Add metadata
        data.homophily = target_homophily
        data.informativeness = li
        data.p_0 = p_0
        data.p_1 = p_1
        data.p_2 = p_2
        data.random_features = random_features
        
        graphs.append(data)
        info_values.append(li)
    
    return graphs, info_values

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
    parser = argparse.ArgumentParser(description='Generate SBM graphs with fixed homophily and varying informativeness')
    parser.add_argument('--homophily', type=float, default=0.5,
                        help='Fixed homophily level (0.0-1.0)')
    parser.add_argument('--num_graphs', type=int, default=10,
                        help='Number of graphs to generate with varying informativeness')
    parser.add_argument('--feature_dim', type=int, default=100,
                        help='Dimension of node features')
    parser.add_argument('--noise_level', type=float, default=0.1,
                        help='Noise level for node features')
    parser.add_argument('--community_size', type=int, default=250,
                        help='Size of each community')
    parser.add_argument('--degree', type=int, default=10,
                        help='Average node degree')
    parser.add_argument('--random_features', action='store_true',
                        help='Use random node features instead of community-correlated features')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='data/pyg_sbm',
                        help='Output directory for saving graphs')
    
    args = parser.parse_args()
    
    # Generate graphs
    graphs, info_values = generate_fixed_homophily_graphs(
        args.homophily,
        args.num_graphs,
        feature_dim=args.feature_dim,
        noise_level=args.noise_level,
        community_size=args.community_size,
        degree=args.degree,
        random_features=args.random_features,
        seed=args.seed
    )
    
    # Create folder name with feature type info
    feature_type = "random" if args.random_features else "correlated"
    output_folder = os.path.join(args.output_dir, f"hom_{args.homophily:.2f}")
    
    # Save graphs
    save_graphs(graphs, output_folder, name_prefix="sbm")
    
    print(f"\nGenerated {len(graphs)} graphs with fixed homophily {args.homophily:.2f}")
    print(f"Using {feature_type} features")
    print(f"Informativeness range: {min(info_values):.4f} - {max(info_values):.4f}")
    print(f"Saved to: {output_folder}")
    
    # Save metadata
    with open(os.path.join(output_folder, "metadata.txt"), "w") as f:
        f.write(f"Homophily: {args.homophily}\n")
        f.write(f"Number of graphs: {args.num_graphs}\n")
        f.write(f"Feature dimension: {args.feature_dim}\n")
        f.write(f"Feature type: {feature_type}\n")
        f.write(f"Noise level: {args.noise_level}\n")
        f.write(f"Community size: {args.community_size}\n")
        f.write(f"Average degree: {args.degree}\n\n")
        
        f.write("Informativeness levels:\n")
        for i, li in enumerate(info_values):
            f.write(f"{i}: {li:.6f}\n")

if __name__ == "__main__":
    main()