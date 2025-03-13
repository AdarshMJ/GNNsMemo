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
from scipy.optimize import minimize

def H(*args):
    """Calculate entropy of probability distribution."""
    x = np.array(args)
    x = x[x > 0]  # Avoid log(0)
    return - (x * np.log2(x)).sum()

def create_sbm_graph(sizes, p_mat, seed=42):
    """Create stochastic block model graph."""
    # Ensure probabilities are valid (not exactly 0 or 1)
    p_mat = np.clip(p_mat, 1e-10, 0.9999)
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

def find_p0_for_homophily(homophily):
    """Convert homophily value to p_0 value for SBM."""
    return homophily + 0.25

def calculate_informativeness(p_0, p_1, p_2):
    """Calculate label informativeness from SBM probabilities."""
    # Ensure probabilities are valid for entropy calculation
    probs = np.array([p_0, p_1, p_2, p_2])
    probs = np.clip(probs, 1e-10, 1.0)
    probs = probs / probs.sum()  # Renormalize if needed
    
    return 1 - H(p_0, p_1, p_2, p_2) / np.log2(4)

def find_p1_p2_for_target_informativeness(target_informativeness, p_0):
    """
    Find p_1 and p_2 values that achieve target informativeness for a given p_0.
    
    Args:
        target_informativeness: Target label informativeness (0-1)
        p_0: Probability of connection within communities
        
    Returns:
        (p_1, p_2) tuple of probabilities that achieve target informativeness
    """
    def objective(p):
        p_1 = p[0]
        p_2 = (1 - p_0 - p_1) / 2  # Ensure probabilities sum correctly
        
        # Skip invalid probabilities
        if p_2 <= 0.01 or p_1 <= 0.01 or p_0 <= 0.01 or p_1 >= 0.99 or p_2 >= 0.99 or p_0 >= 0.99:
            return float('inf')
        
        # Calculate how far we are from the target informativeness
        current_info = calculate_informativeness(p_0, p_1, p_2)
        return abs(current_info - target_informativeness)
    
    # Try multiple starting points to find global minimum
    best_result = None
    best_error = float('inf')
    
    for start_p1 in np.linspace(0.05, 0.95, 10):
        # Skip invalid starting points
        p_2_start = (1 - p_0 - start_p1) / 2
        if p_2_start <= 0.01 or p_2_start >= 0.99:
            continue
            
        result = minimize(objective, [start_p1], bounds=[(0.01, 0.99)])
        
        if result.fun < best_error:
            best_error = result.fun
            best_result = result
    
    if best_result is None:
        raise ValueError(f"Could not find valid p_1, p_2 values for p_0={p_0} and target informativeness={target_informativeness}")
    
    p_1 = best_result.x[0]
    p_2 = (1 - p_0 - p_1) / 2
    
    # Safety check for valid probabilities (keep away from 0 and 1)
    p_1 = np.clip(p_1, 0.01, 0.99)
    p_2 = np.clip(p_2, 0.01, 0.99)
    
    # Verify the solution
    actual_info = calculate_informativeness(p_0, p_1, p_2)
    if abs(actual_info - target_informativeness) > 0.05:
        print(f"Warning: Could not achieve exact target informativeness. "
              f"Target: {target_informativeness:.4f}, Achieved: {actual_info:.4f}")
    
    return p_1, p_2

def generate_fixed_informativeness_graphs(
    target_informativeness,
    homophily_levels,
    feature_dim=100,
    noise_level=0.1,
    community_size=250,
    degree=10,
    random_features=False,
    seed=42
):
    """
    Generate graphs with fixed informativeness but varying homophily levels.
    
    Args:
        target_informativeness: Fixed informativeness (label information) value
        homophily_levels: List of homophily values to generate
        feature_dim: Dimension of node features
        noise_level: Level of noise to add to node features
        community_size: Size of each community
        degree: Average node degree
        random_features: If True, use random node features instead of community-correlated ones
        seed: Random seed
        
    Returns:
        List of PyG Data objects and corresponding homophily values
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Set up parameters
    K = degree / community_size
    sizes = [community_size] * 4
    total_nodes = sum(sizes)
    
    graphs = []
    achieved_info_values = []
    
    for i, homophily in enumerate(tqdm(homophily_levels, desc=f"Generating graphs (informativeness={target_informativeness:.2f})")):
        # Calculate p_0 from homophily
        p_0 = find_p0_for_homophily(homophily)
        
        # Find p_1 and p_2 that achieve target informativeness
        try:
            p_1, p_2 = find_p1_p2_for_target_informativeness(target_informativeness, p_0)
        except ValueError as e:
            print(f"Skipping homophily={homophily}: {str(e)}")
            continue
        
        # Verify probabilities are within valid range
        if not (0.01 <= p_0 <= 0.99 and 0.01 <= p_1 <= 0.99 and 0.01 <= p_2 <= 0.99):
            print(f"Skipping homophily={homophily}: Invalid probabilities p_0={p_0}, p_1={p_1}, p_2={p_2}")
            continue
            
        # Create probability matrix
        probs = np.array([
            [p_0, p_1, p_2, p_2],
            [p_1, p_0, p_2, p_2],
            [p_2, p_2, p_0, p_1],
            [p_2, p_2, p_1, p_0],
        ]) * K
        
        # Ensure valid probabilities for networkx
        probs = np.clip(probs, 1e-10, 0.9999)
        
        # Double-check informativeness
        informativeness = calculate_informativeness(p_0, p_1, p_2)
        
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
        data.homophily = homophily
        data.informativeness = informativeness
        data.target_informativeness = target_informativeness
        data.p_0 = p_0
        data.p_1 = p_1
        data.p_2 = p_2
        data.random_features = random_features
        
        graphs.append(data)
        achieved_info_values.append(informativeness)
    
    return graphs, achieved_info_values

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
    parser = argparse.ArgumentParser(description='Generate SBM graphs with fixed informativeness and varying homophily')
    parser.add_argument('--informativeness', type=float, default=0.5,
                        help='Fixed informativeness level (0.0-1.0)')
    parser.add_argument('--num_graphs', type=int, default=10,
                        help='Number of graphs to generate with varying homophily')
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
    parser.add_argument('--output_dir', type=str, default='data/fixed_info_sbm',
                        help='Output directory for saving graphs')
    parser.add_argument('--min_homophily', type=float, default=-0.2,
                        help='Minimum homophily value')
    parser.add_argument('--max_homophily', type=float, default=0.7,
                        help='Maximum homophily value')
    
    args = parser.parse_args()
    
    # Generate homophily levels
    homophily_levels = np.linspace(args.min_homophily, args.max_homophily, args.num_graphs)
    
    # Generate graphs
    graphs, info_values = generate_fixed_informativeness_graphs(
        args.informativeness,
        homophily_levels,
        feature_dim=args.feature_dim,
        noise_level=args.noise_level,
        community_size=args.community_size,
        degree=args.degree,
        random_features=args.random_features,
        seed=args.seed
    )
    
    # Create folder name with feature type info
    feature_type = "random" if args.random_features else "correlated"
    output_folder = os.path.join(args.output_dir, f"info_{args.informativeness:.2f}")
    
    # Save graphs
    save_graphs(graphs, output_folder, name_prefix="sbm")
    
    # Calculate homophily values actually achieved
    homophily_values = [graph.homophily for graph in graphs]
    
    print(f"\nGenerated {len(graphs)} graphs with fixed informativeness {args.informativeness:.2f}")
    print(f"Using {feature_type} features")
    print(f"Homophily range: {min(homophily_values):.4f} - {max(homophily_values):.4f}")
    print(f"Actual informativeness range: {min(info_values):.4f} - {max(info_values):.4f}")
    print(f"Saved to: {output_folder}")
    
    # Save metadata
    with open(os.path.join(output_folder, "metadata.txt"), "w") as f:
        f.write(f"Target informativeness: {args.informativeness}\n")
        f.write(f"Number of graphs: {len(graphs)}\n")
        f.write(f"Feature dimension: {args.feature_dim}\n")
        f.write(f"Feature type: {feature_type}\n")
        f.write(f"Noise level: {args.noise_level}\n")
        f.write(f"Community size: {args.community_size}\n")
        f.write(f"Average degree: {args.degree}\n\n")
        
        f.write("Homophily and informativeness levels:\n")
        for i, (graph, info) in enumerate(zip(graphs, info_values)):
            f.write(f"{i}: Homophily={graph.homophily:.6f}, Informativeness={info:.6f}\n")

if __name__ == "__main__":
    main()