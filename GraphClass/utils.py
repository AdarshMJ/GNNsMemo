import torch
import random
import numpy as np
import scipy.sparse as sp
import copy

def delete_row_col(input_matrix, drop_list, only_row=False):
    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    if only_row:
        return input_matrix[remain_list, :]
    return input_matrix[remain_list, :][:, remain_list]

def aug_random_edge(input_adj, drop_percent=0.2):
    """Randomly drop edges from the graph."""
    # Convert to dense numpy array if it's a sparse matrix
    if sp.issparse(input_adj):
        adj_dense = input_adj.todense()
    else:
        adj_dense = input_adj
    
    # Get upper triangular indices (to avoid double counting edges)
    rows, cols = np.triu_indices(adj_dense.shape[0], k=1)
    edge_indices = [(r, c) for r, c in zip(rows, cols) if adj_dense[r, c] != 0]
    
    # Calculate number of edges to drop
    num_edges = len(edge_indices)
    num_drop = int(num_edges * drop_percent)
    
    # Randomly select edges to drop
    drop_indices = random.sample(edge_indices, num_drop)
    
    # Create new adjacency matrix
    new_adj = np.array(adj_dense)
    for i, j in drop_indices:
        new_adj[i, j] = 0
        new_adj[j, i] = 0  # Maintain symmetry
    
    # Convert back to sparse matrix
    return sp.csr_matrix(new_adj)

def aug_drop_node(input_fea, input_adj, drop_percent=0.2):
    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)

    node_num = input_fea.shape[0]
    drop_num = int(node_num * drop_percent)
    all_node_list = [i for i in range(node_num)]

    drop_node_list = sorted(random.sample(all_node_list, drop_num))

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj