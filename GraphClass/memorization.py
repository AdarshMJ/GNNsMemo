import torch
import numpy as np
from typing import List, Tuple, Dict
from torch_geometric.data import Dataset, Data
from augmentation import NodeDropping  # Add this import
import matplotlib.pyplot as plt
import torch.nn.functional as F

def split_dataset(dataset: Dataset) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Two-stage splitting:
    1. First split into train (80%) and test (20%)
    2. Then split train into shared, candidate, and independent sets
    """
    num_graphs = len(dataset)
    indices = list(range(num_graphs))
    np.random.shuffle(indices)
    
    # First split: 80-20 for train-test
    train_size = int(0.8 * num_graphs)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Second split: Split train_indices into shared, candidate, and independent
    # Distribute train set as: 50% shared, 25% candidate, 25% independent
    num_train = len(train_indices)
    shared_size = int(0.5 * num_train)
    candidate_size = int(0.25 * num_train)
    
    shared_indices = train_indices[:shared_size]
    candidate_indices = train_indices[shared_size:shared_size + candidate_size]
    independent_indices = train_indices[shared_size + candidate_size:]
    
    return shared_indices, candidate_indices, independent_indices, test_indices

def calculate_memorization_score(model_f, model_g, candidate_graphs: List[Data], 
                               augmentor: NodeDropping, device: torch.device, logger=None) -> Tuple[float, List[float], List[float], List[float]]:
    """Calculate memorization scores with optional logging of augmentation details"""
    mem_scores = []
    f_scores = []
    g_scores = []
    
    for idx, graph in enumerate(candidate_graphs):
        augs, aug_stats = augmentor(graph)
        
        if logger:
            logger.info(f"\nAugmentation statistics for graph {idx}:")
            for i, stats in enumerate(aug_stats):
                logger.info(f"Augmentation {i+1}:")
                logger.info(f"  Original nodes: {stats['original_nodes']}, edges: {stats['original_edges']}")
                logger.info(f"  After dropping: nodes: {stats['augmented_nodes']}, edges: {stats['augmented_edges']}")
                logger.info(f"  Nodes removed: {stats['original_nodes'] - stats['augmented_nodes']}")
                logger.info(f"  Edges removed: {stats['original_edges'] - stats['augmented_edges']}")
        
        aug_pairs = [(augs[i], augs[j]) 
                    for i in range(len(augs)) 
                    for j in range(i+1, len(augs))]
        
        f_dists = []
        g_dists = []
        
        for aug1, aug2 in aug_pairs:
            aug1, aug2 = aug1.to(device), aug2.to(device)
            with torch.no_grad():
                _, emb_f1 = model_f(aug1.x, aug1.edge_index, None, return_emb=True)
                _, emb_f2 = model_f(aug2.x, aug2.edge_index, None, return_emb=True)
                _, emb_g1 = model_g(aug1.x, aug1.edge_index, None, return_emb=True)
                _, emb_g2 = model_g(aug2.x, aug2.edge_index, None, return_emb=True)
            
            # Normalize embeddings
            emb_f1 = F.normalize(emb_f1, p=2, dim=1)
            emb_f2 = F.normalize(emb_f2, p=2, dim=1)
            emb_g1 = F.normalize(emb_g1, p=2, dim=1)
            emb_g2 = F.normalize(emb_g2, p=2, dim=1)
            
            f_dist = torch.norm(emb_f1 - emb_f2, p=2).item()
            g_dist = torch.norm(emb_g1 - emb_g2, p=2).item()
            
            f_dists.append(f_dist)
            g_dists.append(g_dist)
        
        f_score = np.mean(f_dists) if f_dists else 0.0
        g_score = np.mean(g_dists) if g_dists else 0.0
        
        # Calculate memorization score as the normalized difference
        mem_score = g_score - f_score
        if abs(f_score) > 1e-8 or abs(g_score) > 1e-8:
            max_abs = max(abs(g_score), abs(f_score))
            mem_score = mem_score / max_abs if max_abs > 1e-8 else 0.0
        
        f_scores.append(f_score)
        g_scores.append(g_score)
        mem_scores.append(mem_score)
    
    final_mem_score = np.mean(mem_scores) if mem_scores else 0.0
    return final_mem_score, f_scores, g_scores, mem_scores

def plot_memorization_analysis(f_scores: List[float], g_scores: List[float], 
                             mem_scores: List[float], save_path: str,
                             candidate_graphs: List[Data] = None,
                             shared_graphs: List[Data] = None,
                             independent_graphs: List[Data] = None,
                             model_f=None, model_g=None, augmentor=None, device=None):
    """Plot alignment scores and memorization score distribution"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Scatter plot of alignment scores
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(f_scores, g_scores, c=mem_scores, 
                         cmap='viridis', alpha=0.6)
    
    # Add y=x line
    max_val = max(max(f_scores), max(g_scores))
    min_val = min(min(f_scores), min(g_scores))
    plt.plot([min_val, max_val], [min_val, max_val], 
             'k--', alpha=0.5, label='y=x')
    
    plt.colorbar(scatter, label='Memorization Score')
    plt.xlabel('Model f Alignment Score')
    plt.ylabel('Model g Alignment Score')
    plt.title('Alignment Score Analysis')
    plt.legend()
    
    # Plot 2: Memorization score distribution
    plt.subplot(1, 2, 2)
    
    # Calculate memorization scores for each set if all required parameters are provided
    if all(x is not None for x in [candidate_graphs, shared_graphs, independent_graphs, 
                                  model_f, model_g, augmentor, device]):
        # Calculate scores for shared and independent graphs
        _, _, _, shared_scores = calculate_memorization_score(
            model_f, model_g, shared_graphs, augmentor, device
        )
        _, _, _, independent_scores = calculate_memorization_score(
            model_f, model_g, independent_graphs, augmentor, device
        )
        
        # Plot histograms for each set
        plt.hist(mem_scores, bins=20, alpha=0.5, label='Candidate', density=True)
        plt.hist(shared_scores, bins=20, alpha=0.5, label='Shared', density=True)
        plt.hist(independent_scores, bins=20, alpha=0.5, label='Independent', density=True)
    else:
        # Just plot histogram of memorization scores for candidate graphs
        plt.hist(mem_scores, bins=20, alpha=0.7, density=True, label='Candidate')
        
    plt.xlabel('Memorization Score')
    plt.ylabel('Frequency')
    plt.title('Memorization Score Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_aggregate_memorization_analysis(all_seed_metrics: List[Dict], save_path: str):
    """Create an aggregate plot showing memorization scores across all seeds in the same style as individual plots"""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Scatter plot of model accuracies across all seeds
    plt.subplot(1, 2, 1)
    f_accs = [m['model_f_acc'] for m in all_seed_metrics]
    g_accs = [m['model_g_acc'] for m in all_seed_metrics]
    mem_scores = [m['memorization_score'] for m in all_seed_metrics]
    
    scatter = plt.scatter(f_accs, g_accs, c=mem_scores, 
                         cmap='viridis', alpha=0.6)
    
    # Add y=x line
    max_val = max(max(f_accs), max(g_accs))
    min_val = min(min(f_accs), min(g_accs))
    plt.plot([min_val, max_val], [min_val, max_val], 
             'k--', alpha=0.5, label='y=x')
    
    plt.colorbar(scatter, label='Memorization Score')
    plt.xlabel('Model f Test Accuracy')
    plt.ylabel('Model g Test Accuracy')
    plt.title('Model Performance Across All Seeds')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Distribution of memorization scores
    plt.subplot(1, 2, 2)
    plt.hist(mem_scores, bins=15, density=True, alpha=0.7)
    plt.axvline(x=np.mean(mem_scores), color='r', linestyle='--', 
                label=f'Mean: {np.mean(mem_scores):.3f}')
    plt.axvline(x=np.mean(mem_scores) + np.std(mem_scores), color='g', linestyle=':', 
                label=f'±1 STD: {np.std(mem_scores):.3f}')
    plt.axvline(x=np.mean(mem_scores) - np.std(mem_scores), color='g', linestyle=':')
    
    plt.xlabel('Memorization Score')
    plt.ylabel('Density')
    plt.title('Distribution of Memorization Scores Across Seeds')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
