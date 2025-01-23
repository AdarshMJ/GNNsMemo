import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import torch.nn.functional as F
import random
from tqdm import tqdm
from torch_geometric.nn import GCNConv

class GraphAugmentor:
    def __init__(self, edge_perturb_ratio=0.1, num_augmentations=5, augment_type='remove'):
        """
        Initialize augmentor with specified strategy
        Args:
            edge_perturb_ratio: ratio of edges to modify
            num_augmentations: number of augmented versions to create
            augment_type: 'remove' or 'add' - whether to remove existing edges or add new ones
        """
        self.edge_perturb_ratio = edge_perturb_ratio
        self.num_augmentations = num_augmentations
        assert augment_type in ['remove', 'add'], "augment_type must be 'remove' or 'add'"
        self.augment_type = augment_type
        
    def generate_edge_augmentations(self, edge_index):
        """Generate multiple augmented edge_index tensors"""
        augmented_edges = []
        num_edges = edge_index.shape[1]
        device = edge_index.device
        num_edges_to_perturb = int(num_edges * self.edge_perturb_ratio)
        num_nodes = edge_index.max() + 1
        
        for _ in range(self.num_augmentations):
            if self.augment_type == 'remove':
                # Only remove edges
                edge_mask = torch.ones(num_edges, dtype=torch.bool, device=device)
                indices_to_remove = torch.randperm(num_edges, device=device)[:num_edges_to_perturb]
                edge_mask[indices_to_remove] = False
                aug_edge_index = edge_index[:, edge_mask]
                
            else:  # add
                # Only add new edges
                new_edges = torch.randint(0, num_nodes, (2, num_edges_to_perturb), device=device)
                aug_edge_index = torch.cat([edge_index, new_edges], dim=1)
            
            augmented_edges.append(aug_edge_index)
            
        return augmented_edges

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.0, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def get_embedding(self, x, edge_index):
        """Get node embeddings from the last hidden layer"""
        x = self.conv1(x, edge_index).relu()
        return x

class MemorizationScorer:
    def __init__(self, model_type='encoder', model_class=None, in_channels=None, 
                 hidden_channels=None, out_channels=None, num_classes=None, device=None, num_epochs=200, lr=0.01):
        """
        Args:
            model_type: 'encoder' or 'gcn' - determines which training approach to use
            model_class: for encoder mode, the encoder class to use
            in_channels: input feature dimensions
            hidden_channels: hidden layer dimensions (for GCN)
            out_channels: output dimensions (embedding dim for encoder)
            num_classes: number of classes (for GCN)
            device: torch device
            num_epochs: number of training epochs
            lr: learning rate
        """
        self.model_type = model_type
        self.model_class = model_class if model_type == 'encoder' else GCN
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels if model_type == 'encoder' else num_classes
        self.num_classes = num_classes
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr

    def train_model(self, model, data, train_mask):
        """Train either encoder or GCN and return test accuracy for GCN"""
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,weight_decay=5e-4)
        best_test_acc = 0.0
        
        #pbar = tqdm(range(self.num_epochs), desc='Training', leave=False)
        for epoch in range(self.num_epochs):
            model.train()
            optimizer.zero_grad()
            
            if self.model_type == 'encoder':
                # Encoder training with reconstruction loss
                z = model.encode(data.x, data.edge_index)
                z_train = z[train_mask]
                loss = F.mse_loss(z_train @ z_train.t(), 
                                torch.eye(z_train.size(0), device=self.device))
            else:
                # GCN training with cross entropy loss
                out = model(data.x, data.edge_index)
                loss = F.cross_entropy(out[train_mask], data.y[train_mask])
                
                # Compute test accuracy
                if hasattr(data, 'test_mask'):
                    model.eval()
                    with torch.no_grad():
                        pred = out[data.test_mask].max(1)[1]
                        test_acc = pred.eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
                        best_test_acc = max(best_test_acc, test_acc)
            
            loss.backward()
            optimizer.step()
            #pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return model, best_test_acc if self.model_type == 'gcn' else (model, None)

    def get_embeddings(self, model, x, edge_index):
        """Get embeddings from either model type"""
        model.eval()
        with torch.no_grad():
            if self.model_type == 'encoder':
                return model.encode(x, edge_index)
            else:
                return model.get_embedding(x, edge_index)

    def compute_alignment(self, model, data, x1_edge_index, x2_edge_index, train_mask):
        """Compute L2 distance between representations"""
        z1 = self.get_embeddings(model, data.x, x1_edge_index)
        z2 = self.get_embeddings(model, data.x, x2_edge_index)
        
        # Only compute alignment for training nodes
        z1_train = z1[train_mask]
        z2_train = z2[train_mask]
        
        return torch.norm(z1_train - z2_train, p=2, dim=1).mean()

    def compute_alignment_metric(self, model, data, augmented_edges, train_mask):
        """Compute average alignment (L2 distance) across all pairs of augmentations"""
        alignment_scores = []
        
        # Compare all pairs of augmentations
        for i in range(len(augmented_edges)):
            for j in range(i+1, len(augmented_edges)):
                x1, x2 = augmented_edges[i], augmented_edges[j]
                alignment = self.compute_alignment(model, data, x1, x2, train_mask)
                alignment_scores.append(alignment)
                    
        return torch.mean(torch.stack(alignment_scores))

    def remove_node_influence(self, data, node_idx):
        """Remove a node's influence from the graph while maintaining structure"""
        # Create a mask for edges that don't involve the target node
        edge_mask = (data.edge_index[0] != node_idx) & (data.edge_index[1] != node_idx)
        new_data = copy.deepcopy(data)
        new_data.edge_index = data.edge_index[:, edge_mask]
        return new_data

    def compute_scores(self, original_data, target_nodes):
        """Compute memorization scores using alignment metric"""
        results = []
        
        for node_idx in target_nodes:
            print(f"Processing node {node_idx}")
            
            # Create two datasets: one with target node, one without
            data_with = copy.deepcopy(original_data)
            data_without = self.remove_node_influence(original_data, node_idx)
            
            # Initialize and train encoders
            encoder_f = self.model_class(self.in_channels, self.out_channels).to(self.device)
            encoder_g = self.model_class(self.in_channels, self.out_channels).to(self.device)
            
            model_f = self.train_model(encoder_f, data_with)
            model_g = self.train_model(encoder_g, data_without)
            
            # Generate augmentations
            augmentor = GraphAugmentor()
            aug_edges = augmentor.generate_edge_augmentations(original_data.edge_index)
            
            # Compute alignment metrics
            f_alignment = self.compute_alignment_metric(model_f, original_data, aug_edges)
            g_alignment = self.compute_alignment_metric(model_g, original_data, aug_edges)
            
            # Compute memorization score
            mem_score = float(g_alignment - f_alignment)
            
            results.append({
                'node_idx': node_idx,
                'f_alignment': float(f_alignment),
                'g_alignment': float(g_alignment),
                'memorization': mem_score
            })
            
        return results

    def compute_scores_with_splits(self, original_data, node_idx, f_train_nodes, g_train_nodes):
        """Compute memorization scores with normalized values between -1 and 1"""
        # Create masks
        f_mask = torch.zeros(original_data.num_nodes, dtype=torch.bool, device=self.device)
        g_mask = torch.zeros(original_data.num_nodes, dtype=torch.bool, device=self.device)
        f_mask[f_train_nodes] = True
        g_mask[g_train_nodes] = True
        
        # Create edge masks
        f_edge_mask = f_mask[original_data.edge_index[0]] & f_mask[original_data.edge_index[1]]
        g_edge_mask = g_mask[original_data.edge_index[0]] & g_mask[original_data.edge_index[1]]
        
        # Create datasets
        data_f = copy.deepcopy(original_data)
        data_g = copy.deepcopy(original_data)
        data_f.edge_index = original_data.edge_index[:, f_edge_mask]
        data_g.edge_index = original_data.edge_index[:, g_edge_mask]
        
        # Initialize models with correct output dimensions
        if self.model_type == 'encoder':
            model_f = self.model_class(self.in_channels, self.out_channels).to(self.device)
            model_g = self.model_class(self.in_channels, self.out_channels).to(self.device)
        else:  # GCN
            model_f = self.model_class(self.in_channels, self.hidden_channels, self.num_classes).to(self.device)
            model_g = self.model_class(self.in_channels, self.hidden_channels, self.num_classes).to(self.device)
        
        # Train models and get test accuracies
        model_f, f_test_acc = self.train_model(model_f, data_f, f_mask)
        model_g, g_test_acc = self.train_model(model_g, data_g, g_mask)
        
        # Generate augmentations and compute alignment
        augmentor = GraphAugmentor(num_augmentations=5)
        aug_edges = augmentor.generate_edge_augmentations(original_data.edge_index)
        
        f_alignment = self.compute_alignment_metric(model_f, original_data, aug_edges, f_mask)
        g_alignment = self.compute_alignment_metric(model_g, original_data, aug_edges, g_mask)
        
        # Compute memorization score and normalize to [-1, 1]
        raw_mem_score = float(g_alignment - f_alignment)
        max_alignment = max(f_alignment, g_alignment)
        min_alignment = min(f_alignment, g_alignment)
        alignment_range = max_alignment - min_alignment
        
        if alignment_range > 0:
            # Normalize to [-1, 1] where:
            # -1: strongest memorization in g (g_alignment is much larger)
            # +1: strongest memorization in f (f_alignment is much larger)
            # 0: no memorization (alignments are equal)
            mem_score = raw_mem_score / alignment_range
        else:
            # If alignments are identical, no memorization
            mem_score = 0.0
        
        return {
            'node_idx': node_idx,
            'f_alignment': float(f_alignment),
            'g_alignment': float(g_alignment),
            'memorization': mem_score,
            'f_test_acc': f_test_acc if f_test_acc is not None else float('nan'),
            'g_test_acc': g_test_acc if g_test_acc is not None else float('nan')
        }

def plot_alignment_memorization(results, save_path='alignment_memorization.png'):
    """Create scatter plot of alignment losses colored by memorization score"""
    # Convert lists to tensors and move to CPU
    f_alignments = torch.tensor([r['f_alignment'] for r in results]).cpu()
    g_alignments = torch.tensor([r['g_alignment'] for r in results]).cpu()
    mem_scores = torch.tensor([r['memorization'] for r in results]).cpu()
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(f_alignments, g_alignments, c=mem_scores, 
                         cmap='viridis', alpha=0.6)
    
    # Add y=x line
    min_val = min(f_alignments.min(), g_alignments.min())
    max_val = max(f_alignments.max(), g_alignments.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', label='y=x')
    
    plt.colorbar(scatter, label='Memorization Score')
    plt.xlabel('Alignment Loss on f')
    plt.ylabel('Alignment Loss on g')
    plt.title('Model Alignment Loss vs. Memorization')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
