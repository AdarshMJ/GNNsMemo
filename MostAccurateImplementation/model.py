import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GraphConv as PyGGraphConv

class NodeGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(num_features, hidden_channels))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Last layer for node-level prediction
        self.convs.append(GCNConv(hidden_channels, num_classes))
        
        # Explicitly reset parameters during initialization
        self.reset_parameters()

    def forward(self, x, edge_index, return_node_emb=False):
        # Process through all layers except last
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.0, training=self.training)
        
        # Store embeddings from the last hidden layer
        node_emb = x
        
        # Final classification layer
        x = self.convs[-1](x, edge_index)
        
        if return_node_emb:
            return x, node_emb
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

class NodeGAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, num_layers, heads=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        
        # First layer (multiply hidden_channels by heads since output will have heads)
        self.convs.append(GATv2Conv(num_features, hidden_channels, heads=heads))
        
        # Middle layers - input dim is hidden_channels * heads from previous layer
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads))
        
        # Last layer with 1 head for final prediction
        self.convs.append(GATv2Conv(hidden_channels * heads, num_classes, heads=1))
        
        self.reset_parameters()

    def forward(self, x, edge_index, return_node_emb=False):
        # Process through all layers except last
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)  # ELU activation typically works better with GAT
            x = F.dropout(x, p=0.0, training=self.training)
        
        # Store embeddings from the last hidden layer
        node_emb = x  # This will have shape [num_nodes, hidden_channels * heads]
        
        # Final prediction layer (outputs [num_nodes, num_classes])
        x = self.convs[-1](x, edge_index)
        
        if return_node_emb:
            return x, node_emb
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

class NodeGraphConv(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, num_layers, aggr='add'):
        super().__init__()
        self.num_layers = num_layers
        
        self.layers = torch.nn.ModuleList()
        
        # First layer
        self.layers.append(PyGGraphConv(num_features, hidden_channels, aggr=aggr))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(PyGGraphConv(hidden_channels, hidden_channels, aggr=aggr))
        
        # Last layer for node-level prediction
        self.layers.append(PyGGraphConv(hidden_channels, num_classes, aggr=aggr))
        
        self.reset_parameters()

    def forward(self, x, edge_index, return_node_emb=False):
        # Process through all layers except last
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.0, training=self.training)
        
        # Store embeddings from the last hidden layer
        node_emb = x
        
        # Final prediction layer
        x = self.layers[-1](x, edge_index)
        
        if return_node_emb:
            return x, node_emb
        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()