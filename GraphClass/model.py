import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GraphConv as PyGGraphConv
from torch_geometric.nn import global_mean_pool

seed = 12345

class GATv2(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, num_layers, heads=4):
        super().__init__()
        torch.manual_seed(seed)
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GATv2Conv(dim_in, dim_h, heads=heads))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(dim_h * heads, dim_h, heads=heads))
        
        # Last layer for graph-level prediction
        self.convs.append(GATv2Conv(dim_h * heads, dim_h, heads=1))
        
        # Final classification layer
        self.lin = nn.Linear(dim_h, dim_out)

    def forward(self, x, edge_index, batch=None, return_emb=False):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=0.0, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        # Global mean pooling (handle None batch)
        if batch is None:
            # If batch is None, treat as single graph
            emb = torch.mean(x, dim=0, keepdim=True)
        else:
            emb = global_mean_pool(x, batch)
        
        # Final classification
        out = self.lin(emb)
        
        if return_emb:
            return out, emb
        return out

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

class SimpleGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, num_layers):
        super().__init__()
        torch.manual_seed(seed)
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(num_features, hidden_channels))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Last layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Final classification layer
        self.lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch=None, return_emb=False):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.0, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        # Global mean pooling (handle None batch)
        if batch is None:
            # If batch is None, treat as single graph
            emb = torch.mean(x, dim=0, keepdim=True)
        else:
            emb = global_mean_pool(x, batch)
        
        # Final classification
        out = self.lin(emb)
        
        if return_emb:
            return out, emb
        return out

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

class SimpleGCNRes(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, num_layers):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(num_features, hidden_channels))
        
        # Input projection if dimensions don't match
        self.input_proj = None
        if num_features != hidden_channels:
            self.input_proj = nn.Linear(num_features, hidden_channels)
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            
        # Last layer
        self.convs.append(GCNConv(hidden_channels, num_classes))

    def forward(self, x, edge_index):
        # Handle first layer separately for residual connection
        identity = x
        x = self.convs[0](x, edge_index)
        x = x.relu()
        
        # Project input if dimensions don't match
        if self.input_proj is not None:
            identity = self.input_proj(identity)
        
        x = x + identity  # First residual connection
        x = F.dropout(x, p=0.0, training=self.training)
        
        # Middle layers with residual connections
        for conv in self.convs[1:-1]:
            identity = x
            x = conv(x, edge_index)
            x = x.relu()
            x = x + identity  # Residual connection
            x = F.dropout(x, p=0.0, training=self.training)
        
        # Final layer without residual connection
        x = self.convs[-1](x, edge_index)
        return x
    

class GraphConv(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, num_layers, aggr='add'):
        super().__init__()
        torch.manual_seed(seed)
        self.num_layers = num_layers  # Store num_layers as instance variable
        
        self.layers = torch.nn.ModuleList()
        
        # First layer
        self.layers.append(PyGGraphConv(num_features, hidden_channels, aggr=aggr))
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.layers.append(PyGGraphConv(hidden_channels, hidden_channels, aggr=aggr))
            
        # Last layer
        self.layers.append(PyGGraphConv(hidden_channels, hidden_channels, aggr=aggr))
        
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def get_layer_representations(self, x, edge_index, batch):
        """Helper method to get representations at each layer"""
        representations = []
        
        # Get representations from each layer
        for layer in self.layers:
            x = layer(x, edge_index)
            x = x.relu()
            # Pool to get graph-level representations
            graph_x = global_mean_pool(x, batch)
            representations.append(graph_x)
            
        return representations

    def forward(self, x, edge_index, batch=None, return_layer_rep=False, return_emb=False):
        if return_layer_rep:
            layer_representations = self.get_layer_representations(x, edge_index, batch)
        
        # Normal forward pass
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.0, training=self.training)
        
        x = self.layers[-1](x, edge_index)
        x = x.relu()
        
        # Global mean pooling (handle None batch)
        if batch is None:
            # If batch is None, treat as single graph
            emb = torch.mean(x, dim=0, keepdim=True)
        else:
            emb = global_mean_pool(x, batch)
        
        # Final classifier
        out = self.lin(emb)
        
        if return_layer_rep:
            return out, layer_representations
        if return_emb:
            return out, emb
        return out

    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()
        self.lin.reset_parameters()
