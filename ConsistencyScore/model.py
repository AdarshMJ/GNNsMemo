import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATv2Conv,GraphConv as PyGGraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import SimpleConv
import math
import itertools

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
            
        # Last layer
        self.convs.append(GATv2Conv(dim_h * heads, dim_out, heads=heads))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(x,p=0.0, training=self.training)
            x = conv(x, edge_index)
            x = F.elu(x)
        
        x = F.dropout(x,p=0.0, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

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
        self.convs.append(GCNConv(hidden_channels, num_classes))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.0, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

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
        self.num_layers = num_layers
        
        self.layers = torch.nn.ModuleList()
        self.layers.append(PyGGraphConv(num_features, hidden_channels, aggr=aggr))
        
        for _ in range(num_layers - 2):
            self.layers.append(PyGGraphConv(hidden_channels, hidden_channels, aggr=aggr))
        
        self.layers.append(PyGGraphConv(hidden_channels, hidden_channels, aggr=aggr))
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch=None):
        # Process through layers
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.0, training=self.training)
        
        x = self.layers[-1](x, edge_index)
        x = x.relu()
        
        # Skip global pooling for node classification
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Final classification
        out = self.lin(x)
        return out

    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()
        self.lin.reset_parameters()

class AsymGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, num_layers):
        super().__init__()
        torch.manual_seed(12345)
        
        # Initialize C matrices for asymmetric components
        self.C_lst = [
            .01 * torch.randn(hidden_channels, hidden_channels) / math.sqrt(hidden_channels) 
            for _ in range(num_layers)
        ]
        
        # Create sparse linear builder
        lin_builder = lambda in_dim, out_dim: SparseLinear(
            in_dim, 
            out_dim, 
            mask_constant=0.5, 
            num_fixed=6, 
            do_normal_mask=True, 
            mask_type='random_subsets'
        )
        
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(AsymConv(num_features, hidden_channels, C=self.C_lst[0], lin_builder=lin_builder))
        
        # Middle layers
        for i in range(num_layers - 2):
            self.convs.append(
                AsymConv(hidden_channels, hidden_channels, C=self.C_lst[i+1], lin_builder=lin_builder)
            )
            
        # Last layer
        self.convs.append(AsymConv(hidden_channels, hidden_channels, C=self.C_lst[-1], lin_builder=lin_builder))
        
        self.lin = lin_builder(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # Process through conv layers
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.dropout(x, p=0.0, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        # Remove global pooling since this is node classification
        # No need for batch parameter
        
        # Final classification
        x = self.lin(x)
        return x.log_softmax(dim=-1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()
        
    def count_unused_params(self):
        count = 0
        for node in self.modules():
            if hasattr(node, 'mask'):
                count += (1-node.mask).sum().int().item()
        return count

class AsymConv(nn.Module):
    def __init__(self, in_dim, out_dim, C=None, lin_builder=None):
        super().__init__()
        # Add these attributes to match GNN convention
        self.in_channels = in_dim
        self.out_channels = out_dim
        
        self.conv = SimpleConv(aggr='mean', combine_root='sum')
        self.mlp = nn.Sequential(
            lin_builder(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            AsymSwiGLU(C)
        )
    
    def reset_parameters(self):
        self.mlp[0].reset_parameters()
        self.mlp[1].reset_parameters()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.mlp(x)
        return x

# Add the necessary classes from asym_nets.py
class AsymSwiGLU(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.register_buffer('C', C)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        gate = self.sigmoid(torch.matmul(x, self.C))
        x = gate * x
        return x

class SparseLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, mask_type='densest', mask_constant=1, mask_num=0, num_fixed=6, do_normal_mask=True, mask_path=None):
        super().__init__()
        assert out_dim < 2**in_dim, 'out dim cannot be much higher than in dim'
        
        if mask_path is not None:
            mask, _ = torch.load(mask_path)
        else:
            mask = make_mask(in_dim, out_dim, mask_type=mask_type, num_fixed=num_fixed, mask_num=mask_num)

        self.register_buffer('mask', mask, persistent=True)
        self.weight = nn.Parameter(torch.empty((out_dim, in_dim)))

        if mask_path is not None:
            _, n_mask = torch.load(mask_path)
            self.register_buffer('normal_mask', n_mask, persistent=True)
        else:
            if do_normal_mask:
                self.register_buffer('normal_mask', normal_mask(out_dim, in_dim, mask_num), persistent=True)
            else:
                self.register_buffer('normal_mask', torch.ones(size=(out_dim, in_dim)), persistent=True)

        hook = self.weight.register_hook(lambda grad: self.mask*grad)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.register_parameter('bias', None)

        self.mask_constant = mask_constant
        self.mask_num = mask_num
        self.num_fixed = num_fixed
        self.reset_parameters()

    def forward(self, x):
        self.weight.data = (self.weight.data * self.mask + (1-self.mask)*self.mask_constant*self.normal_mask)
        return F.linear(x, self.weight, self.bias)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = (self.weight.data * self.mask + (1-self.mask)*self.mask_constant*self.normal_mask)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

# Add helper functions from asym_nets.py
def get_subset(num_cols, row_idx, num_sample, mask_num):
    g = torch.Generator()
    g.manual_seed(row_idx + abs(hash(str(mask_num) + str(seed))))
    indices = torch.arange(num_cols)
    return (indices[torch.randperm(num_cols, generator = g)[:num_sample]])

def normal_mask(out_dim, in_dim, mask_num):
    g = torch.Generator()
    g.manual_seed(abs(hash(str(mask_num)+ str(seed))))
    return torch.randn(size=(out_dim,in_dim), generator = g)

def make_mask(in_dim, out_dim, mask_num = 0, num_fixed = 6, mask_type='densest'):
    # out_dim x in_dim matrix
    # where each row is unique
    assert out_dim < 2**(in_dim)
    assert in_dim > 0 and out_dim > 0

    if mask_type == 'densest':
        mask = torch.ones(out_dim, in_dim)
        mask[0, :] = 1 # first row is dense
        row_idx = 1
        if out_dim == 1:
            return mask

        for nz in range(1, in_dim):
            for zeros_in_row in itertools.combinations(range(in_dim), nz):
                mask[row_idx, zeros_in_row] = 0
                row_idx += 1
                if row_idx >= out_dim:
                    return mask
    elif mask_type == 'bound_zeros':
        # other type of mask based on lower bounding sparsity to break symmetries more
        mask = torch.ones(out_dim, in_dim)
        least_zeros = 2
        row_idx = 0
        for nz in range(least_zeros, in_dim):
            for zeros_in_row in itertools.combinations(range(in_dim), nz):
                mask[row_idx, zeros_in_row] = 0
                row_idx += 1
                if row_idx >= out_dim:
                    return mask

        raise ValueError('Error in making mask, possibly because out_dim is too large for these settings')

    elif mask_type == 'random_subsets':
            # other type of mask based on lower bounding sparsity to break symmetries more
            mask = torch.ones(out_dim, in_dim)
            row_idx = 0
            least_zeros = num_fixed
            for nz in range(least_zeros, in_dim):
                while True:

                    zeros_in_row = get_subset(in_dim, row_idx, least_zeros, mask_num)
                    mask[row_idx, zeros_in_row] = 0
                    row_idx += 1
                    if row_idx >= out_dim:
                        return mask

            raise ValueError('Error in making mask, possibly because out_dim is too large for these settings')
    else:
        raise ValueError('Invalid mask type')
