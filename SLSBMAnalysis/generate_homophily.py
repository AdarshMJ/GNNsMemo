import os
import glob
import torch
import pickle
import numpy as np
from torch_geometric.data import Dataset, Data
import logging

class HomophilySBMDataset(Dataset):
    """
    Dataset class to load SBM graphs:
    1. Fixed homophily with varying informativeness (from generate_sbm_pyg.py)
    2. Fixed informativeness with varying homophily (from generate_fixed_info.py)
    """
    
    def __init__(self, root='data/pyg_sbm', homophily=None, informativeness=None, n_graphs=None, transform=None, pre_transform=None):
        """
        Args:
            root: Root directory where datasets are stored
            homophily: Fixed homophily value to use. If None and informativeness is None, uses all available homophily levels
            informativeness: Fixed informativeness value to use. If specified, loads from the fixed_info_sbm directory
            n_graphs: Number of graphs to load (evenly spaced). If None, loads all
            transform: PyG transform to apply
            pre_transform: PyG pre-transform to apply
        """
        self.homophily = homophily
        self.informativeness = informativeness
        self.n_graphs = n_graphs
        self.dataset_root = root
        self.graphs = []
        super(HomophilySBMDataset, self).__init__(root, transform, pre_transform)
        
        # Load the graphs
        self._load_datasets()
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return []
    
    def _load_datasets(self):
        """Load datasets from appropriate folders based on parameters."""
        # Determine the correct directory to load from
        if self.informativeness is not None:
            # Load from fixed informativeness dataset
            base_dir = 'data/fixed_info_sbm'
            target_folder = os.path.join(base_dir, f"info_{self.informativeness:.2f}")
            if not os.path.exists(target_folder):
                raise ValueError(f"No dataset found for informativeness={self.informativeness} at {target_folder}")
            
            # Load all graphs from this folder
            self._load_from_folder(target_folder)
            
            # If homophily is specified, filter the loaded graphs
            if self.homophily is not None:
                self.graphs = [g for g in self.graphs if abs(g.homophily - self.homophily) < 0.01]
                if not self.graphs:
                    raise ValueError(f"No graphs found with homophily={self.homophily} in {target_folder}")
        
        elif self.homophily is not None:
            # Load from fixed homophily dataset
            homophily_folder = os.path.join(self.dataset_root, f"hom_{self.homophily:.2f}")
            if not os.path.exists(homophily_folder):
                raise ValueError(f"No dataset found for homophily={self.homophily} at {homophily_folder}")
            
            # Load all graphs from this folder
            self._load_from_folder(homophily_folder)
        
        else:
            # Load all homophily levels from original dataset
            homophily_folders = sorted(glob.glob(os.path.join(self.dataset_root, "hom_*")))
            if not homophily_folders:
                raise ValueError(f"No homophily folders found in {self.dataset_root}")
            
            # Load from each folder
            for folder in homophily_folders:
                self._load_from_folder(folder)
        
        # Subset graphs if n_graphs is specified
        if self.n_graphs is not None and self.n_graphs < len(self.graphs):
            indices = np.linspace(0, len(self.graphs)-1, self.n_graphs, dtype=int)
            self.graphs = [self.graphs[i] for i in indices]
        
        # Check that we have graph data
        if not self.graphs:
            raise ValueError(f"No graphs loaded from {self.dataset_root}")
        
        # Rename informativeness attribute for compatibility with existing code
        for graph in self.graphs:
            if hasattr(graph, 'informativeness') and not hasattr(graph, 'label_informativeness'):
                graph.label_informativeness = graph.informativeness
    
    def _load_from_folder(self, folder_path):
        """Load all graphs from the specified folder."""
        # Check if pickle list file exists
        pkl_path = os.path.join(folder_path, "sbm_list.pkl")
        if os.path.exists(pkl_path):
            # Load all graphs from pickle
            with open(pkl_path, "rb") as f:
                folder_graphs = pickle.load(f)
            self.graphs.extend(folder_graphs)
        else:
            # Load individual PT files
            graph_files = sorted(glob.glob(os.path.join(folder_path, "sbm_*.pt")))
            for file_path in graph_files:
                data = torch.load(file_path)
                self.graphs.append(data)
    
    def len(self):
        return len(self.graphs)
    
    def get(self, idx):
        """Get a graph by index."""
        return self.graphs[idx]
    
    def process(self):
        """Not needed as we're loading pre-processed graphs."""
        pass
    
    def download(self):
        """Not needed as we're loading locally generated graphs."""
        pass