import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class EdgeMinibatchLoader:
    """
    Minibatch loader for edge sampling in unsupervised learning
    """
    def __init__(self, adj_list, context_pairs=None, batch_size=100, max_degree=25):
        """
        Args:
            adj_list: Adjacency list for the graph
            context_pairs: Optional list of node pairs from random walks
            batch_size: Number of edges per batch
            max_degree: Maximum node degree to consider
        """
        self.adj_list = adj_list
        self.batch_size = batch_size
        self.max_degree = max_degree
        
        if context_pairs is None:
            # Use direct edges from the graph
            self.edges = []
            for node, neighbors in adj_list.items():
                for neighbor in neighbors:
                    self.edges.append((node, neighbor))
        else:
            # Use random walk co-occurrences
            self.edges = context_pairs
            
        # Shuffle edges
        np.random.shuffle(self.edges)
        
        # Current batch index
        self.batch_idx = 0
        self.num_batches = len(self.edges) // batch_size + 1
        
    def next_batch(self):
        """
        Get the next batch of edges
        
        Returns:
            Tuple of (batch_node1, batch_node2)
        """
        if self.batch_idx * self.batch_size >= len(self.edges):
            self.batch_idx = 0
            np.random.shuffle(self.edges)
            
        start_idx = self.batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.edges))
        batch_edges = self.edges[start_idx:end_idx]
        
        self.batch_idx += 1
        
        # Extract source and target nodes
        batch_node1 = [edge[0] for edge in batch_edges]
        batch_node2 = [edge[1] for edge in batch_edges]
        
        return torch.LongTensor(batch_node1), torch.LongTensor(batch_node2)
    
    def shuffle(self):
        """
        Shuffle the training edges and reset batch index
        """
        np.random.shuffle(self.edges)
        self.batch_idx = 0
        
    def get_validation_edges(self, val_size=None):
        """
        Get a validation set of edges
        
        Args:
            val_size: Size of validation set (if None, use all edges)
            
        Returns:
            Tuple of (val_node1, val_node2)
        """
        # Use the first val_size edges as validation
        if val_size is None:
            val_size = len(self.edges)
            
        val_edges = self.edges[:val_size]
        
        val_node1 = [edge[0] for edge in val_edges]
        val_node2 = [edge[1] for edge in val_edges]
        
        return torch.LongTensor(val_node1), torch.LongTensor(val_node2)
        

class NodeMinibatchLoader:
    """
    Minibatch loader for node sampling in supervised learning
    """
    def __init__(self, adj_list, class_map, train_nodes, batch_size=100, max_degree=25):
        """
        Args:
            adj_list: Adjacency list for the graph
            class_map: Dictionary mapping node IDs to class labels
            train_nodes: List of training node IDs
            batch_size: Number of nodes per batch
            max_degree: Maximum node degree to consider
        """
        self.adj_list = adj_list
        self.class_map = class_map
        self.train_nodes = list(train_nodes)  # Copy to avoid modifying original
        self.batch_size = batch_size
        self.max_degree = max_degree
        
        # Check if multi-label classification
        if isinstance(list(class_map.values())[0], list):
            self.num_classes = len(list(class_map.values())[0])
            self.multi_label = True
        else:
            self.num_classes = len(set(class_map.values()))
            self.multi_label = False
        
        # Current batch index
        self.batch_idx = 0
        self.num_batches = len(self.train_nodes) // batch_size + 1
        
    def next_batch(self):
        """
        Get the next batch of nodes
        
        Returns:
            Tuple of (batch_nodes, batch_labels)
        """
        if self.batch_idx * self.batch_size >= len(self.train_nodes):
            self.batch_idx = 0
            np.random.shuffle(self.train_nodes)
            
        start_idx = self.batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx:end_idx]
        
        self.batch_idx += 1
        
        # Get labels for batch nodes
        if self.multi_label:
            # Multi-label classification
            batch_labels = np.array([self.class_map[n] for n in batch_nodes])
            batch_labels = torch.FloatTensor(batch_labels)
        else:
            # Multi-class classification
            batch_labels = np.array([self.class_map[n] for n in batch_nodes])
            batch_labels = torch.LongTensor(batch_labels)
        
        return torch.LongTensor(batch_nodes), batch_labels
    
    def shuffle(self):
        """
        Shuffle the training nodes and reset batch index
        """
        np.random.shuffle(self.train_nodes)
        self.batch_idx = 0
        
    def get_validation_data(self, val_nodes):
        """
        Get validation data
        
        Args:
            val_nodes: List of validation node IDs
            
        Returns:
            Tuple of (val_nodes, val_labels)
        """
        # Get labels for validation nodes
        if self.multi_label:
            # Multi-label classification
            val_labels = np.array([self.class_map[n] for n in val_nodes])
            val_labels = torch.FloatTensor(val_labels)
        else:
            # Multi-class classification
            val_labels = np.array([self.class_map[n] for n in val_nodes])
            val_labels = torch.LongTensor(val_labels)
        
        return torch.LongTensor(val_nodes), val_labels
