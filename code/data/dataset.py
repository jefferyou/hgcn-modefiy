import numpy as np
import json
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import networkx as nx
from networkx.readwrite import json_graph


class GraphDataset:
    """
    Dataset class for loading graph data
    """

    def __init__(self, prefix, data_dir, normalize=True, walk_type=None, dfs_num_len=None, device="cpu"):
        """
        Args:
            prefix: Prefix for data files
            data_dir: Directory containing the data
            normalize: Whether to normalize features
            walk_type: Type of random walks to use ('rand_edges', 'rand_bfs_walks', or 'rand_bfs_dfs_walks')
            dfs_num_len: Parameters for DFS walks [num_walks, walk_length]
            device: Device to load tensors to
        """
        self.prefix = prefix
        self.data_dir = data_dir
        self.device = device

        # Load graph
        self.G, self.features, self.id_map, self.walks, self.class_map = self._load_data(
            normalize, walk_type, dfs_num_len
        )

        # Create adjacency list
        self.adj = self._create_adj_list()

        # Compute node degrees (for negative sampling)
        self.degrees = self._compute_degrees()

        # Extract train/val/test nodes
        self.train_nodes, self.val_nodes, self.test_nodes = self._split_nodes()

    def _load_data(self, normalize, walk_type, dfs_num_len):
        """
        Load graph data from files
        """
        # Load graph from JSON
        G_data = json.load(open(os.path.join(self.data_dir, f"{self.prefix}-G.json")))
        G = json_graph.node_link_graph(G_data)

        # Convert node IDs to integers if needed
        if isinstance(list(G.nodes())[0], int):
            conversion = lambda n: int(n)
        else:
            conversion = lambda n: n

        # Load features if they exist
        if os.path.exists(os.path.join(self.data_dir, f"{self.prefix}-feats.npy")):
            features = np.load(os.path.join(self.data_dir, f"{self.prefix}-feats.npy"))
        else:
            print("No features present.. Only identity features will be used.")
            features = None

        # Load ID map
        id_map = json.load(open(os.path.join(self.data_dir, f"{self.prefix}-id_map.json")))
        id_map = {conversion(k): int(v) for k, v in id_map.items()}

        # Load class map
        class_map = json.load(open(os.path.join(self.data_dir, f"{self.prefix}-class_map.json")))
        if isinstance(list(class_map.values())[0], list):
            lab_conversion = lambda n: n
        else:
            lab_conversion = lambda n: int(n)
        class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}

        # Remove nodes without val/test annotations
        broken_count = 0
        for node in list(G.nodes()):
            if not 'val' in G.nodes[node] or not 'test' in G.nodes[node]:
                G.remove_node(node)
                broken_count += 1

        print(f"Removed {broken_count} nodes that lacked proper annotations")

        # Mark edges as train_removed if they connect to val/test nodes
        for edge in G.edges():
            if (G.nodes[edge[0]]['val'] or G.nodes[edge[1]]['val'] or
                    G.nodes[edge[0]]['test'] or G.nodes[edge[1]]['test']):
                G[edge[0]][edge[1]]['train_removed'] = True
            else:
                G[edge[0]][edge[1]]['train_removed'] = False

        # Normalize features
        if normalize and features is not None:
            train_ids = np.array([id_map[n] for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']])
            train_feats = features[train_ids]
            scaler = StandardScaler()
            scaler.fit(train_feats)
            features = scaler.transform(features)

        # Load random walks
        walks = []

        if walk_type == 'rand_bfs_walks':
            walk_file = f"{self.prefix}-walks.txt"
            with open(os.path.join(self.data_dir, walk_file)) as fp:
                for line in fp:
                    walks.append([conversion(x) for x in line.split()])

        elif walk_type == 'rand_bfs_dfs_walks':
            # Load BFS walks
            bfs_walk_file = f"{self.prefix}-walks.txt"
            with open(os.path.join(self.data_dir, bfs_walk_file)) as fp:
                for line in fp:
                    walks.append([conversion(x) for x in line.split()])

            # Load DFS walks
            dfs_walk_file = f"{self.prefix}-dfs-walks.txt"
            with open(os.path.join(self.data_dir, dfs_walk_file)) as fp:
                for line in fp:
                    walks.append([conversion(x) for x in line.split()])

        return G, features, id_map, walks, class_map

    def _create_adj_list(self):
        """
        Create adjacency list for efficient neighbor sampling
        """
        adj = {}
        for node in self.G.nodes():
            adj[self.id_map[node]] = []
            for neighbor in self.G.neighbors(node):
                # Only include edges that haven't been removed for training
                if not self.G[node][neighbor].get('train_removed', False):
                    adj[self.id_map[node]].append(self.id_map[neighbor])

        return adj

    def _compute_degrees(self):
        """
        Compute node degrees for negative sampling
        """
        degrees = np.zeros(len(self.id_map))

        for node in self.G.nodes():
            if not self.G.nodes[node]['val'] and not self.G.nodes[node]['test']:
                neighbors = [n for n in self.G.neighbors(node)
                             if not self.G[node][n].get('train_removed', False)]
                degrees[self.id_map[node]] = len(neighbors)

        return degrees

    def _split_nodes(self):
        """
        Split nodes into train, validation, and test sets
        """
        train_nodes = [n for n in self.G.nodes()
                       if not self.G.nodes[n]['val'] and not self.G.nodes[n]['test']]
        val_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['val']]
        test_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['test']]

        # Convert to IDs
        train_ids = [self.id_map[n] for n in train_nodes]
        val_ids = [self.id_map[n] for n in val_nodes]
        test_ids = [self.id_map[n] for n in test_nodes]

        return train_ids, val_ids, test_ids

    def get_features(self):
        """
        Get node features as torch tensor
        """
        if self.features is None:
            return None

        return torch.FloatTensor(self.features).to(self.device)

    def get_adj_list(self):
        """
        Get adjacency list
        """
        return self.adj

    def get_degrees(self):
        """
        Get node degrees
        """
        return self.degrees

    def get_labels(self, nodes=None):
        """
        Get node labels

        Args:
            nodes: Specific nodes to get labels for (if None, get all labels)

        Returns:
            Labels tensor
        """
        if nodes is None:
            nodes = list(self.G.nodes())

        labels = np.array([self.class_map[n] for n in nodes])

        # Check if multi-label or single-label
        if isinstance(self.class_map[nodes[0]], list):
            return torch.FloatTensor(labels).to(self.device)
        else:
            return torch.LongTensor(labels).to(self.device)

    def get_walks(self):
        """
        Get random walks
        """
        return self.walks