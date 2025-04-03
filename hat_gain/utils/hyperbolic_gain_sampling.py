import numpy as np
import random
import tensorflow as tf
import networkx as nx

class GraphSampler:
    """
    Enhanced graph sampler implementing both local and global neighborhood sampling
    for the Hyperbolic GAIN model.
    """
    
    def __init__(self, G, walk_params=None):
        """
        Initialize the graph sampler.
        
        Args:
            G: NetworkX graph
            walk_params: Dictionary containing sampling parameters
                - bfs_num: Number of BFS walks per node
                - bfs_len: Length of BFS walks
                - dfs_num: Number of DFS walks per node
                - dfs_len: Length of DFS walks
        """
        self.G = G
        
        # Default parameters if not provided
        if walk_params is None:
            self.walk_params = {
                'bfs_num': 10,  # Number of BFS walks per node
                'bfs_len': 2,   # Length of BFS walks
                'dfs_num': 10,  # Number of DFS walks per node
                'dfs_len': 3    # Length of DFS walks
            }
        else:
            self.walk_params = walk_params
            
        # Prepare the subgraph for sampling (only training nodes)
        self.prepare_training_subgraph()
        
    def prepare_training_subgraph(self):
        """Prepare a subgraph containing only training nodes for sampling"""
        train_nodes = [n for n in self.G.nodes() if not self.G.nodes[n].get('val', False) 
                                                and not self.G.nodes[n].get('test', False)]
        self.train_G = self.G.subgraph(train_nodes)
        
    def run_bfs_walks(self, nodes=None, walk_len=None, num_walks=None):
        """
        Run BFS-like random walks to capture local neighborhood information.
        
        Args:
            nodes: List of nodes to start walks from (if None, use all training nodes)
            walk_len: Length of each walk (if None, use default)
            num_walks: Number of walks per node (if None, use default)
            
        Returns:
            List of tuples (start_node, visited_node) representing co-occurrences
        """
        if nodes is None:
            nodes = list(self.train_G.nodes())
        if walk_len is None:
            walk_len = self.walk_params['bfs_len']
        if num_walks is None:
            num_walks = self.walk_params['bfs_num']
        
        pairs = []
        for count, node in enumerate(nodes):
            if self.train_G.degree(node) == 0:
                continue
                
            for i in range(num_walks):
                curr_node = node
                for j in range(walk_len):
                    # Get immediate neighbors - BFS characteristic
                    if self.train_G.degree(curr_node) == 0:
                        break
                    next_node = random.choice(list(self.train_G.neighbors(curr_node)))
                    
                    # Add co-occurrence if not the starting node
                    if curr_node != node:
                        pairs.append((node, curr_node))
                    
                    curr_node = next_node
                    
            if count % 1000 == 0 and count > 0:
                print(f"Completed BFS walks for {count} nodes")
                
        print(f'Generated {len(pairs)} local neighborhood pairs')
        return pairs
    
    def run_dfs_walks(self, nodes=None, dfs_len=None, num_walks=None):
        """
        Run DFS-like random walks to capture global structure information.
        
        Args:
            nodes: List of nodes to start walks from (if None, use all training nodes)
            dfs_len: Length of each walk (if None, use default)
            num_walks: Number of walks per node (if None, use default)
            
        Returns:
            List of tuples (start_node, end_node) representing global connections
        """
        if nodes is None:
            nodes = list(self.train_G.nodes())
        if dfs_len is None:
            dfs_len = self.walk_params['dfs_len']
        if num_walks is None:
            num_walks = self.walk_params['dfs_num']
        
        dfs_pairs = []
        for count, node in enumerate(nodes):
            if self.train_G.degree(node) == 0:
                continue
                
            for i in range(num_walks):
                curr_node = node
                prev_node = None
                next_node = None
                
                # First step - choose a random neighbor
                if self.train_G.degree(curr_node) == 0:
                    continue
                    
                depth_1_neighbors = list(self.train_G.neighbors(curr_node))
                if not depth_1_neighbors:
                    continue
                next_node = random.choice(depth_1_neighbors)
                
                # Subsequent steps - try to move away from already visited areas
                for j in range(dfs_len - 1):
                    if self.train_G.degree(next_node) == 0:
                        break
                        
                    # Find neighbors of next_node that aren't neighbors of curr_node (DFS characteristic)
                    depth_2_neighbors = [neigh for neigh in self.train_G.neighbors(next_node) 
                                        if neigh != curr_node and (prev_node is None or neigh != prev_node)]
                    
                    # If no valid neighbors, pick any neighbor of next_node
                    if not depth_2_neighbors:
                        depth_2_neighbors = list(self.train_G.neighbors(next_node))
                        if not depth_2_neighbors:
                            break
                    
                    prev_node = curr_node
                    curr_node = next_node
                    next_node = random.choice(depth_2_neighbors)
                
                # Add only the final node reached to capture long-range dependencies
                if curr_node != node:
                    dfs_pairs.append((node, curr_node))
                    
            if count % 1000 == 0 and count > 0:
                print(f"Completed DFS walks for {count} nodes")
                
        print(f'Generated {len(dfs_pairs)} global structure pairs')
        return dfs_pairs
    
    def generate_context_pairs(self):
        """
        Generate all context pairs using both BFS and DFS walks.
        
        Returns:
            List of context pairs combining both local and global information
        """
        bfs_pairs = self.run_bfs_walks()
        dfs_pairs = self.run_dfs_walks()
        
        all_pairs = bfs_pairs + dfs_pairs
        print(f'Generated total of {len(all_pairs)} context pairs')
        
        return all_pairs
    
    def get_context_tensor(self, id_map):
        """
        Convert context pairs to tensor format using node ID mappings.
        
        Args:
            id_map: Dictionary mapping node IDs to indices
            
        Returns:
            context_tensor: Tensor of context pairs for model training
        """
        context_pairs = self.generate_context_pairs()
        
        # Convert pairs to tensor format
        context_tensor = []
        for src, dst in context_pairs:
            if src in id_map and dst in id_map:
                context_tensor.append((id_map[src], id_map[dst]))
        
        return context_tensor
