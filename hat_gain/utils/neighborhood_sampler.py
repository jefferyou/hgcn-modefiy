import numpy as np
import random
import networkx as nx

class NeighborhoodSampler:
    """
    Neighborhood sampler implementing both local (BFS) and global (DFS) sampling strategies.
    Local neighborhood uses BFS (Breadth-First Search) to capture nearby nodes.
    Global neighborhood uses DFS (Depth-First Search) to capture distant relationships.
    """
    
    def __init__(self, graph, walk_seed=42):
        """
        Initialize the neighborhood sampler.
        
        Args:
            graph: NetworkX graph
            walk_seed: Random seed for reproducibility
        """
        self.graph = graph
        self.random_seed = walk_seed
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
    
    def sample_local_neighborhood(self, node, walk_len=5, num_walks=10):
        """
        Sample local neighborhood using BFS-like random walks.
        
        Args:
            node: Starting node
            walk_len: Length of each walk
            num_walks: Number of walks to perform
            
        Returns:
            List of node pairs (co-occurrences)
        """
        pairs = []
        
        # Skip isolated nodes
        if self.graph.degree(node) == 0:
            return pairs
        
        # Perform multiple random walks
        for _ in range(num_walks):
            curr_node = node
            for _ in range(walk_len):
                # Sample a random neighbor
                if len(list(self.graph.neighbors(curr_node))) == 0:
                    break
                    
                next_node = random.choice(list(self.graph.neighbors(curr_node)))
                
                # Add co-occurrence if not the starting node
                if curr_node != node:
                    pairs.append((node, curr_node))
                
                curr_node = next_node
        
        return pairs
    
    def sample_global_neighborhood(self, node, dfs_len=10, num_walks=10):
        """
        Sample global neighborhood using DFS-like walks.
        
        Args:
            node: Starting node
            dfs_len: Length of each DFS walk
            num_walks: Number of walks to perform
            
        Returns:
            List of node pairs (co-occurrences)
        """
        pairs = []
        
        # Skip isolated nodes
        if self.graph.degree(node) == 0:
            return pairs
            
        for _ in range(num_walks):
            curr_node = node
            prev_node = None
            next_node = None
            
            for j in range(dfs_len):
                # First step: choose random neighbor
                if prev_node is None:
                    neighbors = list(self.graph.neighbors(curr_node))
                    if not neighbors:
                        break
                    next_node = random.choice(neighbors)
                    
                # Subsequent steps: choose node farther from start (DFS-like behavior)
                else:
                    # Find neighbors of next_node that are NOT neighbors of curr_node
                    next_neighbors = list(self.graph.neighbors(next_node))
                    distant_neighbors = [neigh for neigh in next_neighbors if neigh not in 
                                        self.graph.neighbors(curr_node)]
                    
                    # If no distant neighbors, choose any neighbor of next_node
                    if not distant_neighbors:
                        if not next_neighbors:
                            break
                        next_neighbors = [n for n in next_neighbors if n != curr_node]
                        if not next_neighbors:
                            break
                        next_node = random.choice(next_neighbors)
                    else:
                        next_node = random.choice(distant_neighbors)
                
                prev_node = curr_node
                curr_node = next_node
                
                # If we've moved away from starting node, create a co-occurrence
                if curr_node != node:
                    pairs.append((node, curr_node))
        
        return pairs
    
    def generate_all_pairs(self, nodes=None, local_walk_len=5, local_num_walks=10, 
                          global_walk_len=10, global_num_walks=10, 
                          use_local=True, use_global=True):
        """
        Generate all neighborhood pairs from the graph.
        
        Args:
            nodes: Nodes to consider (defaults to all nodes)
            local_walk_len: Length of each local BFS walk
            local_num_walks: Number of local walks per node
            global_walk_len: Length of each global DFS walk
            global_num_walks: Number of global walks per node
            use_local: Whether to use local neighborhood sampling
            use_global: Whether to use global neighborhood sampling
            
        Returns:
            All node pairs (co-occurrences)
        """
        if nodes is None:
            nodes = list(self.graph.nodes())
        
        all_pairs = []
        
        for count, node in enumerate(nodes):
            # Skip nodes with no neighbors
            if self.graph.degree(node) == 0:
                continue
                
            node_pairs = []
            
            # Local neighborhood sampling (BFS-like)
            if use_local:
                local_pairs = self.sample_local_neighborhood(
                    node, walk_len=local_walk_len, num_walks=local_num_walks)
                node_pairs.extend(local_pairs)
                
            # Global neighborhood sampling (DFS-like)
            if use_global:
                global_pairs = self.sample_global_neighborhood(
                    node, dfs_len=global_walk_len, num_walks=global_num_walks)
                node_pairs.extend(global_pairs)
                
            all_pairs.extend(node_pairs)
            
            # Print progress
            if count % 1000 == 0:
                print(f"Generated pairs for {count} nodes")
        
        return all_pairs
    
    def save_pairs_to_file(self, filepath, pairs):
        """
        Save node pairs to a text file.
        
        Args:
            filepath: Path to output file
            pairs: List of node pairs
        """
        with open(filepath, "w") as fp:
            fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
        
        print(f"Saved {len(pairs)} pairs to {filepath}")
    
    def generate_and_save_all_pairs(self, output_dir, prefix, nodes=None, 
                                   local_params={'walk_len': 5, 'num_walks': 10},
                                   global_params={'walk_len': 10, 'num_walks': 10}):
        """
        Generate and save all neighborhood pairs.
        
        Args:
            output_dir: Output directory
            prefix: Filename prefix
            nodes: Nodes to consider (defaults to all nodes)
            local_params: Parameters for local sampling
            global_params: Parameters for global sampling
            
        Returns:
            Dictionary with paths to saved files
        """
        if nodes is None:
            nodes = list(self.graph.nodes())
        
        # Local neighborhood sampling (BFS-like)
        local_pairs = []
        if local_params:
            for count, node in enumerate(nodes):
                if self.graph.degree(node) == 0:
                    continue
                    
                pairs = self.sample_local_neighborhood(
                    node, 
                    walk_len=local_params.get('walk_len', 5),
                    num_walks=local_params.get('num_walks', 10)
                )
                local_pairs.extend(pairs)
                
                if count % 1000 == 0:
                    print(f"Generated local pairs for {count} nodes")
        
        # Global neighborhood sampling (DFS-like)
        global_pairs = []
        if global_params:
            for count, node in enumerate(nodes):
                if self.graph.degree(node) == 0:
                    continue
                    
                pairs = self.sample_global_neighborhood(
                    node, 
                    dfs_len=global_params.get('walk_len', 10),
                    num_walks=global_params.get('num_walks', 10)
                )
                global_pairs.extend(pairs)
                
                if count % 1000 == 0:
                    print(f"Generated global pairs for {count} nodes")
        
        # Save to files
        result = {}
        
        if local_pairs:
            local_file = f"{output_dir}/{prefix}-walks.txt"
            self.save_pairs_to_file(local_file, local_pairs)
            result['local_pairs_file'] = local_file
            
        if global_pairs:
            global_file = f"{output_dir}/{prefix}-dfs-walks.txt"
            self.save_pairs_to_file(global_file, global_pairs)
            result['global_pairs_file'] = global_file
            
        # Combined pairs
        all_pairs = local_pairs + global_pairs
        if all_pairs:
            all_file = f"{output_dir}/{prefix}-all-walks.txt"
            self.save_pairs_to_file(all_file, all_pairs)
            result['all_pairs_file'] = all_file
            
        return result

# Utility functions for neighborhood statistics

def compute_neighborhood_stats(G, neighborhood_sampler, sample_size=100, 
                              local_params={'walk_len': 5, 'num_walks': 10},
                              global_params={'walk_len': 10, 'num_walks': 10}):
    """
    Compute statistics on neighborhood sampling to compare local and global methods.
    
    Args:
        G: Input graph
        neighborhood_sampler: NeighborhoodSampler instance
        sample_size: Number of nodes to sample
        local_params: Parameters for local sampling
        global_params: Parameters for global sampling
        
    Returns:
        Dictionary with statistics
    """
    # Sample nodes
    nodes = list(G.nodes())
    if len(nodes) > sample_size:
        nodes = random.sample(nodes, sample_size)
    
    stats = {
        'avg_local_distance': 0,
        'avg_global_distance': 0,
        'avg_local_unique_nodes': 0,
        'avg_global_unique_nodes': 0,
        'node_count': len(nodes)
    }
    
    # Compute shortest path lengths in the graph
    for node in nodes:
        # Local sampling
        local_pairs = neighborhood_sampler.sample_local_neighborhood(
            node, 
            walk_len=local_params.get('walk_len', 5),
            num_walks=local_params.get('num_walks', 10)
        )
        
        # Global sampling
        global_pairs = neighborhood_sampler.sample_global_neighborhood(
            node, 
            dfs_len=global_params.get('walk_len', 10),
            num_walks=global_params.get('num_walks', 10)
        )
        
        # Compute statistics
        if local_pairs:
            local_targets = [pair[1] for pair in local_pairs]
            local_unique = len(set(local_targets))
            stats['avg_local_unique_nodes'] += local_unique / len(nodes)
            
            # Compute average path length
            local_distances = []
            for target in set(local_targets):
                try:
                    distance = nx.shortest_path_length(G, node, target)
                    local_distances.append(distance)
                except:
                    pass
            
            if local_distances:
                stats['avg_local_distance'] += np.mean(local_distances) / len(nodes)
        
        if global_pairs:
            global_targets = [pair[1] for pair in global_pairs]
            global_unique = len(set(global_targets))
            stats['avg_global_unique_nodes'] += global_unique / len(nodes)
            
            # Compute average path length
            global_distances = []
            for target in set(global_targets):
                try:
                    distance = nx.shortest_path_length(G, node, target)
                    global_distances.append(distance)
                except:
                    pass
            
            if global_distances:
                stats['avg_global_distance'] += np.mean(global_distances) / len(nodes)
    
    return stats
