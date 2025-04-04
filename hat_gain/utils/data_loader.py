import numpy as np
import json
import os
import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from utils.neighborhood_sampler import NeighborhoodSampler

def load_data(prefix, data_dir, normalize=True, use_walks=True, 
              walk_params=None, create_walks=False, compute_neighborhoods=True):
    """
    Load graph data for Hyperbolic GAIN.
    
    Args:
        prefix: Prefix of the dataset files (e.g., 'linkoping-osm')
        data_dir: Directory containing the dataset files
        normalize: Whether to normalize feature vectors
        use_walks: Whether to use random walks for context
        walk_params: Parameters for random walks
        create_walks: Whether to create new walk files
        compute_neighborhoods: Whether to compute neighborhood statistics
        
    Returns:
        G: NetworkX graph
        features: Node features
        id_map: Map from node IDs to indices
        walks: List of random walks (context pairs)
        class_map: Map from node IDs to class labels
        neighborhood_stats: Statistics about neighborhood sampling (if computed)
    """
    # Default walk parameters
    if walk_params is None:
        walk_params = {
            'local_walk_len': 5,
            'local_num_walks': 10,
            'global_walk_len': 10,
            'global_num_walks': 10,
            'walk_seed': 42
        }
    
    # Load graph
    G_data = json.load(open(os.path.join(data_dir, f"{prefix}-G.json")))
    G = json_graph.node_link_graph(G_data)
    
    # Handle node ID conversion
    if isinstance(list(G.nodes())[0], int):
        conversion = lambda n: int(n)
    else:
        conversion = lambda n: n
    
    # Load features
    if os.path.exists(os.path.join(data_dir, f"{prefix}-feats.npy")):
        features = np.load(os.path.join(data_dir, f"{prefix}-feats.npy"))
    else:
        print("No features present. Using identity features.")
        features = None
    
    # Load ID map
    id_map = json.load(open(os.path.join(data_dir, f"{prefix}-id_map.json")))
    id_map = {conversion(k): int(v) for k, v in id_map.items()}
    
    # Load class map
    class_map = json.load(open(os.path.join(data_dir, f"{prefix}-class_map.json")))
    
    # Check if class map contains lists or single values
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n: n
    else:
        lab_conversion = lambda n: int(n)
    
    class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}
    
    # Remove nodes without train/val/test annotations
    broken_count = 0
    nodes_to_remove = []
    for node in G.nodes():
        if 'val' not in G.nodes[node] or 'test' not in G.nodes[node]:
            nodes_to_remove.append(node)
            broken_count += 1
    
    G.remove_nodes_from(nodes_to_remove)
    print(f"Removed {broken_count} nodes without proper annotations")
    
    # Add train_removed attribute to edges
    for edge in G.edges():
        if (G.nodes[edge[0]]['val'] or G.nodes[edge[1]]['val'] or
            G.nodes[edge[0]]['test'] or G.nodes[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False
    
    # Normalize features if requested
    if normalize and features is not None:
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']])
        train_feats = features[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        features = scaler.transform(features)
    
    # Process and load walks (context pairs)
    walks = []
    if use_walks:
        # Check if walk files exist
        local_walks_file = os.path.join(data_dir, f"{prefix}-walks.txt")
        global_walks_file = os.path.join(data_dir, f"{prefix}-dfs-walks.txt")
        
        # Create walks if requested or if files don't exist
        if create_walks or not os.path.exists(local_walks_file) or not os.path.exists(global_walks_file):
            print("Generating new neighborhood walks...")
            sampler = NeighborhoodSampler(G, walk_seed=walk_params['walk_seed'])
            
            # Generate walks
            walk_results = sampler.generate_and_save_all_pairs(
                output_dir=data_dir,
                prefix=prefix,
                nodes=[n for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']],
                local_params={
                    'walk_len': walk_params['local_walk_len'],
                    'num_walks': walk_params['local_num_walks']
                },
                global_params={
                    'walk_len': walk_params['global_walk_len'],
                    'num_walks': walk_params['global_num_walks']
                }
            )
            
            # Update file paths
            local_walks_file = walk_results.get('local_pairs_file', local_walks_file)
            global_walks_file = walk_results.get('global_pairs_file', global_walks_file)
        
        # Load local walks
        if os.path.exists(local_walks_file):
            with open(local_walks_file) as fp:
                for line in fp:
                    walks.append(list(map(conversion, line.split())))
            print(f"Loaded {len(walks)} local neighborhood walks from {local_walks_file}")
        
        # Load global walks
        if os.path.exists(global_walks_file):
            with open(global_walks_file) as fp:
                global_walk_count = 0
                for line in fp:
                    walks.append(list(map(conversion, line.split())))
                    global_walk_count += 1
            print(f"Loaded {global_walk_count} global neighborhood walks from {global_walks_file}")
    
    # Compute neighborhood statistics if requested
    neighborhood_stats = None
    if compute_neighborhoods:
        print("Computing neighborhood statistics...")
        sampler = NeighborhoodSampler(G, walk_seed=walk_params['walk_seed'])
        
        from utils.neighborhood_sampler import compute_neighborhood_stats
        neighborhood_stats = compute_neighborhood_stats(
            G, 
            sampler, 
            sample_size=min(500, len(G.nodes())),
            local_params={
                'walk_len': walk_params['local_walk_len'],
                'num_walks': walk_params['local_num_walks']
            },
            global_params={
                'walk_len': walk_params['global_walk_len'],
                'num_walks': walk_params['global_num_walks']
            }
        )
        
        print("Neighborhood Statistics:")
        print(f"Avg local distance: {neighborhood_stats['avg_local_distance']:.2f}")
        print(f"Avg global distance: {neighborhood_stats['avg_global_distance']:.2f}")
        print(f"Avg local unique nodes: {neighborhood_stats['avg_local_unique_nodes']:.2f}")
        print(f"Avg global unique nodes: {neighborhood_stats['avg_global_unique_nodes']:.2f}")
    
    return G, features, id_map, walks, class_map, neighborhood_stats


def preprocess_graph(graph):
    """
    Convert NetworkX graph to sparse adjacency matrix.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        scipy sparse adjacency matrix
    """
    # Create adjacency matrix
    adj = nx.adjacency_matrix(graph)
    
    # Add self-loops and normalize
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj)
    
    return adj


def normalize_adj(adj):
    """
    Symmetrically normalize adjacency matrix.
    
    Args:
        adj: adjacency matrix
        
    Returns:
        normalized adjacency matrix
    """
    # Calculate degree matrix
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # Normalize
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_features(features):
    """
    Row-normalize feature matrix.
    
    Args:
        features: feature matrix
        
    Returns:
        normalized feature matrix
    """
    # Calculate row sums
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    
    # Normalize
    features = r_mat_inv.dot(features)
    
    return features


def create_minibatch_generators(G, features, walks, id_map, class_map, batch_size=512, 
                                val_frac=0.15, test_frac=0.15, seed=42):
    """
    Create train/val/test splits and minibatch generators.
    
    Args:
        G: NetworkX graph
        features: Feature matrix
        walks: List of random walks
        id_map: Map from node IDs to indices
        class_map: Map from node IDs to classes
        batch_size: Batch size for training
        val_frac: Fraction of nodes for validation
        test_frac: Fraction of nodes for testing
        seed: Random seed
    
    Returns:
        Dictionary with train/val/test splits and minibatch generators
    """
    np.random.seed(seed)
    
    # Get nodes for each split using existing attributes
    train_nodes = [n for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']]
    val_nodes = [n for n in G.nodes() if G.nodes[n]['val']]
    test_nodes = [n for n in G.nodes() if G.nodes[n]['test']]
    
    # Create labels for each node
    labels = {}
    for node in G.nodes():
        labels[node] = class_map[node]
    
    # Create supervised minibatch generator
    from tensorflow.contrib.keras.api.keras.preprocessing.sequence import pad_sequences
    
    def create_supervised_batches(nodes, is_training=True):
        np.random.shuffle(nodes)
        for i in range(0, len(nodes), batch_size):
            batch_nodes = nodes[i:i+batch_size]
            batch_features = []
            batch_labels = []
            
            for node in batch_nodes:
                batch_features.append(features[id_map[node]])
                batch_labels.append(labels[node])
            
            yield batch_nodes, np.array(batch_features), np.array(batch_labels)
    
    # Create unsupervised (link prediction) minibatch generator
    def create_unsupervised_batches(context_pairs, is_training=True):
        np.random.shuffle(context_pairs)
        for i in range(0, len(context_pairs), batch_size):
            batch_pairs = context_pairs[i:i+batch_size]
            batch_node1 = []
            batch_node2 = []
            
            for pair in batch_pairs:
                batch_node1.append(id_map[pair[0]])
                batch_node2.append(id_map[pair[1]])
            
            yield batch_pairs, np.array(batch_node1), np.array(batch_node2)
    
    # Return all generators
    return {
        'train_nodes': train_nodes,
        'val_nodes': val_nodes,
        'test_nodes': test_nodes,
        'supervised_train_gen': lambda: create_supervised_batches(train_nodes, True),
        'supervised_val_gen': lambda: create_supervised_batches(val_nodes, False),
        'supervised_test_gen': lambda: create_supervised_batches(test_nodes, False),
        'unsupervised_train_gen': lambda: create_unsupervised_batches(walks, True)
    }
