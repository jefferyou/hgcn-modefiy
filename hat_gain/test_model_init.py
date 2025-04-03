import os
import argparse
import tensorflow as tf
import numpy as np
import networkx as nx
import random

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Hyperbolic GAIN model initialization')
    
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    
    return parser.parse_args()


def create_synthetic_graph(num_nodes=100, avg_degree=5):
    """
    Create a synthetic graph for testing.
    
    Args:
        num_nodes: Number of nodes
        avg_degree: Average node degree
        
    Returns:
        G: NetworkX graph
        features: Node features
    """
    # Create random graph
    G = nx.random_regular_graph(avg_degree, num_nodes)
    
    # Add node attributes
    for i in range(num_nodes):
        # Train/val/test split
        if i < 0.7 * num_nodes:
            G.nodes[i]['val'] = False
            G.nodes[i]['test'] = False
        elif i < 0.85 * num_nodes:
            G.nodes[i]['val'] = True
            G.nodes[i]['test'] = False
        else:
            G.nodes[i]['val'] = False
            G.nodes[i]['test'] = True
    
    # Create random features
    features = np.random.normal(size=(num_nodes, 10))
    
    return G, features


def generate_context_pairs(G, num_walks=5, walk_len=3):
    """
    Generate context pairs for nodes.
    
    Args:
        G: NetworkX graph
        num_walks: Number of walks per node
        walk_len: Length of each walk
        
    Returns:
        context_pairs: List of context pairs
    """
    context_pairs = []
    nodes = list(G.nodes())
    
    for node in nodes:
        for _ in range(num_walks):
            # Do a simple random walk
            curr_node = node
            for _ in range(walk_len):
                if G.degree(curr_node) == 0:
                    break
                next_node = random.choice(list(G.neighbors(curr_node)))
                
                # Add co-occurrence
                if curr_node != node:
                    context_pairs.append((node, curr_node))
                    
                curr_node = next_node
    
    return context_pairs


def test_model_initialization():
    """Test the initialization of the Hyperbolic GAIN model."""
    print("Creating synthetic graph...")
    G, features = create_synthetic_graph()
    
    print("Generating context pairs...")
    context_pairs = generate_context_pairs(G)
    
    # Create ID map and calculate degrees
    id_map = {node: int(node) for node in G.nodes()}
    degrees = np.array([G.degree(n) for n in range(len(G))])
    
    # Create TensorFlow graph
    with tf.Graph().as_default():
        print("Setting up TensorFlow graph...")
        
        # Create placeholders
        placeholders = {
            'batch1': tf.placeholder(tf.int32, shape=(None), name='batch1'),
            'batch2': tf.placeholder(tf.int32, shape=(None), name='batch2'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
            'batch_size': tf.placeholder(tf.int32, name='batch_size'),
            'weight_decay': tf.placeholder_with_default(0., shape=(), name='weight_decay')
        }
        
        # Create dummy adjacency info for the sampler
        adj_info = tf.Variable(tf.constant(np.zeros((features.shape[0], 10), dtype=np.int32)), 
                              trainable=False, name="adj_info")
        
        # Import required components
        try:
            # Set up required imports
            from utils.neigh_samplers import UniformNeighborSampler
            
            # Define a simple SAGEInfo class for testing
            class SAGEInfo(object):
                def __init__(self, layer_name, neigh_sampler, num_samples, output_dim):
                    self.layer_name = layer_name
                    self.neigh_sampler = neigh_sampler
                    self.num_samples = num_samples
                    self.output_dim = output_dim
            
            # Create sampler and layer infos
            sampler = UniformNeighborSampler(adj_info)
            layer_infos = [
                SAGEInfo("layer1", sampler, 5, 16),
                SAGEInfo("layer2", sampler, 5, 16)
            ]
            
            # Import the model class
            from models.enhanced_hyperbolic_gain import EnhancedHyperbolicGAIN
            
            # Create model
            print("Creating Hyperbolic GAIN model...")
            model = EnhancedHyperbolicGAIN(
                placeholders=placeholders,
                features=features,
                adj=adj_info,
                degrees=degrees,
                layer_infos=layer_infos,
                context_pairs=context_pairs,
                concat=True,
                neg_sample_size=10
            )
            
            # Initialize variables
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            
            # Create session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            
            with tf.Session(config=config) as sess:
                # Initialize variables
                print("Initializing variables...")
                sess.run(init_op)
                
                # Test forward pass
                print("Testing forward pass...")
                batch_size = 5
                feed_dict = {
                    placeholders['batch1']: np.random.randint(0, 10, size=batch_size),
                    placeholders['batch2']: np.random.randint(0, 10, size=batch_size),
                    placeholders['dropout']: 0.0,
                    placeholders['batch_size']: batch_size,
                    placeholders['weight_decay']: 0.0001
                }
                
                try:
                    loss, mrr = sess.run([model.loss, model.mrr], feed_dict=feed_dict)
                    print(f"Forward pass successful!")
                    print(f"Loss: {loss}, MRR: {mrr}")
                    print("✓ Model initialized and forward pass executed successfully!")
                except Exception as e:
                    print(f"Error during forward pass: {e}")
                    print("✗ Model initialization failed during forward pass.")
                    
        except ImportError as e:
            print(f"Import error: {e}")
            print("✗ Missing required components.")
        except Exception as e:
            print(f"Error during model creation: {e}")
            print("✗ Model initialization failed.")


if __name__ == "__main__":
    args = parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # Run test
    test_model_initialization()
