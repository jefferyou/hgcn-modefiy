import os
import time
import datetime
import argparse
import numpy as np
import tensorflow as tf
import networkx as nx
import scipy.sparse as sp
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from utils.data_loader import load_data, preprocess_graph, create_minibatch_generators
from models.hyperbolic_gain import HyperbolicGAINModel

# Set random seed for reproducibility
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hyperbolic GAIN')

    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='graph_data',
                        help='Base directory for graph data')
    parser.add_argument('--dataset', type=str, default='osm_transductive',
                        help='Dataset type: osm_transductive or osm_inductive')
    parser.add_argument('--prefix', type=str, default='linkoping-osm',
                        help='Dataset prefix: linkoping-osm or sweden-osm')

    # Training parameters
    parser.add_argument('--mode', type=str, default='supervised', choices=['supervised', 'unsupervised'],
                        help='Training mode: supervised or unsupervised')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping')
    parser.add_argument('--validate_iter', type=int, default=10,
                        help='Validate every X iterations')
    parser.add_argument('--save_iter', type=int, default=50,
                        help='Save model every X epochs')

    # Model parameters
    parser.add_argument('--hidden_dims', type=str, default='64,64',
                        help='Comma-separated list of hidden dimensions')
    parser.add_argument('--num_heads', type=str, default='8,1',
                        help='Comma-separated list of attention heads per layer')
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'big'],
                        help='Model size: small or big')
    parser.add_argument('--curvature_trainable', type=bool, default=True,
                        help='Whether to train curvature parameter')
    parser.add_argument('--initial_curvature', type=float, default=1.0,
                        help='Initial curvature value')
    parser.add_argument('--activation', type=str, default='elu', choices=['elu', 'relu', 'leaky_relu'],
                        help='Activation function')
    parser.add_argument('--concat_heads', type=bool, default=False,
                        help='Whether to concatenate attention heads')

    # Neighborhood sampling parameters
    parser.add_argument('--use_local_walks', type=bool, default=True,
                        help='Whether to use local (BFS) neighborhood sampling')
    parser.add_argument('--use_global_walks', type=bool, default=True,
                        help='Whether to use global (DFS) neighborhood sampling')
    parser.add_argument('--local_walk_len', type=int, default=5,
                        help='Length of local walks')
    parser.add_argument('--local_num_walks', type=int, default=10,
                        help='Number of local walks per node')
    parser.add_argument('--global_walk_len', type=int, default=10,
                        help='Length of global walks')
    parser.add_argument('--global_num_walks', type=int, default=10,
                        help='Number of global walks per node')
    parser.add_argument('--create_walks', type=bool, default=False,
                        help='Whether to create new walk files')

    # GPU settings
    parser.add_argument('--gpu', type=str, default='0',
                        help='Which GPU to use')
    parser.add_argument('--log_device', type=bool, default=False,
                        help='Whether to log device placement')

    # Output settings
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for logs')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory for saved models')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory for outputs')

    return parser.parse_args()


def get_activation(name):
    """Get TensorFlow activation function by name."""
    if name == 'elu':
        return tf.nn.elu
    elif name == 'relu':
        return tf.nn.relu
    elif name == 'leaky_relu':
        return tf.nn.leaky_relu
    else:
        return tf.nn.elu


def train_supervised(args, data, log_dir):
    """Train Hyperbolic GAIN in supervised mode for node classification."""
    # Unpack data
    G, features, id_map, walks, class_map, _ = data

    # Process graph and features
    adj = preprocess_graph(G)

    # Get class information
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        is_multilabel = True
    else:
        num_classes = max(class_map.values()) + 1
        is_multilabel = False

    print(f"Number of classes: {num_classes}")
    print(f"Multi-label classification: {is_multilabel}")

    # Create minibatch generators
    generators = create_minibatch_generators(
        G, features, walks, id_map, class_map,
        batch_size=args.batch_size
    )

    # Get node splits
    train_nodes = generators['train_nodes']
    val_nodes = generators['val_nodes']
    test_nodes = generators['test_nodes']

    print(f"Number of training nodes: {len(train_nodes)}")
    print(f"Number of validation nodes: {len(val_nodes)}")
    print(f"Number of test nodes: {len(test_nodes)}")

    # Calculate node degrees for sampling
    degrees = np.zeros((len(id_map),))
    for node in G.nodes():
        degrees[id_map[node]] = G.degree(node)

    # Parse hyperparameters
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    num_heads = [int(n) for n in args.num_heads.split(',')]

    # Create placeholders
    placeholders = {
        'batch': tf.placeholder(tf.int32, shape=(None), name='batch'),
        'labels': tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'dropout': tf.placeholder(tf.float32, shape=(), name='dropout'),
        'batch_size': tf.placeholder(tf.int32, name='batch_size')
    }

    # Create model
    activation = get_activation(args.activation)
    model = HyperbolicGAINModel(
        placeholders=placeholders,
        features=features,
        adj=adj,
        degrees=degrees,
        is_supervised=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dims=hidden_dims,
        num_heads=num_heads,
        dropout=args.dropout,
        num_classes=num_classes,
        sparse_inputs=False,
        model_size=args.model_size,
        curvature_trainable=args.curvature_trainable,
        initial_curvature=args.initial_curvature,
        activation=activation,
        concat_heads=args.concat_heads,
        logging=True
    )

    # Set up TensorFlow session
    config = tf.ConfigProto(log_device_placement=args.log_device)
    config.gpu_options.allow_growth = True

    # Initialize session
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # Set up summary writer
    train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(log_dir, 'val'))

    # Track best validation performance for early stopping
    best_val_loss = float('inf')
    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0

    # Start training
    train_time = 0
    print("Starting training...")

    for epoch in range(args.epochs):
        # Training phase
        train_loss = 0
        train_acc = 0
        train_f1_micro = 0
        train_f1_macro = 0
        batch_count = 0

        t1 = time.time()
        train_batches = generators['supervised_train_gen']()

        for batch_nodes, batch_features, batch_labels in train_batches:
            feed_dict = {
                placeholders['batch']: batch_nodes,
                placeholders['labels']: batch_labels,
                placeholders['dropout']: args.dropout,
                placeholders['batch_size']: len(batch_nodes)
            }

            # Run training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs, model.predictions],
                            feed_dict=feed_dict)
            batch_loss, batch_acc, batch_outputs, batch_preds = outs[1], outs[2], outs[3], outs[4]

            # Calculate performance metrics
            if isinstance(batch_preds, np.ndarray) and isinstance(batch_labels, np.ndarray):
                if is_multilabel:
                    # For multi-label classification
                    batch_preds_bin = (batch_outputs > 0).astype(int)
                    batch_f1_micro = f1_score(batch_labels, batch_preds_bin, average='micro')
                    batch_f1_macro = f1_score(batch_labels, batch_preds_bin, average='macro')
                else:
                    # For single-label classification
                    batch_true = np.argmax(batch_labels, axis=1)
                    batch_f1_micro = f1_score(batch_true, batch_preds, average='micro')
                    batch_f1_macro = f1_score(batch_true, batch_preds, average='macro')
            else:
                batch_f1_micro = 0
                batch_f1_macro = 0

            # Update metrics
            train_loss += batch_loss
            train_acc += batch_acc
            train_f1_micro += batch_f1_micro
            train_f1_macro += batch_f1_macro
            batch_count += 1

        # Calculate average metrics
        train_loss /= batch_count
        train_acc /= batch_count
        train_f1_micro /= batch_count
        train_f1_macro /= batch_count
        train_time += time.time() - t1

        # Validation phase
        val_loss = 0
        val_acc = 0
        val_f1_micro = 0
        val_f1_macro = 0
        val_batch_count = 0

        val_batches = generators['supervised_val_gen']()

        for batch_nodes, batch_features, batch_labels in val_batches:
            feed_dict = {
                placeholders['batch']: batch_nodes,
                placeholders['labels']: batch_labels,
                placeholders['dropout']: 0.0,
                placeholders['batch_size']: len(batch_nodes)
            }

            # Run validation
            batch_loss, batch_acc, batch_outputs, batch_preds = sess.run(
                [model.loss, model.accuracy, model.outputs, model.predictions],
                feed_dict=feed_dict
            )

            # Calculate performance metrics
            if isinstance(batch_preds, np.ndarray) and isinstance(batch_labels, np.ndarray):
                if is_multilabel:
                    # For multi-label classification
                    batch_preds_bin = (batch_outputs > 0).astype(int)
                    batch_f1_micro = f1_score(batch_labels, batch_preds_bin, average='micro')
                    batch_f1_macro = f1_score(batch_labels, batch_preds_bin, average='macro')
                else:
                    # For single-label classification
                    batch_true = np.argmax(batch_labels, axis=1)
                    batch_f1_micro = f1_score(batch_true, batch_preds, average='micro')
                    batch_f1_macro = f1_score(batch_true, batch_preds, average='macro')
            else:
                batch_f1_micro = 0
                batch_f1_macro = 0

            # Update metrics
            val_loss += batch_loss
            val_acc += batch_acc
            val_f1_micro += batch_f1_micro
            val_f1_macro += batch_f1_macro
            val_batch_count += 1

        # Calculate average metrics
        val_loss /= val_batch_count
        val_acc /= val_batch_count
        val_f1_micro /= val_batch_count
        val_f1_macro /= val_batch_count

        # Check for early stopping
        if val_loss < best_val_loss or val_f1_micro > best_val_f1:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            if val_f1_micro > best_val_f1:
                best_val_f1 = val_f1_micro

            # Reset patience counter
            patience_counter = 0

            # Save model
            model_path = os.path.join(args.model_dir, f"model_best.ckpt")
            model.save(sess, model_path)
            print(f"Model saved to {model_path}")
        else:
            patience_counter += 1

        # Print progress
        print(f"Epoch: {epoch + 1}/{args.epochs}")
        print(
            f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}, micro-f1={train_f1_micro:.4f}, macro-f1={train_f1_macro:.4f}")
        print(
            f"  Valid: loss={val_loss:.4f}, acc={val_acc:.4f}, micro-f1={val_f1_micro:.4f}, macro-f1={val_f1_macro:.4f}")
        print(f"  Best: loss={best_val_loss:.4f}, acc={best_val_acc:.4f}, micro-f1={best_val_f1:.4f}")

        # Check if training should be stopped due to patience
        if patience_counter >= args.patience:
            print(f"Early stopping after {epoch + 1} epochs!")
            break

        # Periodically save the model
        if (epoch + 1) % args.save_iter == 0:
            model_path = os.path.join(args.model_dir, f"model_epoch_{epoch + 1}.ckpt")
            model.save(sess, model_path)

    # Training completed
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation micro-F1: {best_val_f1:.4f}")

    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss = 0
    test_acc = 0
    test_f1_micro = 0
    test_f1_macro = 0
    test_batch_count = 0
    all_test_preds = []
    all_test_labels = []

    # First load the best model
    model_path = os.path.join(args.model_dir, f"model_best.ckpt")
    model.load(sess, model_path)

    test_batches = generators['supervised_test_gen']()
    for batch_nodes, batch_features, batch_labels in test_batches:
        feed_dict = {
            placeholders['batch']: batch_nodes,
            placeholders['labels']: batch_labels,
            placeholders['dropout']: 0.0,
            placeholders['batch_size']: len(batch_nodes)
        }

        # Run test evaluation
        batch_loss, batch_acc, batch_outputs, batch_preds = sess.run(
            [model.loss, model.accuracy, model.outputs, model.predictions],
            feed_dict=feed_dict
        )

        # Store predictions and labels for overall metrics
        if is_multilabel:
            batch_preds_bin = (batch_outputs > 0).astype(int)
            all_test_preds.append(batch_preds_bin)
        else:
            all_test_preds.append(batch_preds)

        all_test_labels.append(batch_labels)

        # Update metrics
        test_loss += batch_loss
        test_acc += batch_acc
        test_batch_count += 1

    # Calculate average metrics
    test_loss /= test_batch_count
    test_acc /= test_batch_count

    # Calculate overall F1 scores
    all_test_preds = np.concatenate(all_test_preds)
    all_test_labels = np.concatenate(all_test_labels)

    if is_multilabel:
        test_f1_micro = f1_score(all_test_labels, all_test_preds, average='micro')
        test_f1_macro = f1_score(all_test_labels, all_test_preds, average='macro')
    else:
        all_test_true = np.argmax(all_test_labels, axis=1)
        test_f1_micro = f1_score(all_test_true, all_test_preds, average='micro')
        test_f1_macro = f1_score(all_test_true, all_test_preds, average='macro')

    # Print test results
    print("Test results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Micro-F1: {test_f1_micro:.4f}")
    print(f"  Macro-F1: {test_f1_macro:.4f}")

    # Save test results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, f"{args.dataset}_{args.prefix}_supervised_results.txt")

    with open(results_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}, Prefix: {args.prefix}\n")
        f.write(f"Model: Hyperbolic GAIN (Supervised)\n")
        f.write(f"Hidden dims: {args.hidden_dims}\n")
        f.write(f"Num heads: {args.num_heads}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Weight decay: {args.weight_decay}\n")
        f.write(f"Dropout: {args.dropout}\n")
        f.write(f"Curvature trainable: {args.curvature_trainable}\n")
        f.write(f"Initial curvature: {args.initial_curvature}\n")
        f.write(f"Final curvature: {sess.run(model.curvature)[0]:.4f}\n")
        f.write(f"Training time: {train_time:.2f} seconds\n")
        f.write("\nResults:\n")
        f.write(f"Best validation loss: {best_val_loss:.4f}\n")
        f.write(f"Best validation accuracy: {best_val_acc:.4f}\n")
        f.write(f"Best validation micro-F1: {best_val_f1:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Test micro-F1: {test_f1_micro:.4f}\n")
        f.write(f"Test macro-F1: {test_f1_macro:.4f}\n")

    print(f"Results saved to {results_file}")

    # Close session
    sess.close()

    return {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'test_f1_micro': test_f1_micro,
        'test_f1_macro': test_f1_macro,
        'curvature': sess.run(model.curvature)[0]
    }


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Set up GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Set log directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f"{args.dataset}_{args.prefix}_{args.mode}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Set up walk parameters
    walk_params = {
        'local_walk_len': args.local_walk_len,
        'local_num_walks': args.local_num_walks,
        'global_walk_len': args.global_walk_len,
        'global_num_walks': args.global_num_walks,
        'walk_seed': seed
    }

    # Load data
    print(f"Loading {args.dataset} data with prefix {args.prefix}...")
    data_path = os.path.join(args.data_dir, args.dataset)

    data = load_data(
        args.prefix,
        data_path,
        normalize=True,
        use_walks=(args.mode == 'unsupervised'),
        walk_params=walk_params,
        create_walks=args.create_walks,
        compute_neighborhoods=True
    )

    # Display dataset info
    G, features, id_map, walks, class_map, neighborhood_stats = data
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Feature matrix shape: {features.shape}")

    if neighborhood_stats:
        print("Neighborhood statistics:")
        print(f"  Average local distance: {neighborhood_stats['avg_local_distance']:.2f}")
        print(f"  Average global distance: {neighborhood_stats['avg_global_distance']:.2f}")
        print(f"  Average local unique nodes: {neighborhood_stats['avg_local_unique_nodes']:.2f}")
        print(f"  Average global unique nodes: {neighborhood_stats['avg_global_unique_nodes']:.2f}")

    # Train model - THIS PART IS CRUCIAL
    if args.mode == 'supervised':
        print("Training in supervised mode...")
        results = train_supervised(args, data, log_dir)
        print("Supervised training completed!")
        print(f"Test accuracy: {results['test_acc']:.4f}")
        print(f"Test F1 (micro): {results['test_f1_micro']:.4f}")
        print(f"Test F1 (macro): {results['test_f1_macro']:.4f}")
    else:
        print("Training in unsupervised mode...")
        results = train_unsupervised(args, data, log_dir)
        print("Unsupervised training completed!")
        print(f"Best MRR: {results['mrr']:.4f}")

    print(f"Final curvature: {results['curvature']:.4f}")
    return results


# Make sure to call main when the script is run directly
if __name__ == "__main__":
    main()