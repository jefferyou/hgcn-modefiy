import time
import argparse
import numpy as np
import tensorflow as tf
import os
import networkx as nx
from models import SpHyperbolicGAIN
from utils import process
from sklearn.metrics import f1_score
import json
import scipy.sparse as sp
from networkx.readwrite import json_graph


def parse_args():
    parser = argparse.ArgumentParser(description='Train Hyperbolic GAIN on OpenStreetMap datasets')
    parser.add_argument('-gpu', nargs='?', default='0', help='the ID for GPU')
    parser.add_argument('-dataset', nargs='?', default='osm_transductive', help='osm_transductive or osm_inductive')
    parser.add_argument('-prefix', nargs='?', default='linkoping-osm', help='linkoping-osm or sweden-osm')
    parser.add_argument('-lr', default=0.005, type=float, help='learning rate')
    parser.add_argument('-l2', default=0.0001, type=float, help='l2 regularization')
    parser.add_argument('-units', default=256, type=int, help='dimension for hidden unit')
    parser.add_argument('-heads', default=16, type=int, help='number of multi-heads')
    parser.add_argument('-dropout', default=0.1, type=float, help='dropout rate')
    parser.add_argument('-epochs', default=2000, type=int, help='maximum number of epochs')
    parser.add_argument('-patience', default=100, type=int, help='patience for early stopping')
    parser.add_argument('-c', default=1, type=int, help='0: untrainable curvature; 1: trainable curvature')
    parser.add_argument('-data_dir', default='graph_data', help='directory containing the datasets')
    parser.add_argument('-model_size', default='big', help='model size: small or big')
    parser.add_argument('-use_global_walks', type=int, default=1, help='Use global neighborhood walks: 0=off, 1=on')
    return parser.parse_args()


def load_osm_data(args):
    """
    Load OSM data from GAIN's dataset format
    """
    print(f"Loading {args.dataset} with prefix {args.prefix}")

    from sklearn.preprocessing import StandardScaler

    # Set data paths
    data_dir = os.path.join(args.data_dir, args.dataset)
    prefix = args.prefix

    print(f"Loading data from: {data_dir}")

    try:
        # Load graph from JSON
        print("Loading graph...")
        G_data = json.load(open(os.path.join(data_dir, f"{prefix}-G.json")))
        G = json_graph.node_link_graph(G_data)
        print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        # Load features
        print("Loading features...")
        features = np.load(os.path.join(data_dir, f"{prefix}-feats.npy"))
        print(f"Features shape: {features.shape}")

        # Load ID map
        print("Loading ID map...")
        id_map = json.load(open(os.path.join(data_dir, f"{prefix}-id_map.json")))

        # Convert ID map keys and values to integers if needed
        if len(id_map) > 0:
            if isinstance(next(iter(id_map.keys())), str):
                id_map = {int(k): int(v) for k, v in id_map.items()}
            else:
                id_map = {k: int(v) for k, v in id_map.items()}

        # Load class map
        print("Loading class map...")
        class_map = json.load(open(os.path.join(data_dir, f"{prefix}-class_map.json")))

        # Determine if it's multi-label or single-label
        first_value = next(iter(class_map.values()))
        is_multilabel = isinstance(first_value, list)

        # Convert class map keys and values to appropriate types
        if is_multilabel:
            class_map = {int(k) if isinstance(k, str) else k: v for k, v in class_map.items()}
            num_classes = len(first_value)
            print(f"Multi-label classification with {num_classes} classes")
        else:
            class_map = {int(k) if isinstance(k, str) else k: int(v) for k, v in class_map.items()}
            max_class_id = max(class_map.values())
            num_classes = max_class_id + 1
            print(f"Single-label classification with {num_classes} classes")

        # Create train, val, test masks
        print("Creating masks...")
        train_mask = []
        val_mask = []
        test_mask = []

        for _, data in G.nodes(data=True):
            is_val = data.get('val', False)
            is_test = data.get('test', False)

            val_mask.append(is_val)
            test_mask.append(is_test)
            train_mask.append(not is_val and not is_test)

        train_mask = np.array(train_mask)
        val_mask = np.array(val_mask)
        test_mask = np.array(test_mask)

        print(f"Train mask: {np.sum(train_mask)} nodes")
        print(f"Validation mask: {np.sum(val_mask)} nodes")
        print(f"Test mask: {np.sum(test_mask)} nodes")

        # Get adjacency matrix
        print("Creating adjacency matrix...")
        adj = nx.adjacency_matrix(G)

        # Create labels matrix
        print("Creating labels...")
        nodes = list(G.nodes())
        labels = np.zeros((len(nodes), num_classes))

        node_to_idx = {node: i for i, node in enumerate(nodes)}

        for node_id, class_id in class_map.items():
            if node_id not in node_to_idx:
                continue

            idx = node_to_idx[node_id]

            if is_multilabel:
                labels[idx] = class_id
            else:
                labels[idx, class_id] = 1

        # Create train/val/test labels
        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)

        y_train[train_mask] = labels[train_mask]
        y_val[val_mask] = labels[val_mask]
        y_test[test_mask] = labels[test_mask]

        # Preprocess features
        print("Standardizing features...")
        train_features = features[train_mask]
        scaler = StandardScaler()
        scaler.fit(train_features)
        features = scaler.transform(features)

        print("Data loading completed successfully")

        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise


def main(args):
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Load data
    print("Loading dataset...")
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_osm_data(args)

    # Parameters
    batch_size = 1
    nb_epochs = args.epochs
    patience = args.patience
    lr = args.lr
    l2_coef = args.l2
    hid_units = [args.units]
    n_heads = [args.heads, 1]  # Multi-head attention
    drop_out = args.dropout
    nonlinearity = tf.nn.elu
    model = SpHyperbolicGAIN
    c = args.c
    model_size = args.model_size

    # Print configuration
    print('=' * 50)
    print('Hyperbolic GAIN Configuration:')
    print('=' * 50)
    print(f'Dataset: {args.dataset}')
    print(f'Prefix: {args.prefix}')
    print(f'Learning rate: {lr}')
    print(f'L2 regularization: {l2_coef}')
    print(f'Hidden units: {hid_units}')
    print(f'Attention heads: {n_heads}')
    print(f'Dropout rate: {drop_out}')
    print(f'Curvature training: {"Yes" if c == 1 else "No"}')
    print(f'Model size: {model_size}')
    print('=' * 50)

    # Prepare data for training
    sparse = True  # Use sparse tensor operations

    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    nb_classes = y_train.shape[1]

    features = features[np.newaxis]
    y_train = y_train[np.newaxis]
    y_val = y_val[np.newaxis]
    y_test = y_test[np.newaxis]
    train_mask = train_mask[np.newaxis]
    val_mask = val_mask[np.newaxis]
    test_mask = test_mask[np.newaxis]

    # Preprocess adjacency matrix
    if sparse:
        biases = process.preprocess_adj_bias(adj)
    else:
        adj = adj.todense()
        adj = adj[np.newaxis]
        biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

    # Build TensorFlow graph
    with tf.Graph().as_default():
        with tf.name_scope('input'):
            ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
            if sparse:
                bias_in = tf.sparse_placeholder(dtype=tf.float32)
            else:
                bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
            lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
            msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
            attn_drop = tf.placeholder(dtype=tf.float32, shape=())
            ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
            is_train = tf.placeholder(dtype=tf.bool, shape=())

        # Build model
        print("Building Hyperbolic GAIN model...")
        logits, emb, curvature = model.inference(
            ftr_in, nb_classes, nb_nodes, is_train,
            attn_drop, ffd_drop,
            bias_mat=bias_in,
            hid_units=hid_units, n_heads=n_heads,
            activation=nonlinearity, c=c,
            model_size=model_size,
            use_global_walks=(args.use_global_walks == 1)
        )

        # Reshape for loss computation
        log_resh = tf.reshape(logits, [-1, nb_classes])
        lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])
        
        # Define loss and accuracy
        loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

        # For F1 score computation
        pred_all = tf.cast(tf.argmax(log_resh, 1), dtype=tf.int32)
        real_all = tf.cast(tf.argmax(lab_resh, 1), dtype=tf.int32)
        
        # Define optimizer
        train_op = model.my_training(loss, lr, l2_coef)

        # Initialize variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Training settings
        vlss_mn = np.inf
        vacc_mx = 0.0
        curr_step = 0
        
        # Create session
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            
            # Start training
            print("\nStarting training...")
            
            for epoch in range(nb_epochs):
                tr_step = 0
                tr_size = features.shape[0]
                train_loss_avg = 0
                train_acc_avg = 0

                # Training phase
                while tr_step * batch_size < tr_size:
                    if sparse:
                        bbias = biases
                    else:
                        bbias = biases[tr_step * batch_size:(tr_step + 1) * batch_size]

                    _, loss_value_tr, acc_tr, train_emb, curvature_this = sess.run(
                        [train_op, loss, accuracy, emb, curvature],
                        feed_dict={
                            ftr_in: features[tr_step * batch_size:(tr_step + 1) * batch_size],
                            bias_in: bbias,
                            lbl_in: y_train[tr_step * batch_size:(tr_step + 1) * batch_size],
                            msk_in: train_mask[tr_step * batch_size:(tr_step + 1) * batch_size],
                            is_train: True,
                            attn_drop: drop_out, ffd_drop: drop_out
                        }
                    )
                    train_loss_avg += loss_value_tr
                    train_acc_avg += acc_tr
                    tr_step += 1

                # Validation phase
                vl_step = 0
                vl_size = features.shape[0]
                val_loss_avg = 0
                val_acc_avg = 0

                while vl_step * batch_size < vl_size:
                    if sparse:
                        bbias = biases
                    else:
                        bbias = biases[vl_step * batch_size:(vl_step + 1) * batch_size]
                        
                    loss_value_vl, acc_vl = sess.run(
                        [loss, accuracy],
                        feed_dict={
                            ftr_in: features[vl_step * batch_size:(vl_step + 1) * batch_size],
                            bias_in: bbias,
                            lbl_in: y_val[vl_step * batch_size:(vl_step + 1) * batch_size],
                            msk_in: val_mask[vl_step * batch_size:(vl_step + 1) * batch_size],
                            is_train: False,
                            attn_drop: 0.0, ffd_drop: 0.0
                        }
                    )
                    val_loss_avg += loss_value_vl
                    val_acc_avg += acc_vl
                    vl_step += 1

                # Print epoch results
                print(f'Epoch: {epoch:4d}, '
                      f'Train Loss: {train_loss_avg/tr_step:.5f}, '
                      f'Train Acc: {train_acc_avg/tr_step:.5f}, '
                      f'Val Loss: {val_loss_avg/vl_step:.5f}, '
                      f'Val Acc: {val_acc_avg/vl_step:.5f}, '
                      f'Curvature: {curvature_this[0]:.5f}')
                
                # Early stopping check
                if val_acc_avg/vl_step >= vacc_mx or val_loss_avg/vl_step <= vlss_mn:
                    if val_acc_avg/vl_step >= vacc_mx and val_loss_avg/vl_step <= vlss_mn:
                        vacc_early_model = val_acc_avg/vl_step
                        vlss_early_model = val_loss_avg/vl_step
                        best_epoch = epoch
                    
                    vacc_mx = np.max((val_acc_avg/vl_step, vacc_mx))
                    vlss_mn = np.min((val_loss_avg/vl_step, vlss_mn))
                    curr_step = 0
                    
                    # Test model on best validation performance
                    ts_size = features.shape[0]
                    ts_step = 0
                    ts_loss = 0.0
                    ts_acc = 0.0
                    ts_macro = 0.0
                    ts_micro = 0.0
                    
                    while ts_step * batch_size < ts_size:
                        if sparse:
                            bbias = biases
                        else:
                            bbias = biases[ts_step * batch_size:(ts_step + 1) * batch_size]
                            
                        loss_value_ts, acc_ts, test_emb, real_y, pred_y = sess.run(
                            [loss, accuracy, emb, real_all, pred_all],
                            feed_dict={
                                ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size],
                                bias_in: bbias,
                                lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                                msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
                                is_train: False,
                                attn_drop: 0.0, ffd_drop: 0.0
                            }
                        )
                        ts_loss += loss_value_ts
                        ts_acc += acc_ts
                        ts_step += 1
                        
                        # Calculate F1 scores
                        ts_macro += f1_score(real_y[test_mask[0]], pred_y[test_mask[0]], average='macro')
                        ts_micro += f1_score(real_y[test_mask[0]], pred_y[test_mask[0]], average='micro')
                else:
                    curr_step += 1
                    if curr_step == patience:
                        print('Early stopping! Min loss: {:.5f}, Max accuracy: {:.5f}'.format(vlss_mn, vacc_mx))
                        print('Early stop model validation loss: {:.5f}, accuracy: {:.5f}'.format(
                            vlss_early_model, vacc_early_model))
                        print(f'Best epoch: {best_epoch}')
                        break
            
            # Final test results
            print('\nTraining completed!')
            print('Test results:')
            print('Loss: {:.5f}, Accuracy: {:.5f}'.format(ts_loss/ts_step, ts_acc/ts_step))
            print('Macro F1: {:.5f}, Micro F1: {:.5f}'.format(ts_macro/ts_step, ts_micro/ts_step))
            print('Final curvature value: {:.5f}'.format(curvature_this[0]))
            
            # Save results
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            results_dir = 'results'
            os.makedirs(results_dir, exist_ok=True)
            
            results = {
                'dataset': args.dataset,
                'prefix': args.prefix,
                'lr': args.lr,
                'l2': args.l2,
                'units': args.units,
                'heads': args.heads,
                'dropout': args.dropout,
                'model_size': args.model_size,
                'curvature': float(curvature_this[0]),
                'best_epoch': best_epoch,
                'test_loss': float(ts_loss/ts_step),
                'test_acc': float(ts_acc/ts_step),
                'test_macro_f1': float(ts_macro/ts_step),
                'test_micro_f1': float(ts_micro/ts_step),
                'val_loss': float(vlss_mn),
                'val_acc': float(vacc_mx)
            }
            
            # Save to JSON file
            results_file = os.path.join(
                results_dir, 
                f'hyperbolic_gain_{args.dataset}_{args.prefix}_{timestamp}.json'
            )
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
                
            print(f'\nResults saved to: {results_file}')


if __name__ == "__main__":
    args = parse_args()
    main(args)