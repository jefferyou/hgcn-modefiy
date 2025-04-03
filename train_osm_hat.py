import time
import argparse
import numpy as np
import tensorflow as tf
import os
import networkx as nx
from models import SpMHGAT
from utils import process
from sklearn.metrics import f1_score
import json
import scipy.sparse as sp
from networkx.readwrite import json_graph


def parse_args():
    parser = argparse.ArgumentParser(description='Train HAT on OpenStreetMap datasets')
    parser.add_argument('-gpu', nargs='?', default='0', help='the ID for GPU')
    parser.add_argument('-dataset', nargs='?', default='osm_transductive', help='osm_transductive or osm_inductive')
    parser.add_argument('-prefix', nargs='?', default='linkoping-osm', help='linkoping-osm or sweden-osm')
    parser.add_argument('-model', default='hat', choices=['hat', 'gain', 'hybrid'],
                      help='Model type: hat, gain, or hybrid (hat+gain fusion)')
    parser.add_argument('-lr', default=0.005, type=float, help='learning rate')
    parser.add_argument('-l2', default=0.0001, type=float, help='l2 regularization')
    parser.add_argument('-units', default=64, type=int, help='dimension for hidden unit')
    parser.add_argument('-heads', default=8, type=int, help='number of multi-heads')
    parser.add_argument('-dropout', default=0.2, type=float, help='dropout rate')
    parser.add_argument('-epochs', default=2000, type=int, help='maximum number of epochs')
    parser.add_argument('-patience', default=50, type=int, help='patience for early stopping')
    parser.add_argument('-c', default=1, type=int, help='0: untrainable curvature; 1: trainable curvature')
    parser.add_argument('-data_dir', default='graph_data', help='directory containing the datasets')
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

        # Check for val/test attributes
        val_present = False
        test_present = False
        for _, data in list(G.nodes(data=True))[:5]:
            if 'val' in data:
                val_present = True
            if 'test' in data:
                test_present = True

        print(f"Val attribute present: {val_present}")
        print(f"Test attribute present: {test_present}")

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

        print(f"ID map contains {len(id_map)} entries")

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
            # Find the maximum class ID instead of counting unique values
            max_class_id = max(class_map.values())
            num_classes = max_class_id + 1  # Classes are 0-indexed
            print(f"Single-label classification with {num_classes} classes (max class ID: {max_class_id})")

        # Create train, val, test masks
        print("Creating masks...")
        train_mask = []
        val_mask = []
        test_mask = []

        # Use an explicit loop to avoid any API issues
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

        # First, verify all class IDs are within range
        if not is_multilabel:
            out_of_range_classes = [class_id for class_id in class_map.values() if class_id >= num_classes]
            if out_of_range_classes:
                print(f"WARNING: Found {len(out_of_range_classes)} class IDs that are out of range!")
                print(f"Max expected class ID: {num_classes - 1}, but found: {max(out_of_range_classes)}")
                # Adjust num_classes if needed
                old_num_classes = num_classes
                num_classes = max(out_of_range_classes) + 1
                print(f"Adjusting number of classes from {old_num_classes} to {num_classes}")
                # Re-create labels array with correct size
                labels = np.zeros((len(nodes), num_classes))

        for node_id, class_id in class_map.items():
            # Skip nodes not in the graph
            if node_id not in node_to_idx:
                continue

            idx = node_to_idx[node_id]

            if is_multilabel:
                labels[idx] = class_id
            else:
                labels[idx, class_id] = 1

        print(f"Labels shape: {labels.shape}")

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
    import networkx as nx
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_osm_data(args)

    # Parameters
    batch_size = 1
    nb_epochs = args.epochs
    patience = args.patience
    lr = args.lr
    l2_coef = args.l2
    hid_units = [args.units]
    n_heads = [args.heads, 1]
    drop_out = args.dropout
    nonlinearity = tf.nn.elu

    # Select model based on argument
    if args.model == 'hat':
        from models import SpMHGAT as model
        print("Using HAT (Hyperbolic Attention) model")
    elif args.model == 'gain':
        from models import SpHyperbolicGAIN as model
        print("Using GAIN (Graph Attention Isomorphism) in hyperbolic space")
    elif args.model == 'hybrid':
        from models import SpHyperbolicGAIN as model
        print("Using Hybrid HAT-GAIN model")
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    c = args.c

    time_begin = time.strftime("%Y.%m.%d-%H.%M.%S", time.localtime())

    print('Dataset:', args.dataset, args.prefix)
    print('----- Opt. hyperparams -----')
    print('lr:', lr)
    print('l2_coef:', l2_coef)
    print('----- Archi. hyperparams -----')
    print('nb. layers:', len(hid_units))
    print('nb. units per layer:', hid_units)
    print('nb. attention heads:', n_heads)
    print('nonlinearity:', nonlinearity)
    print('model:', model)

    sparse = True

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

    if sparse:
        biases = process.preprocess_adj_bias(adj)
    else:
        adj = adj.todense()
        adj = adj[np.newaxis]
        biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

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

        logits, emb, curvature = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                                 attn_drop, ffd_drop,
                                                 bias_mat=bias_in,
                                                 hid_units=hid_units, n_heads=n_heads,
                                                 activation=nonlinearity, c=c)
        log_resh = tf.reshape(logits, [-1, nb_classes])
        lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
        msk_resh = tf.reshape(msk_in, [-1])
        loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
        accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

        pred_all = tf.cast(tf.argmax(log_resh, 1), dtype=tf.int32)
        real_all = tf.cast(tf.argmax(lab_resh, 1), dtype=tf.int32)
        train_op = model.my_training(loss, lr, l2_coef)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        vlss_mn = np.inf
        vacc_mx = 0.0
        curr_step = 0
        with tf.Session(config=config) as sess:
            sess.run(init_op)
            time_run = time.strftime("%Y.%m.%d-%H.%M.%S", time.localtime())
            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

            for epoch in range(nb_epochs):
                tr_step = 0
                tr_size = features.shape[0]

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
                            attn_drop: drop_out, ffd_drop: drop_out})
                    train_loss_avg += loss_value_tr
                    train_acc_avg += acc_tr
                    tr_step += 1

                vl_step = 0
                vl_size = features.shape[0]

                while vl_step * batch_size < vl_size:
                    if sparse:
                        bbias = biases
                    else:
                        bbias = biases[vl_step * batch_size:(vl_step + 1) * batch_size]
                    loss_value_vl, acc_vl = sess.run([loss, accuracy],
                                                     feed_dict={
                                                         ftr_in: features[
                                                                 vl_step * batch_size:(vl_step + 1) * batch_size],
                                                         bias_in: bbias,
                                                         lbl_in: y_val[vl_step * batch_size:(vl_step + 1) * batch_size],
                                                         msk_in: val_mask[
                                                                 vl_step * batch_size:(vl_step + 1) * batch_size],
                                                         is_train: False,
                                                         attn_drop: 0.0, ffd_drop: 0.0})
                    val_loss_avg += loss_value_vl
                    val_acc_avg += acc_vl
                    vl_step += 1

                print(epoch, 'Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f' %
                      (train_loss_avg / tr_step, train_acc_avg / tr_step,
                       val_loss_avg / vl_step, val_acc_avg / vl_step))

                if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
                    if val_acc_avg / vl_step >= vacc_mx and val_loss_avg / vl_step <= vlss_mn:
                        vacc_early_model = val_acc_avg / vl_step
                        vlss_early_model = val_loss_avg / vl_step
                    vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
                    vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
                    curr_step = 0

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
                                attn_drop: 0.0, ffd_drop: 0.0})
                        ts_loss += loss_value_ts
                        ts_acc += acc_ts
                        ts_step += 1
                        ts_macro += f1_score(real_y[test_mask[0]], pred_y[test_mask[0]], average='macro')
                        ts_micro += f1_score(real_y[test_mask[0]], pred_y[test_mask[0]], average='micro')
                else:
                    curr_step += 1
                    if curr_step == patience:
                        print('Early stop! Min loss: ', vlss_mn, ', Max accuracy: ', vacc_mx)
                        print('Early stop model validation loss: ', vlss_early_model, ', accuracy: ', vacc_early_model)
                        break

                train_loss_avg = 0
                train_acc_avg = 0
                val_loss_avg = 0
                val_acc_avg = 0

            print('Test loss:', ts_loss / ts_step,
                  'acc:', ts_acc / ts_step,
                  'macro f1:', ts_macro / ts_step,
                  'micro f1:', ts_micro / ts_step)

            # Save test results
            results = {
                'dataset': args.dataset,
                'prefix': args.prefix,
                'lr': args.lr,
                'l2': args.l2,
                'units': args.units,
                'heads': args.heads,
                'dropout': args.dropout,
                'curvature': float(curvature_this),
                'test_loss': float(ts_loss / ts_step),
                'test_acc': float(ts_acc / ts_step),
                'test_macro_f1': float(ts_macro / ts_step),
                'test_micro_f1': float(ts_micro / ts_step)
            }

            # Create results directory if it doesn't exist
            os.makedirs('results', exist_ok=True)

            # Save results to file
            result_file = f'results/{args.dataset}_{args.prefix}_{time_begin}.json'
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=4)

            print(f"Results saved to {result_file}")

            sess.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)