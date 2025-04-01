import os
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import json
from datetime import datetime

from data.dataset import GraphDataset
from data.minibatch import NodeMinibatchLoader
from models.supervised_models import SupervisedGraphsage
from utils.metrics import evaluate_supervised, print_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="GAIN Supervised Training")

    # Data settings
    parser.add_argument('--data_dir', type=str, default='../graph_data/osm_transductive/',
                        help='Base directory for graph data')
    parser.add_argument('--prefix', type=str, default='linkoping-osm',
                        help='Prefix for data files')
    parser.add_argument('--random_context', action='store_true',
                        help='Whether to use random context or direct edges')
    parser.add_argument('--walk_type', type=str, default='rand_edges',
                        choices=['rand_edges', 'rand_bfs_walks', 'rand_bfs_dfs_walks'],
                        help='Type of random walks')

    # Model settings
    parser.add_argument('--model', type=str, default='mean',
                        choices=['mean', 'gcn', 'maxpool', 'gain', 'gin', 'attn'],
                        help='Aggregator type')
    parser.add_argument('--model_size', type=str, default='small', choices=['small', 'big'],
                        help='Size of hidden layers')
    parser.add_argument('--identity_dim', type=int, default=0,
                        help='Set positive to use identity features')

    # Training settings
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Minibatch size')
    parser.add_argument('--validate_batch_size', type=int, default=256,
                        help='Batch size for validation')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight for L2 loss')
    parser.add_argument('--max_degree', type=int, default=100,
                        help='Maximum node degree')
    parser.add_argument('--samples_1', type=int, default=9,
                        help='Number of samples in 1st layer')
    parser.add_argument('--samples_2', type=int, default=3,
                        help='Number of samples in 2nd layer')
    parser.add_argument('--samples_3', type=int, default=0,
                        help='Number of samples in 3rd layer (0 to disable)')
    parser.add_argument('--dim_1', type=int, default=256,
                        help='Size of output dim in 1st layer')
    parser.add_argument('--dim_2', type=int, default=256,
                        help='Size of output dim in 2nd layer')
    parser.add_argument('--sigmoid', action='store_true',
                        help='Whether to use sigmoid loss (for multi-label)')

    # System settings
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--log_dir', type=str, default='../logs/supervised/',
                        help='Directory for logging')
    parser.add_argument('--validate_iter', type=int, default=10,
                        help='How often to run a validation batch')
    parser.add_argument('--print_every', type=int, default=5,
                        help='How often to print training info')

    args = parser.parse_args()

    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    return args


def save_config(args, output_dir):
    """
    Save configuration to a file
    """
    config = vars(args)
    config['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join(output_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=2)


def train(args):
    """
    Train a supervised GAIN model
    """
    print("Loading data...")
    dataset = GraphDataset(
        args.prefix,
        args.data_dir,
        normalize=True,
        walk_type=args.walk_type,
        dfs_num_len=[50, 10],  # Default values for DFS walks
        device=args.device
    )

    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.model}_{args.dim_1}_{args.learning_rate:.2e}"
    log_dir = os.path.join(args.log_dir, f"{args.prefix}_{timestamp}_{model_name}")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Save configuration
    save_config(args, log_dir)

    # Create minibatch loader
    minibatch_loader = NodeMinibatchLoader(
        dataset.get_adj_list(),
        dataset.class_map,
        dataset.train_nodes,
        batch_size=args.batch_size,
        max_degree=args.max_degree
    )

    # Get validation data
    val_nodes, val_labels = minibatch_loader.get_validation_data(dataset.val_nodes)
    val_nodes = val_nodes.to(args.device)
    val_labels = val_labels.to(args.device)

    # Get test data
    test_nodes, test_labels = minibatch_loader.get_validation_data(dataset.test_nodes)
    test_nodes = test_nodes.to(args.device)
    test_labels = test_labels.to(args.device)

    # Create model
    features = dataset.get_features()

    # Get number of classes
    if isinstance(list(dataset.class_map.values())[0], list):
        num_classes = len(list(dataset.class_map.values())[0])
    else:
        num_classes = len(set(dataset.class_map.values()))

    # Layer info
    layer_infos = []

    # First layer
    layer_infos.append({
        "num_samples": args.samples_1,
        "output_dim": args.dim_1,
        "dropout": args.dropout
    })

    # Second layer
    layer_infos.append({
        "num_samples": args.samples_2,
        "output_dim": args.dim_2,
        "dropout": args.dropout
    })

    # Third layer (optional)
    if args.samples_3 > 0:
        layer_infos.append({
            "num_samples": args.samples_3,
            "output_dim": args.dim_2,
            "dropout": args.dropout
        })

    model = SupervisedGraphsage(
        num_classes=num_classes,
        features=features,
        adj=dataset.get_adj_list(),
        degrees=dataset.get_degrees(),
        layer_infos=layer_infos,
        concat=True,
        aggregator_type=args.model,
        model_size=args.model_size,
        sigmoid_loss=args.sigmoid,
        identity_dim=args.identity_dim,
        dropout=args.dropout,
        device=args.device
    ).to(args.device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Training loop
    total_steps = 0
    best_val_f1 = 0

    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        minibatch_loader.shuffle()

        epoch_loss = []

        with tqdm(total=minibatch_loader.num_batches, desc=f"Epoch {epoch + 1}/{args.epochs}") as pbar:
            for i in range(minibatch_loader.num_batches):
                # Get batch
                batch_nodes, batch_labels = minibatch_loader.next_batch()
                batch_nodes = batch_nodes.to(args.device)
                batch_labels = batch_labels.to(args.device)

                # Forward pass
                logits = model(batch_nodes)
                loss = model.loss(logits, batch_labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss.append(loss.item())

                # Print stats
                if total_steps % args.print_every == 0:
                    metrics = {"loss": loss.item()}
                    print_metrics(metrics, epoch + 1, prefix="Train", use_tqdm=True)

                # Validation
                if total_steps % args.validate_iter == 0:
                    val_metrics = evaluate_supervised(model, val_nodes, val_labels, args.sigmoid, args.device)
                    print_metrics(val_metrics, epoch + 1, prefix="Val", use_tqdm=True)

                    # Save best model
                    if val_metrics["micro_f1"] > best_val_f1:
                        best_val_f1 = val_metrics["micro_f1"]
                        torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pt"))

                total_steps += 1
                pbar.update(1)
                pbar.set_postfix({"loss": np.mean(epoch_loss[-10:])})

        # Save model after each epoch
        torch.save(model.state_dict(), os.path.join(log_dir, f"model_epoch_{epoch + 1}.pt"))

    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(log_dir, "best_model.pt")))

    # Evaluate on test set
    test_metrics = evaluate_supervised(model, test_nodes, test_labels, args.sigmoid, args.device)
    print_metrics(test_metrics, prefix="Test")

    # Save test metrics
    with open(os.path.join(log_dir, "test_metrics.json"), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f"Training complete. Best validation micro F1: {best_val_f1:.5f}")
    print(f"Test micro F1: {test_metrics['micro_f1']:.5f}, macro F1: {test_metrics['macro_f1']:.5f}")
    print(f"Model saved to {log_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)