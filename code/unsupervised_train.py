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
from data.minibatch import EdgeMinibatchLoader
from models.gain import SampleAndAggregate
from utils.metrics import evaluate_unsupervised, print_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="GAIN Unsupervised Training")
    
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
    parser.add_argument('--learning_rate', type=float, default=0.00002,
                        help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=1000,
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
    parser.add_argument('--dim_1', type=int, default=64,
                        help='Size of output dim in 1st layer')
    parser.add_argument('--dim_2', type=int, default=64,
                        help='Size of output dim in 2nd layer')
    parser.add_argument('--neg_sample_size', type=int, default=12,
                        help='Number of negative samples')
    
    # System settings
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--log_dir', type=str, default='../logs/unsupervised/',
                        help='Directory for logging')
    parser.add_argument('--validate_iter', type=int, default=500,
                        help='How often to run a validation batch')
    parser.add_argument('--print_every', type=int, default=50,
                        help='How often to print training info')
    parser.add_argument('--save_embeddings', action='store_true',
                        help='Whether to save embeddings after training')
    parser.add_argument('--save_embeddings_epoch', type=int, default=10,
                        help='How often to save embeddings')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    
    return args


def save_embeddings(model, dataset, device, output_dir, prefix=""):
    """
    Save node embeddings
    """
    model.eval()
    
    all_nodes = torch.LongTensor(list(range(len(dataset.id_map)))).to(device)
    
    # Process in batches to avoid OOM
    batch_size = 512
    embeddings = []
    
    for i in range(0, len(all_nodes), batch_size):
        batch_nodes = all_nodes[i:i+batch_size]
        
        with torch.no_grad():
            # Generate embeddings (use self as both inputs for unsupervised model)
            _, _, _, batch_embeds = model(batch_nodes, batch_nodes)
            embeddings.append(batch_embeds.cpu().numpy())
    
    # Combine batches
    all_embeddings = np.vstack(embeddings)
    
    # Save embeddings
    np.save(os.path.join(output_dir, f"{prefix}embeddings.npy"), all_embeddings)
    
    # Save mapping from node IDs to indices
    with open(os.path.join(output_dir, f"{prefix}embedding_idx.txt"), 'w') as f:
        for node, idx in dataset.id_map.items():
            f.write(f"{node}\t{idx}\n")


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
    Train an unsupervised GAIN model
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
    minibatch_loader = EdgeMinibatchLoader(
        dataset.get_adj_list(),
        context_pairs=dataset.get_walks() if args.random_context else None,
        batch_size=args.batch_size,
        max_degree=args.max_degree
    )
    
    # Get validation edges
    val_edges = minibatch_loader.get_validation_edges(args.validate_batch_size)
    val_edges = (val_edges[0].to(args.device), val_edges[1].to(args.device))
    
    # Create model
    features = dataset.get_features()
    
    # Layer info
    layer_infos = [
        {
            "num_samples": args.samples_1, 
            "output_dim": args.dim_1,
            "dropout": args.dropout
        },
        {
            "num_samples": args.samples_2, 
            "output_dim": args.dim_2,
            "dropout": args.dropout
        }
    ]
    
    model = SampleAndAggregate(
        features=features,
        adj=dataset.get_adj_list(),
        degrees=dataset.get_degrees(),
        layer_infos=layer_infos,
        concat=True,
        aggregator_type=args.model,
        model_size=args.model_size,
        identity_dim=args.identity_dim,
        neg_sample_size=args.neg_sample_size,
        device=args.device
    ).to(args.device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Training loop
    total_steps = 0
    best_val_mrr = 0
    
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        minibatch_loader.shuffle()
        
        epoch_loss = []
        
        with tqdm(total=minibatch_loader.num_batches, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for i in range(minibatch_loader.num_batches):
                # Get batch
                batch_node1, batch_node2 = minibatch_loader.next_batch()
                batch_node1 = batch_node1.to(args.device)
                batch_node2 = batch_node2.to(args.device)
                
                # Forward pass
                loss, _, _, _ = model(batch_node1, batch_node2)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss.append(loss.item())
                
                # Print stats
                if total_steps % args.print_every == 0:
                    metrics = {"loss": loss.item()}
                    print_metrics(metrics, epoch+1, prefix="Train", use_tqdm=True)
                
                # Validation
                if total_steps % args.validate_iter == 0:
                    val_metrics = evaluate_unsupervised(model, val_edges, args.device)
                    print_metrics(val_metrics, epoch+1, prefix="Val", use_tqdm=True)
                    
                    # Save best model
                    if val_metrics["mrr"] > best_val_mrr:
                        best_val_mrr = val_metrics["mrr"]
                        torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pt"))
                
                total_steps += 1
                pbar.update(1)
                pbar.set_postfix({"loss": np.mean(epoch_loss[-10:])})
        
        # Save embeddings periodically
        if args.save_embeddings and (epoch + 1) % args.save_embeddings_epoch == 0:
            save_embeddings(model, dataset, args.device, log_dir, prefix=f"epoch_{epoch+1}_")
    
    # Save final embeddings
    if args.save_embeddings:
        save_embeddings(model, dataset, args.device, log_dir, prefix="final_")
    
    print(f"Training complete. Best validation MRR: {best_val_mrr:.5f}")
    print(f"Model and embeddings saved to {log_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
