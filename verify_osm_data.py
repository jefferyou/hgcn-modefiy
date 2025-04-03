import os
import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def parse_args():
    parser = argparse.ArgumentParser(description='Verify OSM data for HAT training')
    parser.add_argument('-dataset', nargs='?', default='osm_transductive', help='osm_transductive or osm_inductive')
    parser.add_argument('-prefix', nargs='?', default='linkoping-osm', help='linkoping-osm or sweden-osm')
    parser.add_argument('-data_dir', default='graph_data', help='directory containing the datasets')
    return parser.parse_args()


def load_and_verify_data(args):
    """
    Load OSM data and verify its structure and properties
    """
    # Set data paths
    data_dir = os.path.join(args.data_dir, args.dataset)
    prefix = args.prefix

    print(f"\n===== Verifying {args.dataset}/{prefix} =====")

    # Check file existence
    required_files = [
        f"{prefix}-G.json",
        f"{prefix}-feats.npy",
        f"{prefix}-id_map.json",
        f"{prefix}-class_map.json"
    ]

    print("\nChecking required files:")
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  [OK] {file} ({file_size:.2f} MB)")
        else:
            print(f"  [MISSING] {file} - NOT FOUND")
            return

    # Load graph
    print("\nLoading graph data...")
    try:
        G_data = json.load(open(os.path.join(data_dir, f"{prefix}-G.json")))
        G = json_graph.node_link_graph(G_data)
        print(f"  [OK] Graph loaded successfully")
        print(f"  - Nodes: {G.number_of_nodes()}")
        print(f"  - Edges: {G.number_of_edges()}")

        # Check node attributes
        node_attrs = set()
        for _, attrs in G.nodes(data=True):
            node_attrs.update(attrs.keys())

        print(f"  - Node attributes: {', '.join(sorted(node_attrs))}")

        # Check if val/test attributes exist
        if 'val' in node_attrs and 'test' in node_attrs:
            val_count = sum(1 for _, attrs in G.nodes(data=True) if attrs.get('val', False))
            test_count = sum(1 for _, attrs in G.nodes(data=True) if attrs.get('test', False))
            train_count = G.number_of_nodes() - val_count - test_count

            print(f"  - Train nodes: {train_count} ({train_count / G.number_of_nodes():.2%})")
            print(f"  - Validation nodes: {val_count} ({val_count / G.number_of_nodes():.2%})")
            print(f"  - Test nodes: {test_count} ({test_count / G.number_of_nodes():.2%})")
        else:
            print("  [ERROR] Missing 'val' or 'test' node attributes")

    except Exception as e:
        print(f"  [ERROR] Error loading graph: {str(e)}")
        return

    # Load features
    print("\nLoading features...")
    try:
        features = np.load(os.path.join(data_dir, f"{prefix}-feats.npy"))
        print(f"  [OK] Features loaded successfully")
        print(f"  - Shape: {features.shape}")
        print(f"  - Mean: {np.mean(features):.4f}")
        print(f"  - Std: {np.std(features):.4f}")
        print(f"  - Min: {np.min(features):.4f}")
        print(f"  - Max: {np.max(features):.4f}")

        # Check if features match the number of nodes
        if features.shape[0] == G.number_of_nodes():
            print(f"  [OK] Feature count matches node count")
        else:
            print(f"  [ERROR] Feature count ({features.shape[0]}) doesn't match node count ({G.number_of_nodes()})")

    except Exception as e:
        print(f"  [ERROR] Error loading features: {str(e)}")
        return

    # Load ID map
    print("\nLoading ID map...")
    try:
        id_map = json.load(open(os.path.join(data_dir, f"{prefix}-id_map.json")))
        print(f"  [OK] ID map loaded successfully")
        print(f"  - Entries: {len(id_map)}")

        # Check a few entries
        print(f"  - Sample entries: ")
        for i, (k, v) in enumerate(id_map.items()):
            if i < 3:
                print(f"    {k}: {v}")
            else:
                break
    except Exception as e:
        print(f"  [ERROR] Error loading ID map: {str(e)}")
        return

    # Load class map
    print("\nLoading class map...")
    try:
        class_map = json.load(open(os.path.join(data_dir, f"{prefix}-class_map.json")))
        print(f"  [OK] Class map loaded successfully")
        print(f"  - Entries: {len(class_map)}")

        # Check class distribution
        if isinstance(list(class_map.values())[0], list):
            print(f"  - Multi-label classification detected")
            label_counts = {}
            for labels in class_map.values():
                for i, l in enumerate(labels):
                    if l > 0:
                        label_counts[i] = label_counts.get(i, 0) + 1

            print(f"  - Label distribution:")
            for label, count in sorted(label_counts.items()):
                print(f"    Class {label}: {count} ({count / len(class_map):.2%})")
        else:
            label_counts = {}
            for label in class_map.values():
                label_counts[label] = label_counts.get(label, 0) + 1

            print(f"  - Label distribution:")
            for label, count in sorted(label_counts.items()):
                print(f"    Class {label}: {count} ({count / len(class_map):.2%})")
    except Exception as e:
        print(f"  [ERROR] Error loading class map: {str(e)}")
        return

    # Visualize data
    print("\nGenerating visualizations...")

    # 1. Visualize feature space with PCA
    try:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)

        plt.figure(figsize=(10, 8))

        # Color by train/val/test
        train_mask = np.array(
            [not G.nodes[n].get('val', False) and not G.nodes[n].get('test', False) for n in G.nodes()])
        val_mask = np.array([G.nodes[n].get('val', False) for n in G.nodes()])
        test_mask = np.array([G.nodes[n].get('test', False) for n in G.nodes()])

        plt.scatter(features_2d[train_mask, 0], features_2d[train_mask, 1], alpha=0.5, s=10, label='Train')
        plt.scatter(features_2d[val_mask, 0], features_2d[val_mask, 1], alpha=0.5, s=10, label='Validation')
        plt.scatter(features_2d[test_mask, 0], features_2d[test_mask, 1], alpha=0.5, s=10, label='Test')

        plt.title(f'PCA visualization of node features: {args.dataset}/{prefix}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Create output directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig(f'visualizations/{args.dataset}_{prefix}_features_pca.png')
        print(f"  [OK] PCA visualization saved to visualizations/{args.dataset}_{prefix}_features_pca.png")

        # 2. Visualize network structure (small subset for better visibility)
        if G.number_of_nodes() > 1000:
            print("  - Graph too large for full visualization, sampling 1000 nodes...")
            # Sample nodes
            sampled_nodes = np.random.choice(list(G.nodes()), size=min(1000, G.number_of_nodes()), replace=False)
            G_sub = G.subgraph(sampled_nodes)
        else:
            G_sub = G

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G_sub, seed=42)

        node_colors = []
        for node in G_sub.nodes():
            if G_sub.nodes[node].get('test', False):
                node_colors.append('red')
            elif G_sub.nodes[node].get('val', False):
                node_colors.append('blue')
            else:
                node_colors.append('green')

        nx.draw_networkx(G_sub, pos=pos, node_size=30, width=0.5,
                         alpha=0.7, node_color=node_colors, with_labels=False)

        plt.title(f'Network structure: {args.dataset}/{prefix}')
        plt.axis('off')

        plt.savefig(f'visualizations/{args.dataset}_{prefix}_network.png')
        print(f"  [OK] Network visualization saved to visualizations/{args.dataset}_{prefix}_network.png")

    except Exception as e:
        print(f"  [ERROR] Error generating visualizations: {str(e)}")

    print("\n[OK] Data verification completed successfully!")


if __name__ == '__main__':
    args = parse_args()
    load_and_verify_data(args)