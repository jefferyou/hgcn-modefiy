import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from datetime import datetime
import random
from sklearn.preprocessing import OneHotEncoder
from shapely.geometry import LineString
import json
import os

"""
This script extracts road network data for Munich, Germany, centered at Marienplatz (48.1351, 11.5820),
converts it to a line graph representation, and prepares it for graph representation learning,
following the approach described in the GAIN paper.
"""


def get_params():
    """Define parameters for Munich dataset extraction"""
    PARAMS = {
        # dataset parameters
        'prefix': 'munich-osm',
        'poi': (48.1351, 11.5820),  # Marienplatz, Munich coordinates
        'buffer': 5000,  # 5km buffer around center point
        'geom_vector_len': 20,
        # 'exclude_edge_attributes': ['osmid', 'bridge', 'tunnel', 'width', 'ref', 'junction',
        #                             'access', 'lanes', 'oneway', 'name', 'key'],
        'exclude_edge_attributes': ['osmid', 'bridge', 'tunnel', 'width', 'ref', 'junction',
                                    'access', 'name', 'key'],
        'exclude_node_attributes': ['ref', 'osmid'],

        # Road type classification
        'label_lookup': {'motorway': 0,
                         'trunk': 0,  # merge for class balance
                         'primary': 0,  # merge for class balance
                         'secondary': 0,  # merge for class balance
                         'tertiary': 1,
                         'unclassified': 2,
                         'residential': 3,
                         'motorway_link': 0,  # merge for class balance
                         'trunk_link': 0,  # merge for class balance
                         'primary_link': 0,  # merge for class balance
                         'secondary_link': 0,  # merge for class balance
                         'tertiary_link': 1,  # merge for class balance
                         'living_street': 4,
                         'road': 5,
                         'yes': 0,
                         'planned': 5,
                         'pedestrian': 6,  # Adding pedestrian roads common in European cities
                         'service': 7,  # Adding service roads
                         'steps': 8,  # Adding steps
                         'path': 9,  # Adding paths
                         'footway': 6,  # Merging footways with pedestrian
                         'cycleway': 10,  # Munich has many bike paths
                         'track': 11,  # Including tracks
                         },

        # sampling parameters
        'sampling_seed': 42,
        'split_method': 'percentage',  # 'fixed' or 'percentage'
        'train_percent': 0.3,  # 30% for training
        'val_percent': 0.35,  # 35% for validation
        'test_percent': 0.35,  # 35% for testing
        'n_test': 800,  # Only used if split_method is 'fixed'
        'n_val': 400,  # Only used if split_method is 'fixed'

        # random walk parameters
        'walk_seed': 42,
        'walk_len': 5,
        'walk_num': 50
    }

    return PARAMS


def extract_osm_network():
    """Extract Munich road network using OSMnx"""
    PARAMS = get_params()
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S %z")

    print(f"Extracting road network for Munich centered at {PARAMS['poi']} with buffer {PARAMS['buffer']}m")

    # Retrieve osm data by center coordinate and spatial buffer
    # Use the newer API format if needed
    try:
        g = ox.graph_from_point(PARAMS['poi'], dist=PARAMS['buffer'], network_type='all', simplify=True)
    except TypeError:
        # Alternative approach for newer OSMnx versions
        center_point = PARAMS['poi']
        g = ox.graph.graph_from_point(center_point, dist=PARAMS['buffer'], network_type='all', simplify=True)

    # Try the projection with error handling
    try:
        g = ox.project_graph(g, to_crs="EPSG:32632")  # UTM zone 32N for Munich
    except:
        g = ox.projection.project_graph(g, to_crs="EPSG:32632")

    g.graph['osm_query_date'] = timestamp
    g.graph['name'] = PARAMS['prefix']
    g.graph['poi'] = PARAMS['poi']
    g.graph['buffer'] = PARAMS['buffer']

    # create incremental node ids
    g = nx.relabel.convert_node_labels_to_integers(g, first_label=0, ordering='default')

    # convert to undirected graph (i.e. directions and parallel edges are removed)
    g = nx.Graph(g.to_undirected())

    print(f"Extracted road network with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")

    return g, PARAMS


def sample_nodes(node_list, n_samples):
    """Sample nodes for train/val/test split"""
    samples = []
    np.random.shuffle(node_list)
    while len(node_list) != 0 and len(samples) < n_samples:
        samples.append(node_list.pop())

    return node_list, samples


def split_train_test_val_nodes(g, PARAMS):
    """Split nodes into train, validation, and test sets using fixed counts"""
    np.random.seed(PARAMS['sampling_seed'])
    remain_nodes, test_nodes = sample_nodes(list(g.nodes), PARAMS['n_test'])
    _, val_nodes = sample_nodes(remain_nodes, PARAMS['n_val'])

    test_dict = {}
    val_dict = {}

    for n in g.nodes:  # default
        test_dict[n] = False
        val_dict[n] = False
    for n in test_nodes:
        test_dict[n] = True
    for n in val_nodes:
        val_dict[n] = True

    nx.set_node_attributes(g, test_dict, 'test')
    nx.set_node_attributes(g, val_dict, 'val')

    print(
        f"Split nodes into {len(g.nodes) - len(test_nodes) - len(val_nodes)} train, {len(val_nodes)} validation, and {len(test_nodes)} test nodes")


def split_train_test_val_nodes_by_percentage(g, train_percent=0.3, val_percent=0.35, test_percent=0.35):
    """Split nodes into train, validation, and test sets by percentage"""
    np.random.seed(42)
    nodes = list(g.nodes)
    np.random.shuffle(nodes)

    n_nodes = len(nodes)
    n_test = int(n_nodes * test_percent)
    n_val = int(n_nodes * val_percent)

    test_nodes = nodes[:n_test]
    val_nodes = nodes[n_test:n_test + n_val]
    train_nodes = nodes[n_test + n_val:]

    # Set node attributes
    test_dict = {n: (n in test_nodes) for n in g.nodes}
    val_dict = {n: (n in val_nodes) for n in g.nodes}

    nx.set_node_attributes(g, test_dict, 'test')
    nx.set_node_attributes(g, val_dict, 'val')

    print(f"Split nodes into {len(train_nodes)} train ({len(train_nodes) / n_nodes:.1%}), "
          f"{len(val_nodes)} validation ({len(val_nodes) / n_nodes:.1%}), and "
          f"{len(test_nodes)} test ({len(test_nodes) / n_nodes:.1%}) nodes")


def convert_class_labels(g, PARAMS):
    """Convert highway types to class labels"""
    cnt = 0
    labels = nx.get_edge_attributes(g, 'highway')
    labels_int = {}
    for edge in g.edges:
        # set default attributes
        if edge not in labels:
            labels[edge] = 'road'

        # some edges have two attributes, take only their first
        if isinstance(labels[edge], list):
            labels[edge] = labels[edge][0]

        # handle attributes not in label lookup
        if labels[edge] not in PARAMS['label_lookup']:
            cnt += 1
            labels[edge] = 'road'

        labels_int[edge] = PARAMS['label_lookup'][labels[edge]]

    print(f"Found {cnt} edges with highway types not in lookup table, set to 'road'")
    nx.set_edge_attributes(g, labels_int, 'label')


def remove_unwanted_attributes(g, PARAMS):
    """Remove unwanted node and edge attributes"""
    # deleting some node attributes
    for n in g:
        for att in PARAMS['exclude_node_attributes']:
            if att in g.nodes[n]:
                g.nodes[n].pop(att, None)

    # deleting some edge attributes
    for n1, n2, d in g.edges(data=True):
        for att in PARAMS['exclude_edge_attributes']:
            if att in d:
                d.pop(att, None)


def standardize_geometries(g, PARAMS, attr_name='geom', verbose=1):
    """Standardize edge geometries to fixed length vectors"""
    steps = PARAMS['geom_vector_len']

    if verbose > 0:
        print(f'Generating fixed length ({steps}) geometry vectors...')

    geoms = nx.get_edge_attributes(g, 'geometry')
    xs = nx.get_node_attributes(g, 'x')
    ys = nx.get_node_attributes(g, 'y')
    np_same_length_geoms = {}
    count_no = 0
    count_yes = 0

    for e in g.edges():
        points = []

        if e not in geoms:  # edges that don't have a geometry
            line = LineString([(xs[e[0]], ys[e[0]]), (xs[e[1]], ys[e[1]])])
            for step in np.linspace(0, 1, steps):
                point = line.interpolate(step, normalized=True)
                points.append([point.x, point.y])
            count_no += 1

        else:  # all other edges
            for step in np.linspace(0, 1, steps):
                point = geoms[e].interpolate(step, normalized=True)
                points.append([point.x, point.y])
            count_yes += 1

        np_same_length_geoms[e] = np.array([np.array((p[0], p[1])) for p in points])

    if verbose > 0:
        print(f'- Geometry inserted from intersection coordinates for {count_no} edges')
        print(f'- Standardized geometry created for {count_no + count_yes} edges')

    nx.set_edge_attributes(g, np_same_length_geoms, attr_name)

    if verbose > 0:
        print('Done standardizing geometries')


def midpoint(p1, p2):
    """Calculate midpoint between two points"""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def midpoint_generation(g):
    """Generate midpoints for all edges"""
    pos = {}
    for u, d in g.nodes(data=True):
        pos[u] = (d['x'], d['y'])

    new_pos = {}
    for u, v, d in g.edges(data=True):
        e = (u, v)
        new_pos[e] = {'midpoint': np.array(midpoint(pos[u], pos[v]))}

    nx.set_edge_attributes(g, new_pos)


def midpoint_subtraction(g):
    """Subtract midpoint from geometry vectors for translation invariance"""
    for u, v, d in g.edges(data=True):
        e = (u, v)
        if 'geom' in d and 'midpoint' in d:
            d['geom'] = d['geom'] - d['midpoint']


def one_hot_encode_maxspeeds(g, verbose=1):
    """One-hot encode speed limits"""
    if verbose > 0:
        print('\nGenerating one-hot encoding of speed limits...')

    # Common speed limits in Germany (in km/h)
    maxspeeds_standard = ['5', '10', '20', '30', '50', '60', '70', '80', '100', '120', 'unknown']

    maxspeeds = nx.get_edge_attributes(g, 'maxspeed')
    maxspeeds_single_val = {}

    for e in g.edges():
        if e not in maxspeeds:
            maxspeeds[e] = 'unknown'

        if isinstance(maxspeeds[e], list):
            maxspeeds_single_val[e] = maxspeeds[e][0]
        else:
            maxspeeds_single_val[e] = maxspeeds[e]

    for e in maxspeeds_single_val:
        if maxspeeds_single_val[e] not in maxspeeds_standard:
            if maxspeeds_single_val[e].isdigit():
                maxspeeds_standard.append(maxspeeds_single_val[e])
            else:
                maxspeeds_single_val[e] = 'unknown'

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(np.array(maxspeeds_standard).reshape(-1, 1))

    if verbose > 0:
        print('- One-hot encoder fitted to data with following categories:')
        print('-', np.array(enc.categories_).flatten().tolist())

    maxspeeds_one_hot = {k: enc.transform(np.array(v).reshape(1, -1)).toarray().flatten().tolist() for k, v in
                         maxspeeds_single_val.items()}

    if verbose > 0:
        print('- One-hot encoded speed limits generated')

    nx.set_edge_attributes(g, maxspeeds_one_hot, 'maxspeed_one_hot')

    if verbose > 0:
        print('Done encoding speed limits')


def copy_edge_attributes_to_nodes(g, l, verbose=1):
    """Copy edge attributes from original graph to nodes of line graph"""
    if verbose > 0:
        print('Copying edge attributes to new line graph nodes...')

    node_attr = {}
    for u, v, d in g.edges(data=True):
        node_attr[(u, v)] = d

    nx.set_node_attributes(l, node_attr)


def convert_to_line_graph(g, verbose=1):
    """Convert original graph to line graph"""
    # print input graph summary
    if verbose > 0:
        print('\n---Original Graph---')
        print(nx.info(g))

    # make edges to nodes, create edges where common nodes existed
    if verbose > 0:
        print('\nConverting to line graph...')

    l = nx.line_graph(g)

    # copy graph attributes
    l.graph['name'] = g.graph['name'] + '_line'
    l.graph['osm_query_date'] = g.graph['osm_query_date']
    l.graph['poi'] = g.graph['poi']
    l.graph['buffer'] = g.graph['buffer']

    # copy edge attributes to new nodes
    copy_edge_attributes_to_nodes(g, l, verbose=verbose)

    # relabel new nodes, storing old id in attribute
    mapping = {}
    for n in l:
        mapping[n] = n

    nx.set_node_attributes(l, mapping, 'original_id')

    l = nx.relabel.convert_node_labels_to_integers(l, first_label=0, ordering='default')

    # print output graph summary
    if verbose > 0:
        print('\n---Converted Line Graph---')
        print(nx.info(l))
        print('Done line graph conversion')

    return l


def save_data(L, path, prefix):
    """Save processed graph data to files"""
    if not os.path.exists(path):
        os.makedirs(path)

    prefix = '/' + prefix

    # Remove unnecessary attributes before saving
    for n in L:
        for att in ['geometry', 'highway', 'maxspeed']:
            if att in L.nodes[n]:
                L.nodes[n].pop(att, None)

    # Convert numpy arrays to lists for JSON serialization
    for u, d in L.nodes(data=True):
        for key, val in d.items():
            if isinstance(val, np.ndarray):
                d[key] = val.flatten().tolist()

    for u, v, d in L.edges(data=True):
        for key, val in d.items():
            if isinstance(val, np.ndarray):
                d[key] = val.flatten().tolist()

    # --------------- Save ID-Map
    id_map = {}
    for n in L.nodes:
        id_map[str(n)] = n

    out_file = path + prefix + '-id_map.json'
    with open(out_file, 'w') as fp:
        json.dump(id_map, fp)
    print('ID-Map saved in', out_file)

    # --------------- Save Class-Maps
    class_map = {}
    for n in L.nodes:
        if 'label' in L.nodes[n]:
            class_map[str(n)] = L.nodes[n]['label']
        else:
            # Default to 'road' class if no label
            class_map[str(n)] = 13

    out_file = path + prefix + '-class_map.json'
    with open(out_file, 'w') as fp:
        json.dump(class_map, fp)
    print('Class-Map saved in', out_file)

    # --------------- Save Features
    data_arr = []
    for n, d in L.nodes(data=True):
        if 'midpoint' in d and 'maxspeed_one_hot' in d and 'geom' in d and 'length' in d:
            data_arr.append(np.hstack(
                ([
                    d['midpoint'],
                    np.array(d['maxspeed_one_hot']),
                    np.array(d['geom']),
                    d['length']
                ])))

    out_file = path + prefix + '-feats.npy'
    np.save(out_file, np.array(data_arr))
    print('Features saved in', out_file)

    # --------------- Save Graph
    for n in L:
        #for att in ['length', 'label', 'geom', 'midpoint', 'maxspeed_one_hot']:
        for att in ['label', 'geom']:
            if att in L.nodes[n]:
                L.nodes[n].pop(att, None)

    data = nx.node_link_data(L)
    out_file = path + prefix + '-G.json'
    with open(out_file, 'w') as fp:
        json.dump(data, fp)
    print('Graph saved in', out_file)

def save_original_graph(G, path, prefix):
    """保存原始图的JSON文件，处理CRS和numpy类型等不可序列化对象"""
    if not os.path.exists(path):
        os.makedirs(path)

    # 复制原图以避免修改原始数据
    G_copy = G.copy()

    # 移除图级别的CRS属性
    if 'crs' in G_copy.graph:
        del G_copy.graph['crs']
    # 处理图属性中的numpy类型
    for key in list(G_copy.graph.keys()):
        val = G_copy.graph[key]
        # 转换numpy数值类型为Python原生类型
        if isinstance(val, np.integer):
            G_copy.graph[key] = int(val)
        elif isinstance(val, np.floating):
            G_copy.graph[key] = float(val)
        elif isinstance(val, np.ndarray):
            G_copy.graph[key] = val.tolist()

    # 处理节点属性
    for u, d in G_copy.nodes(data=True):
        # 先处理节点ID（如果是numpy类型）
        if isinstance(u, np.integer):
            u = int(u)
        # 处理节点属性值
        for key in list(d.keys()):
            val = d[key]
            if isinstance(val, np.ndarray):
                d[key] = val.flatten().tolist()
            elif isinstance(val, np.integer):
                d[key] = int(val)  # 转换numpy.int64为int
            elif isinstance(val, np.floating):
                d[key] = float(val)
            elif key in ['geometry']:  # 移除几何对象
                del d[key]

    # 处理边属性
    for u, v, d in G_copy.edges(data=True):
        # 处理边的节点ID
        if isinstance(u, np.integer):
            u = int(u)
        if isinstance(v, np.integer):
            v = int(v)
        # 处理边属性值
        for key in list(d.keys()):
            val = d[key]
            if isinstance(val, np.ndarray):
                d[key] = val.flatten().tolist()
            elif isinstance(val, np.integer):
                d[key] = int(val)  # 转换numpy.int64为int
            elif isinstance(val, np.floating):
                d[key] = float(val)
            elif key in ['geometry']:
                del d[key]

    # 保存原始图的JSON文件
    data = nx.node_link_data(G_copy)
    out_file = os.path.join(path, f"{prefix}-original-G.json")
    with open(out_file, 'w') as fp:
        json.dump(data, fp)
    print('原始图保存于:', out_file)

def save_topological_pairs(G, path, PARAMS, bfs_walk=None, dfs_walk=None):
    """Save topological pairs for random walks (local and global neighborhood)"""
    WALK_LEN = PARAMS['walk_len']
    WALK_NUM = PARAMS['walk_num']
    prefix = '/' + PARAMS['prefix']

    # Extract training nodes
    nodes = [n for n in G.nodes() if not G.nodes[n]["val"] and not G.nodes[n]["test"]]
    G_sub = G.subgraph(nodes)

    # Run local random walks (BFS-like)
    if bfs_walk:
        pairs_bfs = []
        for count, node in enumerate(nodes):
            if G_sub.degree(node) == 0:
                continue
            for i in range(WALK_NUM):
                curr_node = node
                for j in range(WALK_LEN):
                    neighbors = list(G_sub.neighbors(curr_node))
                    if not neighbors:
                        break
                    next_node = random.choice(neighbors)
                    # self co-occurrences are useless
                    if curr_node != node:
                        pairs_bfs.append((node, curr_node))
                    curr_node = next_node
            if count % 1000 == 0:
                print("Done local walks for", count, "nodes")

        print('Number of local neighbors:', len(pairs_bfs))
        out_file = path + prefix + '-walks.txt'
        # Save into file
        with open(out_file, "w") as fp:
            fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs_bfs]))

    # Run global random walks (DFS-like)
    if dfs_walk:
        pairs_dfs = []
        for count, node in enumerate(nodes):
            if G_sub.degree(node) == 0:
                continue
            for i in range(WALK_NUM):
                curr_node = node
                prev_node = None
                next_node = None
                for j in range(2 * WALK_LEN):  # Longer walks for global neighborhoods
                    if prev_node is None:
                        neighbors = list(G_sub.neighbors(curr_node))
                        if not neighbors:
                            break
                        next_node = random.choice(neighbors)
                    else:
                        # Try to find neighbors that aren't also neighbors of current node
                        candidates = [neigh for neigh in G_sub.neighbors(next_node)
                                      if neigh not in G_sub.neighbors(curr_node)]
                        if not candidates:
                            # If no candidates, just pick any neighbor
                            candidates = list(G_sub.neighbors(next_node))
                            if not candidates:
                                break
                        prev_node = curr_node
                        curr_node = next_node
                        next_node = random.choice(candidates)

                if curr_node != node:
                    pairs_dfs.append((node, curr_node))

            if count % 1000 == 0:
                print("Done global walks for", count, "nodes")

        print('Number of global neighbors:', len(pairs_dfs))
        out_file = path + prefix + "-dfs-walks.txt"
        # Save into file
        with open(out_file, "w") as fp:
            fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs_dfs]))


def generate_graph(g, PARAMS, verbose=1):
    """Apply all graph transformations and feature generation steps"""
    # Convert class labels
    convert_class_labels(g, PARAMS)

    # Remove unwanted attributes
    remove_unwanted_attributes(g, PARAMS)

    # Standardize geometries
    standardize_geometries(g, PARAMS, verbose=verbose)

    # Generate midpoints
    midpoint_generation(g)

    # Subtract midpoints from geometries
    midpoint_subtraction(g)

    # One-hot encode speed limits
    one_hot_encode_maxspeeds(g, verbose=verbose)

    return g


def visualize_graph(G, L=None):
    """Visualize the original graph and optionally the line graph"""
    fig, ax = plt.subplots(figsize=(15, 15))

    # Get node positions from coordinates
    pos = {}
    for n, data in G.nodes(data=True):
        pos[n] = (data['x'], data['y'])

    # Get edge colors from road types
    edge_colors = []
    for u, v, data in G.edges(data=True):
        if 'label' in data:
            edge_colors.append(data['label'])
        else:
            edge_colors.append(0)

    # Plot original graph
    nx.draw_networkx(G, pos=pos, with_labels=False, node_size=15,
                     node_color='#333333', edge_color=edge_colors,
                     width=1.5, alpha=0.8, ax=ax)

    # Plot line graph if provided
    if L is not None:
        # Get midpoint positions for line graph nodes
        line_pos = {}
        for n, data in L.nodes(data=True):
            if 'midpoint' in data:
                line_pos[n] = tuple(data['midpoint'])

        # Plot line graph nodes at midpoints of original edges
        nx.draw_networkx_nodes(L, pos=line_pos, node_size=30,
                               node_color='red', alpha=0.5, ax=ax)

    plt.title('Munich Road Network')
    plt.axis('off')
    plt.tight_layout()

    # Save figure
    plt.savefig('munich_road_network.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to extract and process Munich road network"""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Settings
    output_path = '../munich_data'
    save_graphs = True
    save_walks = True
    visualize = True

    print("Extracting Munich road network data...")

    # Extract the road network from OSM
    G, PARAMS = extract_osm_network()

    # Process the graph
    G = generate_graph(G, PARAMS, verbose=1)

    # Convert to line graph
    L = convert_to_line_graph(G, verbose=1)

    # Split nodes for train/val/test based on specified method
    if PARAMS.get('split_method', 'fixed') == 'fixed':
        # Use fixed counts of nodes
        split_train_test_val_nodes(L, PARAMS)
    elif PARAMS.get('split_method') == 'percentage':
        # Use percentage-based split
        split_train_test_val_nodes_by_percentage(
            L,
            train_percent=PARAMS.get('train_percent', 0.3),
            val_percent=PARAMS.get('val_percent', 0.35),
            test_percent=PARAMS.get('test_percent', 0.35)
        )
    else:
        # Default to fixed counts
        split_train_test_val_nodes(L, PARAMS)

    # Visualize the graph
    if visualize:
        visualize_graph(G, L)

    # Save the processed data
    if save_graphs:
        save_data(L, output_path, PARAMS['prefix'])
        save_original_graph(G, output_path, PARAMS['prefix'])
    # Save topological pairs for random walks
    if save_walks:
        save_topological_pairs(L, output_path, PARAMS, bfs_walk=True, dfs_walk=True)

    print("Munich road network extraction and processing complete!")


if __name__ == "__main__":
    main()



