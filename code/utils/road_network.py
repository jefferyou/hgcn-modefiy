import os
import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from datetime import datetime
import random
from sklearn.preprocessing import OneHotEncoder
from shapely.geometry import LineString
import json


def serialize_graph(G):
    """
    将图转换为可以被JSON序列化的格式

    Args:
        G: NetworkX 图

    Returns:
        可JSON序列化的图
    """
    # 创建一个新图，不包含不可序列化的对象
    data = {'directed': G.is_directed(),
            'multigraph': G.is_multigraph(),
            'graph': {},
            'nodes': [],
            'links': []}

    # 添加图的属性
    for key, value in G.graph.items():
        if isinstance(value, (int, float, str, bool, list, dict, tuple)) or value is None:
            data['graph'][key] = value

    # 添加节点及其属性
    for node, node_attrs in G.nodes(data=True):
        node_data = {'id': node}
        for key, value in node_attrs.items():
            if isinstance(value, (int, float, str, bool, list, dict, tuple)) or value is None:
                node_data[key] = value
        data['nodes'].append(node_data)

    # 添加边及其属性
    for u, v, edge_attrs in G.edges(data=True):
        edge_data = {'source': u, 'target': v}
        for key, value in edge_attrs.items():
            if isinstance(value, (int, float, str, bool, list, dict, tuple)) or value is None:
                edge_data[key] = value
        data['links'].append(edge_data)

    return data











class RoadNetworkGenerator:
    """
    Class for generating road network graphs from OpenStreetMap data
    """

    def __init__(self, output_dir="../graph_data", is_transductive=True):
        """
        Args:
            output_dir: Directory to save the generated graphs
            is_transductive: Whether to generate transductive or inductive data
        """
        self.output_dir = output_dir
        self.is_transductive = is_transductive

        # Set directory based on dataset type
        if is_transductive:
            self.data_dir = os.path.join(output_dir, "osm_transductive")
        else:
            self.data_dir = os.path.join(output_dir, "osm_inductive")

        # Create directories if they don't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # Set parameters
        self.params = self._get_params()

    def _get_params(self):
        """
        Get parameters for graph generation
        """
        if self.is_transductive:
            # Transductive dataset parameters (single city)
            params = {
                # Dataset parameters
                'prefix': 'linkoping-osm',
                'poi': (58.408909, 15.618521),  # Linköping city center
                'buffer': 7000,  # 7km radius
                'geom_vector_len': 20,
                'exclude_edge_attributes': ['osmid', 'bridge', 'tunnel', 'width', 'ref', 'junction',
                                            'access', 'lanes', 'oneway', 'name', 'key'],
                'exclude_node_attributes': ['ref', 'osmid'],

                # Original road type labels
                'label_lookup': {'motorway': 0,
                                 'trunk': 0,
                                 'primary': 0,
                                 'secondary': 0,
                                 'tertiary': 4,
                                 'unclassified': 5,
                                 'residential': 6,
                                 'motorway_link': 0,
                                 'trunk_link': 0,
                                 'primary_link': 0,
                                 'secondary_link': 0,
                                 'tertiary_link': 4,
                                 'living_street': 12,
                                 'road': 13,
                                 'yes': 0,
                                 'planned': 13
                                 },

                # Sampling parameters
                'sampling_seed': 42,
                'n_test': 1000,
                'n_val': 500,

                # Random walk parameters
                'walk_seed': 42,
                'walk_len': 5,
                'walk_num': 50
            }
        else:
            # Inductive dataset parameters (multiple cities)
            params = {
                # Dataset parameters
                'prefix': 'sweden-osm',
                'places': {
                    'Uppsala': (59.857994, 17.638622),
                    'Västerås': (56.609789, 16.544657),
                    'Örebro': (59.274752, 15.214113),
                    'Linköping': (58.408909, 15.618521),
                    'Helsingborg': (56.046472, 12.695231),
                    'Jönköping': (57.782611, 14.162930),
                    'Norrköping': (58.586859, 16.193182),
                    'Lund': (55.703863, 13.191811),
                    'Umeå': (63.825855, 20.265303),
                    'Gävle': (60.674963, 17.141546),
                    'Borås': (57.721223, 12.939515),
                    'Södertälje': (59.194800, 17.626693),
                    'Eskilstuna': (59.370546, 16.509992),
                    'Halmstad': (56.673874, 12.863075),
                    'Växjö': (56.877798, 14.907140),
                    'Karlstad': (59.403223, 13.512568),
                    'Sundsvall': (62.392445, 17.305561)
                },
                'buffer': 7000,
                'geom_vector_len': 20,
                'exclude_edge_attributes': ['osmid', 'bridge', 'tunnel', 'width', 'ref', 'junction',
                                            'access', 'lanes', 'oneway', 'name', 'key'],
                'exclude_node_attributes': ['ref', 'osmid'],

                # Road type labels (merged for better class balance)
                'label_lookup': {'motorway': 0,
                                 'trunk': 0,
                                 'primary': 0,
                                 'secondary': 0,
                                 'tertiary': 4,
                                 'unclassified': 5,
                                 'residential': 6,
                                 'motorway_link': 0,
                                 'trunk_link': 0,
                                 'primary_link': 0,
                                 'secondary_link': 0,
                                 'tertiary_link': 4,
                                 'living_street': 12,
                                 'road': 5,
                                 'yes': 0,
                                 'planned': 5
                                 },

                # Sampling parameters
                'sampling_seed': 1337,
                'n_test': 2,  # Number of cities for test
                'n_val': 2,  # Number of cities for validation

                # Random walk parameters
                'walk_seed': 42,
                'walk_len': 5,
                'walk_num': 50
            }

        return params

    def extract_road_network(self):
        """
        Extract road network from OpenStreetMap

        Returns:
            G: Original network graph
            L: Line graph (edge-to-node transformed graph)
        """
        if self.is_transductive:
            G = self._extract_transductive_network()
        else:
            G = self._extract_inductive_network()

        # Process the graph
        G = self._process_graph(G)

        # Convert to line graph
        L = self._convert_to_line_graph(G)

        # Split into train/val/test sets
        if self.is_transductive:
            self._split_train_test_val_nodes(L)

        return G, L

    def _extract_transductive_network(self):
        """
        Extract road network for transductive setting (single city)
        """
        print("Extracting transductive road network...")
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S %z")

        # Retrieve OSM data by center coordinate and spatial buffer
        # 修改后
        G = ox.graph_from_point(self.params['poi'], dist=self.params['buffer'], network_type='drive', simplify=True,
                                multigraph=True)
        G = ox.project_graph(G, to_crs="EPSG:32633")

        # Add metadata
        G.graph['osm_query_date'] = timestamp
        G.graph['name'] = self.params['prefix']
        G.graph['poi'] = self.params['poi']
        G.graph['buffer'] = self.params['buffer']

        # Create incremental node IDs
        G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default')

        # Convert to undirected graph
        G = nx.Graph(G.to_undirected())

        return G

    def _extract_inductive_network(self):
        """
        Extract road network for inductive setting (multiple cities)
        """
        print("Extracting inductive road network...")
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S %z")

        # Split cities into train/val/test
        places, test_places, val_places = self._split_train_test_val_graphs()

        # Extract networks for each city
        sub_graphs = []

        # Training set
        print("Processing training cities...")
        for poi_name, poi_coord in places.items():
            print(f"Extracting road network for {poi_name} {poi_coord}")
            sub_G = ox.graph_from_point(poi_coord, dist=self.params['buffer'],
                                        network_type='drive', simplify=True, multigraph=True)

            # Mark nodes as not val/test
            nx.set_node_attributes(sub_G, False, 'test')
            nx.set_node_attributes(sub_G, False, 'val')
            nx.set_edge_attributes(sub_G, False, 'test')
            nx.set_edge_attributes(sub_G, False, 'val')

            sub_graphs.append(sub_G)

        # Test set
        print("Processing test cities...")
        for poi_name, poi_coord in test_places.items():
            print(f"Extracting road network for {poi_name} {poi_coord}")
            sub_G = ox.graph_from_point(poi_coord, dist=self.params['buffer'],
                                        network_type='drive', simplify=True, multigraph=True)

            # Mark nodes as test
            nx.set_node_attributes(sub_G, True, 'test')
            nx.set_node_attributes(sub_G, False, 'val')
            nx.set_edge_attributes(sub_G, True, 'test')
            nx.set_edge_attributes(sub_G, False, 'val')

            sub_graphs.append(sub_G)

        # Validation set
        print("Processing validation cities...")
        for poi_name, poi_coord in val_places.items():
            print(f"Extracting road network for {poi_name} {poi_coord}")
            sub_G = ox.graph_from_point(poi_coord, dist=self.params['buffer'],
                                        network_type='drive', simplify=True, multigraph=True)

            # Mark nodes as val
            nx.set_node_attributes(sub_G, False, 'test')
            nx.set_node_attributes(sub_G, True, 'val')
            nx.set_edge_attributes(sub_G, False, 'test')
            nx.set_edge_attributes(sub_G, True, 'val')

            sub_graphs.append(sub_G)

        # Combine all graphs
        G = nx.compose_all(sub_graphs)
        G = ox.project_graph(G, to_crs="EPSG:32633")

        # Add metadata
        G.graph['osm_query_date'] = timestamp
        G.graph['name'] = self.params['prefix']

        # Create incremental node IDs
        G = nx.relabel.convert_node_labels_to_integers(G, first_label=0, ordering='default')

        # Convert to undirected graph
        G = nx.Graph(G.to_undirected())

        return G

    def _split_train_test_val_graphs(self):
        """
        Split cities into train/val/test sets for inductive learning
        """
        places = self.params['places'].copy()

        print(f"Total cities: {len(places)}")

        # Set random seed
        np.random.seed(self.params['sampling_seed'])

        # Sample test cities
        test_places = {}
        test_keys = np.random.choice(list(places.keys()), size=self.params['n_test'], replace=False)
        for key in test_keys:
            test_places[key] = places[key]
            places.pop(key)

        # Sample validation cities
        val_places = {}
        val_keys = np.random.choice(list(places.keys()), size=self.params['n_val'], replace=False)
        for key in val_keys:
            val_places[key] = places[key]
            places.pop(key)

        print(f"Training cities: {len(places)}")
        print(f"Test cities: {len(test_places)}")
        print(f"Validation cities: {len(val_places)}")

        return places, test_places, val_places

    def _process_graph(self, G):
        """
        Process the graph: convert class labels, remove unwanted attributes,
        standardize geometries, generate features
        """
        print("Processing graph...")

        # Convert class labels
        self._convert_class_labels(G)

        # Remove unwanted attributes
        self._remove_unwanted_attributes(G)

        # Standardize geometries
        self._standardize_geometries(G)

        # Generate midpoint features
        self._generate_midpoints(G)

        # Subtract midpoint from geometry
        self._midpoint_subtraction(G)

        # Encode speed limits
        self._one_hot_encode_maxspeeds(G)

        return G

    def _convert_class_labels(self, G):
        """
        Convert road type labels to integers
        """
        print("Converting class labels...")

        cnt = 0
        labels = nx.get_edge_attributes(G, 'highway')
        labels_int = {}

        for edge in G.edges():
            # Set default attributes for edges without highway type
            if edge not in labels:
                labels[edge] = 'road'

            # Some edges have multiple attributes, take only the first
            if isinstance(labels[edge], list):
                labels[edge] = labels[edge][0]

            # Handle attributes not in label lookup
            if labels[edge] not in self.params['label_lookup']:
                cnt += 1
                labels[edge] = 'road'

            # Convert to integer labels
            labels_int[edge] = self.params['label_lookup'][labels[edge]]

        print(f"Added {cnt} new road labels")

        # Set edge attributes
        nx.set_edge_attributes(G, labels_int, 'label')

    def _remove_unwanted_attributes(self, G):
        """
        Remove unwanted node and edge attributes
        """
        print("Removing unwanted attributes...")

        # Remove node attributes
        for n in G:
            for att in self.params['exclude_node_attributes']:
                if att in G.nodes[n]:
                    G.nodes[n].pop(att)

        # Remove edge attributes
        for n1, n2, d in G.edges(data=True):
            for att in self.params['exclude_edge_attributes']:
                if att in d:
                    d.pop(att)

    def _standardize_geometries(self, G):
        """
        Create standardized geometry vectors of fixed length
        """
        print("Standardizing geometries...")

        steps = self.params['geom_vector_len']

        geoms = nx.get_edge_attributes(G, 'geometry')
        xs = nx.get_node_attributes(G, 'x')
        ys = nx.get_node_attributes(G, 'y')

        np_same_length_geoms = {}
        count_no_geom = 0
        count_with_geom = 0

        for e in G.edges():
            points = []

            if e not in geoms:
                # Edges without geometry - create straight line
                line = LineString([(xs[e[0]], ys[e[0]]), (xs[e[1]], ys[e[1]])])
                for step in np.linspace(0, 1, steps):
                    point = line.interpolate(step, normalized=True)
                    points.append([point.x, point.y])
                count_no_geom += 1
            else:
                # Edges with geometry
                for step in np.linspace(0, 1, steps):
                    point = geoms[e].interpolate(step, normalized=True)
                    points.append([point.x, point.y])
                count_with_geom += 1

            np_same_length_geoms[e] = np.array([np.array((p[0], p[1])) for p in points])

        print(f"Geometry inserted from coordinates for {count_no_geom} edges")
        print(f"Standardized geometry created for {count_no_geom + count_with_geom} edges")

        # Set edge attributes
        nx.set_edge_attributes(G, np_same_length_geoms, 'geom')

    def _generate_midpoints(self, G):
        """
        Generate midpoint coordinates for each edge
        """
        print("Generating midpoints...")

        pos = {}
        for u, d in G.nodes(data=True):
            pos[u] = (d['x'], d['y'])

        new_pos = {}
        for u, v, d in G.edges(data=True):
            # Calculate midpoint
            mid_x = (pos[u][0] + pos[v][0]) / 2
            mid_y = (pos[u][1] + pos[v][1]) / 2

            # Store as attribute
            new_pos[(u, v)] = {'midpoint': np.array([mid_x, mid_y])}

        # Set edge attributes
        nx.set_edge_attributes(G, new_pos)

    def _midpoint_subtraction(self, G):
        """
        Subtract midpoint from geometry to center features
        """
        print("Subtracting midpoints from geometries...")

        for u, v, d in G.edges(data=True):
            d['geom'] = d['geom'] - d['midpoint']

    def _one_hot_encode_maxspeeds(self, G):
        """
        Create one-hot encoding for speed limits
        """
        print("One-hot encoding speed limits...")

        # Standard speed limits
        maxspeeds_standard = ['5', '7', '10', '20', '30', '40', '50', '60',
                              '70', '80', '90', '100', '110', '120', 'unknown']

        # Get speed limits
        maxspeeds = nx.get_edge_attributes(G, 'maxspeed')
        maxspeeds_single_val = {}

        for e in G.edges():
            # Set default value
            if e not in maxspeeds:
                maxspeeds[e] = 'unknown'

            # Handle lists of values
            if isinstance(maxspeeds[e], list):
                maxspeeds_single_val[e] = maxspeeds[e][0]
            else:
                maxspeeds_single_val[e] = maxspeeds[e]

            # Add to standard list if numeric and not already included
            if maxspeeds_single_val[e] not in maxspeeds_standard:
                if maxspeeds_single_val[e].isdigit():
                    maxspeeds_standard.append(maxspeeds_single_val[e])
                else:
                    maxspeeds_single_val[e] = 'unknown'

        # Create encoder
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(np.array(maxspeeds_standard).reshape(-1, 1))

        # Apply encoding
        maxspeeds_one_hot = {}
        for k, v in maxspeeds_single_val.items():
            encoded = enc.transform(np.array(v).reshape(1, -1)).toarray().flatten().tolist()
            maxspeeds_one_hot[k] = encoded

        # Set edge attributes
        nx.set_edge_attributes(G, maxspeeds_one_hot, 'maxspeed_one_hot')

        print(f"Speed limits encoded with {len(maxspeeds_standard)} categories")

    def _convert_to_line_graph(self, G):
        """
        Convert original graph to line graph (edge-to-node transformation)
        """
        print("Converting to line graph...")

        # Create line graph
        L = nx.line_graph(G)

        # Copy graph attributes
        L.graph['name'] = G.graph['name'] + '_line'
        L.graph['osm_query_date'] = G.graph['osm_query_date']

        # Copy edge attributes to new nodes
        node_attr = {}
        for u, v, d in G.edges(data=True):
            node_attr[(u, v)] = d

        nx.set_node_attributes(L, node_attr)

        # Relabel nodes to integers
        mapping = {}
        for n in L:
            mapping[n] = n

        nx.set_node_attributes(L, mapping, 'original_id')
        L = nx.relabel.convert_node_labels_to_integers(L, first_label=0, ordering='default')

        print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"Line graph: {L.number_of_nodes()} nodes, {L.number_of_edges()} edges")

        return L

    def _split_train_test_val_nodes(self, L):
        """
        Split nodes into train/val/test sets for transductive learning
        """
        print("Splitting into train/val/test sets...")

        np.random.seed(self.params['sampling_seed'])

        # Get all nodes
        all_nodes = list(L.nodes())
        np.random.shuffle(all_nodes)

        # Sample test nodes
        test_nodes = all_nodes[:self.params['n_test']]

        # Sample validation nodes
        val_nodes = all_nodes[self.params['n_test']:self.params['n_test'] + self.params['n_val']]

        # Create attribute dictionaries
        test_dict = {}
        val_dict = {}

        for n in L.nodes():
            # Default values
            test_dict[n] = False
            val_dict[n] = False

        # Set test and val nodes
        for n in test_nodes:
            test_dict[n] = True
        for n in val_nodes:
            val_dict[n] = True

        # Set node attributes
        nx.set_node_attributes(L, test_dict, 'test')
        nx.set_node_attributes(L, val_dict, 'val')

        print(f"Train: {len(all_nodes) - len(test_nodes) - len(val_nodes)} nodes")
        print(f"Test: {len(test_nodes)} nodes")
        print(f"Validation: {len(val_nodes)} nodes")

    def save_graph_data(self, L):
        """
        Save processed graph data to files

        Args:
            L: Line graph to save
        """
        print(f"Saving graph data to {self.data_dir}...")

        # Create output directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        prefix = self.params['prefix']

        # Save ID map
        id_map = {}
        for n in L.nodes():
            id_map[str(n)] = n

        with open(os.path.join(self.data_dir, f"{prefix}-id_map.json"), 'w') as f:
            json.dump(id_map, f)
        print(f"ID map saved to {prefix}-id_map.json")

        # Save class map
        class_map = {}
        for n in L.nodes():
            class_map[str(n)] = np.array(L.nodes[n]['label']).astype(int).tolist()

        with open(os.path.join(self.data_dir, f"{prefix}-class_map.json"), 'w') as f:
            json.dump(class_map, f)
        print(f"Class map saved to {prefix}-class_map.json")

        # Save features
        print("Extracting feature vectors...")
        data_arr = []
        for n, d in L.nodes(data=True):
            # 确保所有特征都是一维数组
            midpoint = d['midpoint'].flatten() if hasattr(d['midpoint'], 'flatten') else d['midpoint']
            maxspeed_one_hot = np.array(d['maxspeed_one_hot']).flatten()
            geom = np.array(d['geom']).flatten() if hasattr(d['geom'], 'flatten') else d['geom'].reshape(-1)
            length = np.array([d['length']]) if 'length' in d else np.array([0.0])

            # 水平堆叠所有特征
            feature_vector = np.hstack([midpoint, maxspeed_one_hot, geom, length])
            data_arr.append(feature_vector)

        np.save(os.path.join(self.data_dir, f"{prefix}-feats.npy"), np.array(data_arr))
        print(f"Features saved to {prefix}-feats.npy")

        # 创建一个不包含不可序列化属性的图副本
        L_copy = L.copy()

        # 移除不可序列化的属性
        for n in L_copy:
            for att in list(L_copy.nodes[n].keys()):
                value = L_copy.nodes[n][att]
                # 检查是否可序列化
                if not isinstance(value, (int, float, str, bool, list, dict, tuple)) or isinstance(value,
                                                                                                   (complex, set)):
                    L_copy.nodes[n].pop(att)

        # 将图转换为可序列化格式
        graph_data = serialize_graph(L_copy)

        # 保存图
        with open(os.path.join(self.data_dir, f"{prefix}-G.json"), 'w') as f:
            json.dump(graph_data, f)
        print(f"Graph saved to {prefix}-G.json")

    def generate_random_walks(self, L):
        """
        Generate random walks for the graph

        Args:
            L: Line graph
        """
        print("Generating random walks...")

        # Set random seed
        np.random.seed(self.params['walk_seed'])
        random.seed(self.params['walk_seed'])

        prefix = self.params['prefix']

        # Extract training nodes
        train_nodes = [n for n in L.nodes() if not L.nodes[n]['val'] and not L.nodes[n]['test']]
        L_train = L.subgraph(train_nodes)

        # Generate BFS walks (local neighborhood)
        bfs_pairs = self._run_random_walks(
            L_train, train_nodes,
            walk_len=self.params['walk_len'],
            num_walks=self.params['walk_num']
        )

        # Save BFS walks
        bfs_file = os.path.join(self.data_dir, f"{prefix}-walks.txt")
        with open(bfs_file, "w") as f:
            f.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in bfs_pairs]))
        print(f"BFS walks saved to {prefix}-walks.txt")

        # Generate DFS walks (global neighborhood)
        dfs_pairs = self._run_dfs_walks(
            L_train, train_nodes,
            dfs_len=2 * self.params['walk_len'],
            num_walks=self.params['walk_num']
        )

        # Save DFS walks
        dfs_file = os.path.join(self.data_dir, f"{prefix}-dfs-walks.txt")
        with open(dfs_file, "w") as f:
            f.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in dfs_pairs]))
        print(f"DFS walks saved to {prefix}-dfs-walks.txt")

    def _run_random_walks(self, G, nodes, walk_len=5, num_walks=50):
        """
        Run random walks on the graph

        Args:
            G: Graph to run walks on
            nodes: Starting nodes for walks
            walk_len: Length of each walk
            num_walks: Number of walks per node

        Returns:
            List of co-occurring node pairs
        """
        pairs = []

        for count, node in enumerate(nodes):
            # Skip isolated nodes
            if G.degree(node) == 0:
                continue

            for i in range(num_walks):
                curr_node = node

                for j in range(walk_len):
                    # Get neighbors and select one randomly
                    neighbors = list(G.neighbors(curr_node))
                    next_node = random.choice(neighbors)

                    # Add node pair if not self
                    if curr_node != node:
                        pairs.append((node, curr_node))

                    curr_node = next_node

            if count % 1000 == 0:
                print(f"Completed walks for {count} nodes")

        print(f"Generated {len(pairs)} local neighborhood pairs")
        return pairs

    def _run_dfs_walks(self, G, nodes, dfs_len=10, num_walks=50):
        """
        Run deeper DFS walks on the graph

        Args:
            G: Graph to run walks on
            nodes: Starting nodes for walks
            dfs_len: Length of each DFS walk
            num_walks: Number of walks per node

        Returns:
            List of co-occurring node pairs
        """
        dfs_pairs = []

        for count, node in enumerate(nodes):
            # Skip isolated nodes
            if G.degree(node) == 0:
                continue

            for i in range(num_walks):
                curr_node = node
                prev_node = None
                next_node = None

                for j in range(dfs_len):
                    # First step: choose random neighbor
                    if prev_node is None:
                        depth_1_neighbors = list(G.neighbors(curr_node))
                        if not depth_1_neighbors:  # Skip if no neighbors
                            break
                        next_node = random.choice(depth_1_neighbors)

                    # Find neighbors of next_node that aren't neighbors of curr_node
                    curr_node_neighbors = set(G.neighbors(curr_node))
                    depth_2_neighbors = [neigh for neigh in G.neighbors(next_node)
                                         if neigh not in curr_node_neighbors]

                    # If no such neighbors, break the walk
                    if not depth_2_neighbors:
                        break

                    # Update nodes
                    prev_node = curr_node
                    curr_node = next_node
                    next_node = random.choice(depth_2_neighbors)

                # Add pair if not self
                if curr_node != node:
                    dfs_pairs.append((node, curr_node))

            if count % 1000 == 0:
                print(f"Completed DFS walks for {count} nodes")

        print(f"Generated {len(dfs_pairs)} global neighborhood pairs")
        return dfs_pairs

    def generate_dataset(self):
        """
        Main function to generate dataset
        """
        # Extract road network
        G, L = self.extract_road_network()

        # Save graph data
        self.save_graph_data(L)

        # Generate random walks
        self.generate_random_walks(L)

        print(f"Dataset generation complete. Files saved to {self.data_dir}")

        return G, L


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate road network graph dataset')
    parser.add_argument('--output_dir', type=str, default='../graph_data',
                        help='Directory to save the generated graphs')
    parser.add_argument('--transductive', action='store_true', default=True,
                        help='Generate transductive dataset (single city)')
    parser.add_argument('--inductive', action='store_true',
                        help='Generate inductive dataset (multiple cities)')

    args = parser.parse_args()

    # Set dataset type
    is_transductive = not args.inductive

    # Create generator
    generator = RoadNetworkGenerator(args.output_dir, is_transductive)

    # Generate dataset
    G, L = generator.generate_dataset()

    print("Dataset generation complete!")


if __name__ == "__main__":
    main()