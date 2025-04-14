import os
import time
import datetime
import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from datetime import datetime
import random
import collections
from sklearn.preprocessing import OneHotEncoder
from shapely.geometry import LineString
import json

# 设置随机种子以确保结果可重现
np.random.seed(42)
random.seed(42)

# 定义目录结构
BASE_DIR = 'italian_graph_data'
INDUCTIVE_DIR = os.path.join(BASE_DIR, 'osm_inductive')

# 如果目录不存在则创建
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
if not os.path.exists(INDUCTIVE_DIR):
    os.makedirs(INDUCTIVE_DIR)


def get_params_inductive():
    """
    定义意大利城市及其参数
    """
    # 选择17个意大利城市及其坐标
    # 这些城市大多具有历史中心和放射状/非网格状的道路网络
    PLACES = {
        'Roma': (41.8933, 12.4829),  # 罗马 - 历史中心呈辐射状
        'Firenze': (43.7696, 11.2558),  # 佛罗伦萨 - 历史中心结构
        'Siena': (43.3186, 11.3305),  # 锡耶纳 - 山城，高度非线性路网
        'Genova': (44.4056, 8.9463),  # 热那亚 - 受山地和海岸线限制
        'Napoli': (40.8518, 14.2681),  # 那不勒斯 - 非规则道路网络
        'Perugia': (43.1122, 12.3888),  # 佩鲁贾 - 山顶城市，高度非线性
        'Assisi': (43.0707, 12.6191),  # 阿西西 - 山城
        'Verona': (45.4384, 10.9916),  # 维罗纳 - 历史中心
        'Lucca': (43.8429, 10.5027),  # 卢卡 - 历史城墙城市
        'Matera': (40.6669, 16.6106),  # 马泰拉 - "石头城"，独特地形
        'Orvieto': (42.7185, 12.1098),  # 奥尔维耶托 - 山顶城市
        'San Gimignano': (43.4677, 11.0431),  # 圣吉米尼亚诺 - 小型中世纪城镇
        'Urbino': (43.7262, 12.6365),  # 乌尔比诺 - 山城
        'Pisa': (43.7229, 10.4017),  # 比萨 - 历史城市
        'Bologna': (44.4949, 11.3426),  # 博洛尼亚 - 历史城市
        'Ravenna': (44.4183, 12.2035),  # 拉韦纳 - 历史城市
        'Bergamo': (45.6983, 9.6773),  # 贝加莫 - 上城区和下城区结构
    }

    PARAMS = {
        # 数据集参数
        'prefix': 'italy-osm',
        'places': PLACES,
        'buffer': 5000,  # 城市中心点周围5000米半径内的道路网络
        'geom_vector_len': 20,  # 每个路段的几何形状采样点数
        'exclude_edge_attributes': ['osmid', 'bridge', 'tunnel', 'width', 'ref', 'junction',
                                    'access', 'lanes', 'oneway', 'name', 'key'],
        'exclude_node_attributes': ['ref', 'osmid'],

        # 道路类型标签映射
        'label_lookup': {
            'motorway': 0,
            'trunk': 0,  # 与主干道合并
            'primary': 0,  # 与主干道合并
            'secondary': 0,  # 与主干道合并
            'tertiary': 1,
            'unclassified': 2,
            'residential': 3,
            'motorway_link': 0,  # 与主干道合并
            'trunk_link': 0,  # 与主干道合并
            'primary_link': 0,  # 与主干道合并
            'secondary_link': 0,  # 与主干道合并
            'tertiary_link': 1,  # 与三级道路合并
            'living_street': 4,
            'road': 2,  # 与未分类道路合并
            'pedestrian': 4,  # 与生活街区合并
            'service': 2,  # 与未分类道路合并
            'yes': 0,  # 默认为主干道
            'planned': 2  # 与未分类道路合并
        },

        # 采样参数
        'sampling_seed': 42,
        'n_test': 2,  # 分配2个城市作为测试集
        'n_val': 2,  # 分配2个城市作为验证集

        # 随机游走参数
        'walk_seed': 42,
        'walk_len': 5,
        'walk_num': 50
    }

    return PARAMS


def split_train_test_val_graphs(PARAMS):
    """
    将城市分割为训练、测试和验证集
    """
    places = PARAMS['places']
    print('Total set size:', len(places))

    random.seed(PARAMS['sampling_seed'])
    test_places = random.sample(list(places.items()), PARAMS['n_test'])
    for place in test_places:
        del places[place[0]]

    val_places = random.sample(list(places.items()), PARAMS['n_val'])
    for place in val_places:
        del places[place[0]]

    print('Training set size:', len(places))
    print('Validation set size:', len(test_places))
    print('Test set size:', len(val_places))
    return places, dict(test_places), dict(val_places)


def extract_osm_network_inductive():
    """
    提取意大利城市的道路网络并合并为一个大图
    """
    PARAMS = get_params_inductive()
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S %z")

    places, test_places, val_places = split_train_test_val_graphs(PARAMS)

    sub_Gs = []

    # 处理训练集城市
    print('Training set')
    for poi in places:
        print(f'Extracting road network for {poi} {places[poi]}')
        try:
            # 获取原始图并直接投影
            sub_G = ox.graph_from_point(places[poi], dist=PARAMS['buffer'],
                                        network_type='drive', simplify=True)
            # 显式投影到目标坐标系
            sub_G = ox.project_graph(sub_G, to_crs="EPSG:32633")
            # 转换为NetworkX无向图
            sub_G = nx.Graph(sub_G.to_undirected())
            print(nx.info(sub_G), '\n')
            nx.set_node_attributes(sub_G, False, 'test')
            nx.set_node_attributes(sub_G, False, 'val')
            nx.set_edge_attributes(sub_G, False, 'test')
            nx.set_edge_attributes(sub_G, False, 'val')
            sub_Gs.append(sub_G)
        except Exception as e:
            print(f"Error extracting {poi}: {e}")
            continue

    # 处理测试集城市
    print('Test set')
    for poi in test_places:
        print(f'Extracting road network for {poi} {test_places[poi]}')
        try:
            # 获取原始图并直接投影
            sub_G = ox.graph_from_point(test_places[poi], dist=PARAMS['buffer'],
                                        network_type='drive', simplify=True)
            # 显式投影到目标坐标系
            sub_G = ox.project_graph(sub_G, to_crs="EPSG:32633")
            # 转换为NetworkX无向图
            sub_G = nx.Graph(sub_G.to_undirected())
            print(nx.info(sub_G), '\n')
            nx.set_node_attributes(sub_G, True, 'test')
            nx.set_node_attributes(sub_G, False, 'val')
            nx.set_edge_attributes(sub_G, True, 'test')
            nx.set_edge_attributes(sub_G, False, 'val')
            sub_Gs.append(sub_G)
        except Exception as e:
            print(f"Error extracting {poi}: {e}")
            continue

    # 处理验证集城市
    print('Validation set')
    for poi in val_places:
        print(f'Extracting road network for {poi} {val_places[poi]}')
        try:
            # 获取原始图并直接投影
            sub_G = ox.graph_from_point(val_places[poi], dist=PARAMS['buffer'],
                                        network_type='drive', simplify=True)
            # 显式投影到目标坐标系
            sub_G = ox.project_graph(sub_G, to_crs="EPSG:32633")
            # 转换为NetworkX无向图
            sub_G = nx.Graph(sub_G.to_undirected())
            print(nx.info(sub_G), '\n')
            nx.set_node_attributes(sub_G, False, 'test')
            nx.set_node_attributes(sub_G, True, 'val')
            nx.set_edge_attributes(sub_G, False, 'test')
            nx.set_edge_attributes(sub_G, True, 'val')
            sub_Gs.append(sub_G)
        except Exception as e:
            print(f"Error extracting {poi}: {e}")
            continue

    # 将所有城市图合并为一个大图
    g = nx.compose_all(sub_Gs)

    # 图属性
    g.graph['osm_query_date'] = timestamp
    g.graph['name'] = PARAMS['prefix']

    # 创建顺序节点ID
    g = nx.relabel.convert_node_labels_to_integers(g, first_label=0, ordering='default')

    # 确保是无向图
    g = nx.Graph(g.to_undirected())

    return g, PARAMS


def convert_class_labels(g, PARAMS):
    """
    转换OSM道路类型标签为标准类别
    """
    cnt = 0
    labels = nx.get_edge_attributes(g, 'highway')
    labels_int = {}
    for edge in g.edges:
        # 设置默认属性
        if not edge in labels:
            labels[edge] = 'road'

        # 一些边可能有多个属性，只取第一个
        if type(labels[edge]) == list:
            labels[edge] = labels[edge][0]

        # 一些边的属性可能不在我们的标签映射中
        if not labels[edge] in PARAMS['label_lookup']:
            cnt += 1
            labels[edge] = 'road'

    print('Number of newly added road labels by OSM:', cnt)

    # 将字符串标签转换为整数类别
    for edge in g.edges:
        labels_int[edge] = PARAMS['label_lookup'][labels[edge]]

    nx.set_edge_attributes(g, labels_int, 'label')


def remove_unwanted_attributes(g, PARAMS):
    """
    删除不需要的节点和边属性
    """
    # 删除节点属性
    for n in g:
        for att in PARAMS['exclude_node_attributes']:
            if att in g.nodes[n]:
                g.nodes[n].pop(att, None)

    # 删除边属性
    for n1, n2, d in g.edges(data=True):
        for att in PARAMS['exclude_edge_attributes']:
            if att in d:
                d.pop(att, None)


def standardize_geometries(g, PARAMS, attr_name='geom', verbose=1):
    """
    将道路几何形状标准化为固定长度的向量
    """
    steps = PARAMS['geom_vector_len']

    if verbose > 0:
        print(f'\nGenerating fixed length ({steps}) geometry vectors...')

    geoms = nx.get_edge_attributes(g, 'geometry')
    xs = nx.get_node_attributes(g, 'x')
    ys = nx.get_node_attributes(g, 'y')
    np_same_length_geoms = {}
    count_no = 0
    count_yes = 0

    for e in g.edges():
        points = []

        if e not in geoms:  # 没有几何形状的边
            line = LineString([(xs[e[0]], ys[e[0]]), (xs[e[1]], ys[e[1]])])
            for step in np.linspace(0, 1, steps):
                point = line.interpolate(step, normalized=True)
                points.append([point.x, point.y])
            count_no += 1

        else:  # 所有其他边
            for step in np.linspace(0, 1, steps):
                point = geoms[e].interpolate(step, normalized=True)
                points.append([point.x, point.y])
            count_yes += 1

        np_same_length_geoms[e] = np.array([np.array((p[0], p[1])) for p in points])

    if verbose > 0:
        print(f'- Geometry inserted from intersection coordinates for {count_no} nodes.')
        print(f'- Standardized geometry created for {count_no + count_yes} nodes.')

    nx.set_edge_attributes(g, np_same_length_geoms, attr_name)

    if verbose > 0:
        print('Done.')


def midpoint(p1, p2):
    """
    计算两点的中点
    """
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def midpoint_generation(g):
    """
    为每条边生成中点坐标
    """
    pos = {}
    for u, d in g.nodes(data=True):
        pos[u] = (d['x'], d['y'])

    new_pos = {}
    for u, v, d in g.edges(data=True):
        e = (u, v)
        new_pos[e] = {'midpoint': np.array(midpoint(pos[u], pos[v]))}

    nx.set_edge_attributes(g, new_pos)


def midpoint_subtraction(g):
    """
    从几何形状向量中减去中点坐标
    """
    for u, v, d in g.edges(data=True):
        e = (u, v)
        if 'geom' in d and 'midpoint' in d:
            d['geom'] = d['geom'] - d['midpoint']


def one_hot_encode_maxspeeds(g, verbose=1):
    """
    将道路速度限制转换为独热编码
    """
    if verbose > 0:
        print('\nGenerating one-hot encoding maxspeed limits...')

    # 意大利常见的速度限制（单位：km/h）
    maxspeeds_standard = ['5', '10', '20', '30', '40', '50', '60',
                          '70', '80', '90', '100', '110', '130', 'unknown']

    maxspeeds = nx.get_edge_attributes(g, 'maxspeed')
    maxspeeds_single_val = {}

    for e in g.edges():
        if e not in maxspeeds:
            maxspeeds[e] = 'unknown'

        if type(maxspeeds[e]) == list:
            maxspeeds_single_val[e] = maxspeeds[e][0]
        else:
            maxspeeds_single_val[e] = maxspeeds[e]

    # 处理不在标准列表中的速度限制
    for e in maxspeeds_single_val:
        if maxspeeds_single_val[e] not in maxspeeds_standard:
            if maxspeeds_single_val[e].isdigit():
                maxspeeds_standard.append(maxspeeds_single_val[e])
            else:
                maxspeeds_single_val[e] = 'unknown'

    # 创建独热编码器
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(np.array(maxspeeds_standard).reshape(-1, 1))

    if verbose > 0:
        print('- One-hot encoder fitted to data with following categories:')
        print('-', np.array(enc.categories_).flatten().tolist())

    # 生成独热编码
    maxspeeds_one_hot = {k: enc.transform(np.array(v).reshape(1, -1)).toarray().flatten().tolist()
                         for k, v in maxspeeds_single_val.items()}

    if verbose > 0:
        print('- One-hot encoded maxspeed limits generated.')

    nx.set_edge_attributes(g, maxspeeds_one_hot, 'maxspeed_one_hot')

    if verbose > 0:
        print('Done.')


def generate_graph(g, PARAMS, convert_labels=True, remove_unwanted=True,
                   one_hot_maxspeed=True, standardize_geoms=True, verbose=1):
    """
    对图进行处理，生成特征
    """
    if convert_labels:
        convert_class_labels(g, PARAMS)

    if remove_unwanted:
        remove_unwanted_attributes(g, PARAMS)

    if standardize_geoms:
        standardize_geometries(g, PARAMS, verbose=verbose)

    midpoint_generation(g)
    midpoint_subtraction(g)

    if one_hot_maxspeed:
        one_hot_encode_maxspeeds(g, verbose=verbose)

    return g


def copy_edge_attributes_to_nodes(g, l, verbose=1):
    """
    将原始图的边属性复制到线图的节点属性
    """
    if verbose > 0:
        print('Copying old edge attributes to new node attributes...')

    node_attr = {}
    for u, v, d in g.edges(data=True):
        node_attr[(u, v)] = d

    nx.set_node_attributes(l, node_attr)


def convert_to_line_graph(g, verbose=1):
    """
    将原始图转换为线图
    """
    if verbose > 0:
        print('\n---Original Graph---')
        print(nx.info(g))

    if verbose > 0:
        print('\nConverting to line graph...')

    # 创建线图
    l = nx.line_graph(g)

    # 复制图属性
    l.graph['name'] = g.graph['name'] + '_line'
    l.graph['osm_query_date'] = g.graph['osm_query_date']
    l.graph['name'] = g.graph['name']

    # 复制边属性到新节点
    copy_edge_attributes_to_nodes(g, l, verbose=verbose)

    # 重新标记节点
    mapping = {}
    for n in l:
        mapping[n] = n
    nx.set_node_attributes(l, mapping, 'original_id')
    l = nx.relabel.convert_node_labels_to_integers(l, first_label=0, ordering='default')

    if verbose > 0:
        print('\n---Converted Graph---')
        print(nx.info(l))
        print('Done.')

    return l


def save_data(L, path, prefix):
    """
    保存线图数据
    """
    prefix = '/' + prefix

    # 删除不需要的属性
    for n in L:
        for att in ['geometry', 'highway', 'maxspeed']:
            if att in L.nodes[n]:
                L.nodes[n].pop(att, None)

    # 转换NumPy数组为列表
    for u, d in L.nodes(data=True):
        for key, val in d.items():
            if isinstance(val, np.ndarray):
                d[key] = val.flatten().tolist()

    for u, v, d in L.edges(data=True):
        for key, val in d.items():
            if isinstance(val, np.ndarray):
                d[key] = val.flatten().tolist()

    # 保存ID映射
    id_map = {}
    for n in L.nodes:
        id_map[str(n)] = n

    out_file = path + prefix + '-id_map.json'
    with open(out_file, 'w') as fp:
        json.dump(id_map, fp)
    print('ID-Map saved in', out_file)

    # 保存类别映射
    class_map = {}
    for n in L.nodes:
        if 'label' in L.nodes[n]:
            class_map[str(n)] = np.array(L.nodes[n]['label']).astype(int).tolist()

    out_file = path + prefix + '-class_map.json'
    with open(out_file, 'w') as fp:
        json.dump(class_map, fp)
    print('Class-Map saved in', out_file)

    # 保存特征
    data_arr = []
    out_file = path + prefix + '-feats.npy'
    for n, d in L.nodes(data=True):
        if 'midpoint' in d and 'maxspeed_one_hot' in d and 'geom' in d and 'length' in d:
            data_arr.append(np.hstack(
                ([
                    d['midpoint'],
                    np.array(d['maxspeed_one_hot']),
                    np.array(d['geom']),
                    d['length']
                ])))

    np.save(out_file, np.array(data_arr))
    print('Features saved in', out_file)

    # 保存图结构
    for n in L:
        for att in ['length', 'label', 'geom', 'midpoint', 'maxspeed_one_hot']:
            if att in L.nodes[n]:
                L.nodes[n].pop(att, None)

    data = nx.json_graph.node_link_data(L)
    out_file = path + prefix + '-G.json'
    with open(out_file, 'w') as fp:
        json.dump(data, fp)
    print('Graph saved in', out_file)


def save_topological_pairs(G, path, PARAMS, bfs_walk=None, dfs_walk=None):
    """
    保存图的拓扑邻居对
    """
    WALK_LEN = PARAMS['walk_len']  # 随机游走长度
    WALK_NUM = PARAMS['walk_num']  # 每个节点的游走次数
    prefix = '/' + PARAMS['prefix']

    # 提取训练节点
    nodes = [n for n in G.nodes() if not G.nodes[n]["val"] and not G.nodes[n]["test"]]
    G_sub = G.subgraph(nodes)

    # 进行广度优先随机游走
    if bfs_walk:
        pairs_bfs = run_random_walks(G_sub, nodes, walk_len=WALK_LEN, num_walks=WALK_NUM)
        out_file = path + prefix + '-walks.txt'
        # 保存到文件
        with open(out_file, "w") as fp:
            fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs_bfs]))
        print('BFS walks saved in', out_file)

    # 进行深度优先随机游走
    if dfs_walk:
        pairs_dfs = run_dfs_walks(G_sub, nodes, dfs_len=2 * WALK_LEN, num_walks=WALK_NUM)
        out_file = path + prefix + "-dfs-walks.txt"
        # 保存到文件
        with open(out_file, "w") as fp:
            fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs_dfs]))
        print('DFS walks saved in', out_file)


def run_random_walks(G, nodes, walk_len=None, num_walks=None):
    """
    在给定图上对所有节点进行无偏随机游走
    返回：共现节点对数组
    """
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(walk_len):
                neighbors = list(G.neighbors(curr_node))
                if not neighbors:
                    break
                next_node = random.choice(neighbors)  # 无偏随机游走
                # 自我共现没有用处
                if curr_node != node:
                    pairs.append((node, curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")

    print('Number of local neighbors:', len(pairs))
    return pairs


def run_dfs_walks(G, nodes, dfs_len=None, num_walks=None):
    """
    在给定图上进行深度优先随机游走
    """
    dfs_pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            prev_node = None
            next_node = None
            for j in range(dfs_len):
                if prev_node is None:
                    depth_1_neighbors = list(G.neighbors(curr_node))
                    if not depth_1_neighbors:
                        break
                    next_node = random.choice(depth_1_neighbors)

                # 找到depth_2_neighbors (不在curr_node的邻居中的next_node的邻居)
                depth_2_neighbors = [neigh for neigh in G.neighbors(next_node)
                                     if neigh not in G.neighbors(curr_node) and neigh != curr_node]
                if not depth_2_neighbors:
                    break

                prev_node = curr_node
                curr_node = next_node
                next_node = random.choice(depth_2_neighbors)

            if curr_node != node:
                dfs_pairs.append((node, curr_node))

        if count % 1000 == 0:
            print("Done DFS walks for", count, "nodes")

    print('Number of global neighbors:', len(dfs_pairs))
    return dfs_pairs


def calculate_hyperbolicities(G, sample_size=100):
    """
    计算图的δ-hyperbolicity (近似)
    使用采样来处理大图
    """
    import scipy.spatial.distance as dist

    # 对于大图，采样节点
    if len(G) > sample_size:
        sampled_nodes = random.sample(list(G.nodes()), sample_size)
        G_sample = G.subgraph(sampled_nodes)
    else:
        G_sample = G

    # 计算所有节点对之间的最短路径距离
    all_paths = dict(nx.all_pairs_shortest_path_length(G_sample))

    # 计算四元组的δ值
    delta_values = []
    nodes = list(G_sample.nodes())

    # 限制计算量为1000个四元组
    max_quadruples = 1000
    count = 0

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            for k in range(j + 1, len(nodes)):
                for l in range(k + 1, len(nodes)):
                    x, y, u, v = nodes[i], nodes[j], nodes[k], nodes[l]

                    # 如果某些节点对之间没有路径，跳过
                    if x not in all_paths or y not in all_paths[x] or \
                            u not in all_paths or v not in all_paths[u] or \
                            x not in all_paths or u not in all_paths[x] or \
                            y not in all_paths or v not in all_paths[y] or \
                            x not in all_paths or v not in all_paths[x] or \
                            y not in all_paths or u not in all_paths[y]:
                        continue

                    # 计算三个距离和
                    S = [
                        all_paths[x][y] + all_paths[u][v],
                        all_paths[x][u] + all_paths[y][v],
                        all_paths[x][v] + all_paths[y][u]
                    ]

                    # 按降序排列
                    S.sort(reverse=True)

                    # 计算δ值 = (S_1 - S_2) / 2
                    delta = (S[0] - S[1]) / 2
                    delta_values.append(delta)

                    count += 1
                    if count >= max_quadruples:
                        break
                if count >= max_quadruples:
                    break
            if count >= max_quadruples:
                break
        if count >= max_quadruples:
            break

    if delta_values:
        return max(delta_values)
    else:
        return float('inf')  # 如果无法计算，返回无穷大


def get_pos(g):
    """
    获取图节点的坐标
    """
    x = nx.get_node_attributes(g, 'x')
    y = nx.get_node_attributes(g, 'y')

    pos = {}
    for n in g:
        if n in x and n in y:
            pos[n] = (x[n], y[n])

    return pos


def get_midpoint(g):
    """
    获取线图节点的中点坐标
    """
    pos = {}
    for u, d in g.nodes(data=True):
        if 'midpoint' in d:
            pos[u] = d['midpoint']

    return pos


def draw_graph(G, L, city_name, save_path=None):
    """
    绘制原始图和线图，并按道路类型着色
    """
    # 创建大画布
    fig, ax = plt.subplots(1, 1, figsize=(20, 20), sharex=True, sharey=True)

    # 绘制原始图
    nx.draw_networkx(G, pos=get_pos(G), ax=ax, with_labels=False,
                     node_size=25, node_color='#999999', edge_color='#999999', width=3, alpha=0.7)

    # 绘制线图
    nx.draw_networkx(L, pos=get_midpoint(L), ax=ax, with_labels=False,
                     node_size=25, node_color='black', edge_color='darkred', width=1, alpha=0.5)

    # 关联标签与名称和颜色
    label_values = [0, 1, 2, 3, 4]
    label_names = ['Major Roads', 'Tertiary', 'Unclassified', 'Residential', 'Living Streets']
    colors = ['red', 'orange', 'yellow', 'skyblue', 'lime']

    # 按道路类型绘制线图节点
    for c, label, label_name in zip(colors, label_values, label_names):
        L_sub = L.subgraph([n for n, v in L.nodes(data=True) if 'label' in v and v['label'] == label])
        nx.draw_networkx(L_sub, pos=get_midpoint(L_sub), ax=ax, with_labels=False,
                         node_size=15, node_color=c, edge_color=c, width=0, label=label_name)

    # 添加图例和标题
    plt.legend(fontsize=16)
    plt.title(f"Road Network of {city_name}", fontsize=20)

    # 如果指定了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Graph visualization saved to {save_path}")

    plt.show()


def main():
    """
    主函数：创建意大利城市的OSM Inductive数据集
    """
    print("Starting Italian Cities OSM Inductive Dataset Generation...")
    start_time = time.time()

    # 提取OSM数据
    print("\n1. Extracting OSM road networks for Italian cities...")
    G, PARAMS = extract_osm_network_inductive()

    # 生成特征
    print("\n2. Processing graph and generating features...")
    G = generate_graph(G, PARAMS, verbose=1)

    # 转换为线图
    print("\n3. Converting to line graph...")
    L = convert_to_line_graph(G)

    # 对某些城市计算hyperbolicity度量
    print("\n4. Calculating hyperbolicity for sample cities...")
    city_deltas = {}
    for city, coords in list(PARAMS['places'].items())[:5]:  # 仅对前5个城市计算
        try:
            # 获取原始图并直接投影
            city_g = ox.graph_from_point(coords, dist=PARAMS['buffer'],
                                         network_type='drive', simplify=True)
            # 转换为NetworkX无向图
            city_g = nx.Graph(city_g.to_undirected())
            delta = calculate_hyperbolicities(city_g)
            city_deltas[city] = delta
            print(f"{city}: δ-hyperbolicity = {delta}")
        except Exception as e:
            print(f"Error calculating hyperbolicity for {city}: {e}")

    # 保存数据
    print("\n5. Saving dataset...")
    save_data(L, INDUCTIVE_DIR, PARAMS['prefix'])

    # 生成拓扑邻接对
    print("\n6. Generating topological neighbor pairs...")
    save_topological_pairs(L, INDUCTIVE_DIR, PARAMS, bfs_walk=True, dfs_walk=True)

    # 可视化一些示例城市
    print("\n7. Visualizing sample cities...")
    visualization_dir = os.path.join(BASE_DIR, 'visualizations')
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)

    # 选取2个有代表性的城市进行可视化
    for city_name in list(PARAMS['places'].keys())[:2]:  # 仅可视化前两个城市
        try:
            coords = PARAMS['places'][city_name]
            # 获取原始图并直接投影
            city_g = ox.graph_from_point(coords, dist=PARAMS['buffer'],
                                         network_type='drive', simplify=True)
            # 转换为NetworkX无向图
            city_g = nx.Graph(city_g.to_undirected())
            city_g = generate_graph(city_g, PARAMS, verbose=0)
            city_l = convert_to_line_graph(city_g, verbose=0)

            save_path = os.path.join(visualization_dir, f"{city_name}_road_network.png")
            draw_graph(city_g, city_l, city_name, save_path)
            print(f"Visualized {city_name}")
        except Exception as e:
            print(f"Error visualizing {city_name}: {e}")

    # 完成
    end_time = time.time()
    print("\nDataset generation completed!")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Data saved in: {INDUCTIVE_DIR}")

    # 打印hyperbolicity信息
    if city_deltas:
        print("\nHyperbolicity values for sample cities:")
        for city, delta in city_deltas.items():
            print(f"{city}: δ = {delta}")
        print(f"Average δ-hyperbolicity: {sum(city_deltas.values()) / len(city_deltas)}")

    print("\nThe dataset is now ready to be used with the GAIN model!")
    print("You can load this data using the same loading mechanism as in the original GAIN code.")
    print("Example usage:")
    print("  from codes.load_input import load_data")
    print("  G, feats, id_map, walks, class_map = load_data('italy-osm', './italian_graph_data/osm_inductive')")


if __name__ == "__main__":
    main()