"""
TensorFlow 1.x 兼容版本的重要性引导DFS实现

关键修复：
1. 移除所有f-string print语句
2. 使用TF 1.x兼容的稀疏操作
3. 简化某些复杂的TF操作以确保兼容性
"""

import tensorflow as tf
import numpy as np
from utils import util

PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0
clip_value = 0.98


def compute_importance_scores_simple(adj_sparse, nb_nodes, beta=0.3):
    """
    简化版重要性评分计算（TF 1.x兼容）

    计算 s_i = d_i + β · b_i
    其中 b_i 使用简化的估计方法

    Args:
        adj_sparse: 稀疏邻接矩阵 SparseTensor
        nb_nodes: 节点总数
        beta: 介数中心性权重参数

    Returns:
        importance_scores: shape [nb_nodes], 每个节点的重要性评分
        high_importance_mask: shape [nb_nodes], bool mask标识高重要性节点
    """
    print("Computing importance scores (TF 1.x compatible)...")

    indices = adj_sparse.indices

    # 1. 计算度中心性
    source_nodes = tf.cast(indices[:, 0], tf.int32)
    degree_centrality = tf.unsorted_segment_sum(
        tf.ones_like(source_nodes, dtype=tf.float32),
        source_nodes,
        nb_nodes
    )

    # 2. 简化的介数中心性估计
    # 使用2-hop邻居数作为介数代理
    # 对于每个节点，统计其2-hop可达的节点数

    # 首先转换为密集矩阵（对于较小的图）
    adj_dense = tf.sparse_tensor_to_dense(adj_sparse)

    # 计算A²（邻接矩阵的平方）
    adj_squared = tf.matmul(adj_dense, adj_dense)

    # 取对角线作为介数代理
    betweenness_proxy = tf.linalg.diag_part(adj_squared)

    # 3. 综合评分
    importance_scores = degree_centrality + beta * betweenness_proxy

    # 4. 节点分层
    mean_score = tf.reduce_mean(importance_scores)
    std_score = tf.sqrt(tf.reduce_mean(tf.square(importance_scores - mean_score)))
    threshold = mean_score + std_score

    high_importance_mask = tf.greater(importance_scores, threshold)

    print("  Importance scores computed for nodes")

    return importance_scores, high_importance_mask


def create_dfs_adjacency_simple(adj_sparse, importance_scores, high_importance_mask,
                                nb_nodes, dfs_steps=5, max_neighbors_per_node=10):
    """
    简化版DFS邻接矩阵构建（TF 1.x兼容）

    使用更简单的策略：
    1. 计算k-hop邻居
    2. 根据重要性过滤和加权

    Args:
        adj_sparse: 稀疏邻接矩阵
        importance_scores: 节点重要性评分
        high_importance_mask: 高重要性节点mask
        nb_nodes: 节点总数
        dfs_steps: DFS步数
        max_neighbors_per_node: 每个节点最大邻居数

    Returns:
        dfs_indices: DFS邻接矩阵的indices
    """
    print("Creating DFS adjacency with", dfs_steps, "steps (simplified)...")

    # 转换为密集矩阵进行操作
    adj_dense = tf.sparse_tensor_to_dense(adj_sparse)

    # 初始化累积矩阵
    accumulated = adj_dense
    result_matrix = adj_dense

    # 多步乘法
    for step in range(dfs_steps - 1):
        # A^(step+2) = A^(step+1) × A
        accumulated = tf.matmul(accumulated, adj_dense)

        # 二值化（只保留连接信息）
        accumulated_binary = tf.cast(tf.greater(accumulated, 0.0), tf.float32)

        # 累积到结果
        result_matrix = result_matrix + accumulated_binary

    # 二值化最终结果
    result_matrix = tf.cast(tf.greater(result_matrix, 0.0), tf.float32)

    # 应用重要性权重
    # 为每条边应用目标节点的重要性
    importance_matrix = tf.expand_dims(importance_scores, 0)  # [1, nb_nodes]
    importance_matrix = tf.tile(importance_matrix, [nb_nodes, 1])  # [nb_nodes, nb_nodes]

    # 加权
    weighted_matrix = result_matrix * (1.0 + importance_matrix * 0.1)

    # 限制每个节点的邻居数 - 使用更高效的方法避免整数溢出
    # 计算目标边数（每个节点最多max_neighbors_per_node个邻居）
    target_edges = nb_nodes * max_neighbors_per_node

    # 将矩阵展平
    flat_matrix = tf.reshape(weighted_matrix, [-1])

    # 计算实际非零元素数量
    num_nonzero = tf.reduce_sum(tf.cast(tf.greater(flat_matrix, 0.0), tf.int32))

    # 选择较小的k值（避免溢出）
    k = tf.minimum(target_edges, num_nonzero)
    k = tf.minimum(k, 10000000)  # 硬性上限：1000万条边

    # 获取top-k值
    top_values, _ = tf.nn.top_k(flat_matrix, k)

    # 使用最小的top-k值作为阈值
    threshold = top_values[-1]

    # 应用阈值
    final_matrix = tf.cast(tf.greater_equal(weighted_matrix, threshold), tf.float32)

    # 转换回稀疏格式
    nonzero_indices = tf.where(tf.greater(final_matrix, 0.0))

    print("DFS adjacency created")

    return nonzero_indices


def hyperbolic_gain_attention_head(input_seq, num_heads, out_sz, adj_mat, activation, nb_nodes,
                                   tr_c=1, pre_curvature=None, in_drop=0.0, coef_drop=0.0,
                                   model_size="small", name=None, walk_type="local", dfs_steps=5):
    """
    TensorFlow 1.x 兼容的Hyperbolic GAIN attention

    简化版DFS实现，避免复杂的稀疏操作
    """
    distance_list = []
    seq_list = []
    ret_list = []

    # 设置曲率
    is_curv_train = (tr_c == 1)
    with tf.name_scope("hyperbolic_gain") as scope:
        c = tf.Variable([1.0], trainable=is_curv_train, name="curvature")

    if pre_curvature is None:
        pre_curvature = c

    # MLP维度
    if model_size == "small":
        hidden_dim = 128
    elif model_size == "big":
        hidden_dim = 256
    else:
        hidden_dim = 128

    # 处理输入序列
    for attn_num, seq in enumerate(input_seq):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq = tf.squeeze(seq, 0)
        seq = tf.transpose(seq)
        seq_size = seq.shape.as_list()

        # 创建变换权重
        scope_name = "transform_{}".format(attn_num) if name is None else "{}/transform_{}".format(name, attn_num)
        with tf.variable_scope(scope_name):
            W = tf.get_variable(
                name="weights",
                shape=[out_sz, seq_size[0]],
                initializer=tf.contrib.layers.xavier_initializer()
            )

            epsilon = tf.Variable(0.5, name="epsilon", trainable=True)

        # 映射到双曲空间
        seq_log = util.tf_my_prod_mat_log_map_zero(seq, pre_curvature)
        seq_fts_log = tf.matmul(W, seq_log)
        seq_fts_exp = tf.transpose(util.tf_my_prod_mat_exp_map_zero(tf.transpose(seq_fts_log), c))

        # 获取邻接矩阵indices
        adj_indices = adj_mat.indices
        adj_idx_x = adj_indices[:, 0]
        adj_idx_y = adj_indices[:, 1]

        # 根据walk类型修改邻接关系
        if walk_type == "global" and dfs_steps > 1:
            scope_name = "global_walk_{}".format(attn_num) if name is None else "{}/global_walk_{}".format(name,
                                                                                                           attn_num)
            with tf.variable_scope(scope_name):
                print("Computing importance-guided DFS...")

                # 计算重要性评分
                importance_scores, high_importance_mask = compute_importance_scores_simple(
                    adj_mat, nb_nodes, beta=0.3
                )

                # 创建DFS邻接
                dfs_indices = create_dfs_adjacency_simple(
                    adj_mat,
                    importance_scores,
                    high_importance_mask,
                    nb_nodes,
                    dfs_steps=min(dfs_steps, 3),  # 限制步数避免过大
                    max_neighbors_per_node=max_neighbors_per_node if 'max_neighbors_per_node' in locals() else 10
                )

                adj_idx_x = dfs_indices[:, 0]
                adj_idx_y = dfs_indices[:, 1]
                adj_indices = dfs_indices

                # 全局权重
                global_weight = tf.get_variable(
                    name="global_weight",
                    shape=[],
                    initializer=tf.constant_initializer(0.3),
                    trainable=True
                )

        # 收集特征
        fts_x = tf.gather(tf.transpose(seq_fts_exp), adj_idx_x)
        fts_y = tf.gather(tf.transpose(seq_fts_exp), adj_idx_y)

        # 计算双曲距离
        sparse_distance = util.tf_my_mobius_list_distance(fts_x, fts_y, c)

        # 根据walk类型调整距离
        if walk_type == "global" and dfs_steps > 1:
            sparse_distance_modified = sparse_distance * (1.0 - global_weight)
            distance_list.append(sparse_distance_modified)
        else:
            distance_list.append(sparse_distance)

        seq_fts = tf.transpose(seq_fts_log)
        seq_list.append(seq_fts)

    # 计算注意力系数
    prod_dis = tf.stack(distance_list, axis=-1)
    if len(prod_dis.shape) > 1:
        prod_dis = tf.reduce_sum(prod_dis ** 2, axis=-1)
        prod_dis = tf.reshape(prod_dis, [-1])
    else:
        prod_dis = prod_dis ** 2

    # 创建稀疏注意力张量
    lrelu = tf.SparseTensor(
        indices=adj_indices,
        values=-tf.sqrt(prod_dis + 1e-8),
        dense_shape=adj_mat.dense_shape
    )

    # Softmax
    coefs = tf.sparse_softmax(lrelu)

    # 应用注意力到节点特征
    for attn_num, seq_fts in enumerate(seq_list):
        scope_name = "aggregate_{}".format(attn_num) if name is None else "{}/aggregate_{}".format(name, attn_num)
        with tf.variable_scope(scope_name):
            seq_fts = tf.expand_dims(seq_fts, 0)

            if coef_drop != 0.0:
                coefs = tf.SparseTensor(
                    indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape
                )
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

            coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
            seq_fts = tf.squeeze(seq_fts)

            # 应用注意力权重
            vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)

            vals = tf.expand_dims(vals, axis=0)
            vals.set_shape([1, nb_nodes, out_sz])

            ret_before = tf.contrib.layers.bias_add(vals)

            # GAIN风格的自环
            from_neighs = ret_before

            self_transform_name = "self_transform_{}".format(
                attn_num) if name is None else "{}/self_transform_{}".format(name, attn_num)
            from_self = tf.matmul(
                tf.squeeze(seq_fts),
                tf.get_variable(
                    self_transform_name,
                    shape=[seq_fts.shape.as_list()[-1], out_sz],
                    initializer=tf.contrib.layers.xavier_initializer()
                )
            )
            from_self = tf.expand_dims(from_self, 0)

            if walk_type == "global":
                effective_epsilon = epsilon * 0.7
            else:
                effective_epsilon = epsilon

            output = from_neighs + (1.0 + effective_epsilon) * from_self

            # MLP
            mlp_input = tf.squeeze(output)

            mlp_hidden_name = "mlp_hidden_{}".format(attn_num) if name is None else "{}/mlp_hidden_{}".format(name,
                                                                                                              attn_num)
            hidden = tf.layers.dense(
                mlp_input,
                hidden_dim,
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name=mlp_hidden_name
            )

            mlp_bn_name = "mlp_bn_{}".format(attn_num) if name is None else "{}/mlp_bn_{}".format(name, attn_num)
            hidden = tf.layers.batch_normalization(
                hidden,
                training=tf.constant(True),
                name=mlp_bn_name
            )

            mlp_output_name = "mlp_output_{}".format(attn_num) if name is None else "{}/mlp_output_{}".format(name,
                                                                                                              attn_num)
            output = tf.layers.dense(
                hidden,
                out_sz,
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name=mlp_output_name
            )

            # 映射回双曲空间
            ret = util.tf_my_prod_mat_exp_map_zero(activation(output), c)
            ret = tf.expand_dims(ret, 0)
            ret_list.append(ret)

    return ret_list, c
