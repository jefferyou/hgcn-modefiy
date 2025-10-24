import tensorflow as tf
import numpy as np
from utils import util

PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0
clip_value = 0.98


def hyperbolic_gain_attention_head(input_seq, num_heads, out_sz, adj_mat, activation, nb_nodes,
                                   tr_c=1, pre_curvature=None, in_drop=0.0, coef_drop=0.0,
                                   model_size="small", name=None, walk_type="local"):
    """
    Hyperbolic Graph Attention Isomorphism Network layer
    Combines principles from:
    - HAT: Hyperbolic Graph Attention Network
    - GAIN: Graph Attention Isomorphism Network

    Args:
        input_seq: List of input sequences [batch_size, nb_nodes, input_dim]
        num_heads: Number of attention heads
        out_sz: Output dimension
        adj_mat: Adjacency matrix (sparse)
        activation: Activation function
        nb_nodes: Number of nodes
        tr_c: Whether to train curvature parameter (1) or fix it (0)
        pre_curvature: Initial curvature value
        in_drop: Input dropout rate
        coef_drop: Coefficient dropout rate
        model_size: Model size ('small' or 'big')
        name: Layer name
        walk_type: Type of walks to use for attention ('local' or 'global')

    Returns:
        List of output sequences [batch_size, nb_nodes, out_sz]
    """
    distance_list = []
    seq_list = []
    ret_list = []

    # Set up curvature
    is_curv_train = (tr_c == 1)
    with tf.name_scope("hyperbolic_gain") as scope:
        c = tf.Variable([1.0], trainable=is_curv_train, name="curvature")

    if pre_curvature is None:
        pre_curvature = c

    # Set up MLP dimensions based on model size
    if model_size == "small":
        hidden_dim = 128
    elif model_size == "big":
        hidden_dim = 256

    # Process input sequences
    for attn_num, seq in enumerate(input_seq):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        # Reshape for hyperbolic operations
        seq = tf.squeeze(seq, 0)  # [nb_nodes, features]
        seq = tf.transpose(seq)  # [features, nb_nodes]
        seq_size = seq.shape.as_list()

        # Create transformation weights
        with tf.variable_scope(f"transform_{attn_num}" if name is None else f"{name}/transform_{attn_num}"):
            W = tf.get_variable(
                name="weights",
                shape=[out_sz, seq_size[0]],
                initializer=tf.contrib.layers.xavier_initializer()
            )

            # Create attention weights
            att_weights = tf.get_variable(
                name="attention_weights",
                shape=[2 * out_sz, 1],
                initializer=tf.contrib.layers.xavier_initializer()
            )

            neigh_weights = tf.get_variable(
                name="neigh_weights",
                shape=[out_sz, out_sz],
                initializer=tf.contrib.layers.xavier_initializer()
            )

            self_weights = tf.get_variable(
                name="self_weights",
                shape=[seq_size[0], out_sz],
                initializer=tf.contrib.layers.xavier_initializer()
            )

            # For GAIN self-attention with epsilon
            epsilon = tf.Variable(0.5, name="epsilon", trainable=True)

        # Map features to hyperbolic space
        seq_log = util.tf_my_prod_mat_log_map_zero(seq, pre_curvature)
        seq_fts_log = tf.matmul(W, seq_log)
        seq_fts_exp = tf.transpose(util.tf_my_prod_mat_exp_map_zero(tf.transpose(seq_fts_log), c))

        # Get adjacency matrix indices
        adj_indices = adj_mat.indices
        adj_idx_x = adj_indices[:, 0]
        adj_idx_y = adj_indices[:, 1]

        # Modify adjacency based on walk type
        if walk_type == "global":
            # For global walks, we want to focus on nodes that are farther in the graph
            # but still connected through paths (similar to DFS walks)

            # Create a global adjacency matrix modifier - down-weight immediate neighbors
            # and up-weight nodes that are 2-hops away
            with tf.variable_scope(f"global_walk_{attn_num}" if name is None else f"{name}/global_walk_{attn_num}"):
                # Get 2-hop neighbors information
                two_hop_mask = tf.SparseTensor(
                    indices=adj_indices,
                    values=tf.ones_like(adj_indices[:, 0], dtype=tf.float32),
                    dense_shape=adj_mat.dense_shape
                )

                # Square the adjacency to get 2-hop connections
                two_hop_connections = tf.sparse_tensor_dense_matmul(
                    two_hop_mask,
                    tf.sparse_tensor_to_dense(two_hop_mask)
                )

                # Mask out direct connections to focus on strictly 2-hop neighbors
                direct_connections = tf.sparse_tensor_to_dense(two_hop_mask)
                strictly_two_hop = tf.cast(
                    tf.logical_and(
                        tf.greater(two_hop_connections, 0),
                        tf.equal(direct_connections, 0)
                    ),
                    tf.float32
                )

                # Get indices of 2-hop connections
                two_hop_indices = tf.where(tf.greater(strictly_two_hop, 0))

                # Use a blend of original and 2-hop connections based on a learnable parameter
                global_weight = tf.get_variable(
                    name="global_weight",
                    shape=[],
                    initializer=tf.constant_initializer(0.3),
                    trainable=True
                )

        # Gather features for source and target nodes
        fts_x = tf.gather(tf.transpose(seq_fts_exp), adj_idx_x)  # [edges, features]
        fts_y = tf.gather(tf.transpose(seq_fts_exp), adj_idx_y)  # [edges, features]

        # Calculate hyperbolic distance between node pairs
        sparse_distance = util.tf_my_mobius_list_distance(fts_x, fts_y, c)

        # Modify distance based on walk type
        if walk_type == "global":
            # For global walks, we down-weight distances for 2-hop neighbors
            # to encourage global exploration
            # 基于节点特征动态调整全局走的权重
            with tf.variable_scope(
                    f"dynamic_global_weight_{attn_num}" if name is None else f"{name}/dynamic_global_weight_{attn_num}"):
                # 使用源节点特征预测调整权重
                source_features = tf.gather(tf.transpose(seq_fts_exp), adj_idx_x)
                feature_avg = tf.reduce_mean(source_features, axis=1, keepdims=True)
                adjustment = tf.layers.dense(
                    feature_avg,
                    units=1,
                    activation=tf.nn.sigmoid,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name="weight_adjustment"
                )

                # 确保adjustment是一维的并且形状匹配
                adjustment = tf.reshape(adjustment, [-1])  # 将调整系数展平为一维

                # 打印形状用于调试（可选）
                # tf.print("sparse_distance shape:", tf.shape(sparse_distance))
                # tf.print("adjustment shape:", tf.shape(adjustment))

                # 应用动态调整的全局权重
                dynamic_global_weight = global_weight * adjustment

                # 确保动态权重和距离形状匹配
                sparse_distance_modified = sparse_distance * (1.0 - dynamic_global_weight)

                distance_list.append(sparse_distance_modified)
        else:
            # For local walks, use the original distance
            distance_list.append(sparse_distance)

        # Store feature representation for aggregation
        seq_fts = tf.transpose(seq_fts_log)  # [nb_nodes, features]
        seq_list.append(seq_fts)

    # Compute attention coefficients based on distances
    prod_dis = tf.stack(distance_list, axis=-1)
    # Make sure to reduce to a 1D tensor
    if len(prod_dis.shape) > 1:
        prod_dis = tf.reduce_sum(prod_dis ** 2, axis=-1)
        # Ensure prod_dis is 1D
        prod_dis = tf.reshape(prod_dis, [-1])
    else:
        prod_dis = prod_dis ** 2

    # Create sparse attention tensor with negative distances (closer = higher attention)
    lrelu = tf.SparseTensor(
        indices=adj_indices,
        values=-tf.sqrt(prod_dis + 1e-8),  # Negative distance for attention
        dense_shape=adj_mat.dense_shape
    )

    # Apply softmax to get normalized attention coefficients
    coefs = tf.sparse_softmax(lrelu)

    # Apply attention to node features
    for attn_num, seq_fts in enumerate(seq_list):
        with tf.variable_scope(f"aggregate_{attn_num}" if name is None else f"{name}/aggregate_{attn_num}"):
            seq_fts = tf.expand_dims(seq_fts, 0)  # [1, nb_nodes, features]

            # Apply dropout if specified
            if coef_drop != 0.0:
                coefs = tf.SparseTensor(
                    indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape
                )
            if in_drop != 0.0:
                seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

            # Reshape to match sparse tensor operations
            coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
            seq_fts = tf.squeeze(seq_fts)  # [nb_nodes, features]

            # Apply attention weights
            vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)  # [nb_nodes, features]

            # Reshape for consistency
            vals = tf.expand_dims(vals, axis=0)  # [1, nb_nodes, features]
            vals.set_shape([1, nb_nodes, out_sz])

            # Add bias
            ret_before = tf.contrib.layers.bias_add(vals)

            # GAIN-style self-loop importance with epsilon parameter
            from_neighs = ret_before
            from_self = tf.matmul(tf.squeeze(seq_fts), tf.get_variable(
                f"self_transform_{attn_num}" if name is None else f"{name}/self_transform_{attn_num}",
                shape=[seq_fts.shape.as_list()[-1], out_sz],
                initializer=tf.contrib.layers.xavier_initializer()
            ))
            from_self = tf.expand_dims(from_self, 0)

            # Combine with (1+ε) factor for self-loops from GAIN
            # Adjust epsilon based on walk type
            if walk_type == "global":
                # For global walks, we want to reduce self-importance
                effective_epsilon = epsilon * 0.7
            else:
                effective_epsilon = epsilon

            output = from_neighs + (1.0 + effective_epsilon) * from_self

            # Create MLP for multiset functions (GAIN component)
            mlp_layers = []

            # First MLP layer
            mlp_input = tf.squeeze(output)
            hidden = tf.layers.dense(
                mlp_input,
                hidden_dim,
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name=f"mlp_hidden_{attn_num}" if name is None else f"{name}/mlp_hidden_{attn_num}"
            )

            # Batch normalization
            hidden = tf.layers.batch_normalization(
                hidden,
                training=tf.constant(True),  # For simplicity, always apply BN
                name=f"mlp_bn_{attn_num}" if name is None else f"{name}/mlp_bn_{attn_num}"
            )

            # Final MLP layer
            output = tf.layers.dense(
                hidden,
                out_sz,
                activation=tf.nn.leaky_relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name=f"mlp_output_{attn_num}" if name is None else f"{name}/mlp_output_{attn_num}"
            )

            # Apply activation and map back to hyperbolic space
            ret = util.tf_my_prod_mat_exp_map_zero(activation(output), c)
            ret = tf.expand_dims(ret, 0)
            ret_list.append(ret)

    # Return list of outputs and curvature
    return ret_list, c