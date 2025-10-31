import tensorflow as tf
import numpy as np
from utils import util

PROJ_EPS = 1e-5
EPS = 1e-15
MAX_TANH_ARG = 15.0
clip_value = 0.98


def compute_importance_scores_simple(adj_sparse, nb_nodes, beta=0.3):
    """
    Compute node importance scores for DFS exploration.

    Implements Equation 8: s_i = d_i + β · b_i
    where d_i is degree centrality and b_i is betweenness centrality proxy.

    Args:
        adj_sparse: Sparse adjacency matrix (SparseTensor)
        nb_nodes: Total number of nodes
        beta: Weighting parameter for betweenness centrality (default: 0.3 as per paper)

    Returns:
        importance_scores: shape [nb_nodes], importance score per node
        high_importance_mask: shape [nb_nodes], bool mask for high-importance nodes
    """
    print("Computing importance scores (TF 1.x compatible)...")

    indices = adj_sparse.indices

    # 1. Compute degree centrality (d_i)
    source_nodes = tf.cast(indices[:, 0], tf.int32)
    degree_centrality = tf.unsorted_segment_sum(
        tf.ones_like(source_nodes, dtype=tf.float32),
        source_nodes,
        nb_nodes
    )

    # 2. Compute betweenness centrality proxy (b_i)
    # Use 2-hop neighbor count as betweenness proxy
    adj_dense = tf.sparse_tensor_to_dense(adj_sparse)
    adj_squared = tf.matmul(adj_dense, adj_dense)
    betweenness_proxy = tf.linalg.diag_part(adj_squared)

    # 3. Compute combined importance scores (Equation 8)
    importance_scores = degree_centrality + beta * betweenness_proxy

    # 4. Classify nodes into importance tiers (Equation 9)
    mean_score = tf.reduce_mean(importance_scores)
    std_score = tf.sqrt(tf.reduce_mean(tf.square(importance_scores - mean_score)))
    threshold = mean_score + std_score

    high_importance_mask = tf.greater(importance_scores, threshold)

    print("  Importance scores computed for nodes")

    return importance_scores, high_importance_mask


def create_dfs_adjacency_simple(adj_sparse, importance_scores, high_importance_mask,
                                nb_nodes, dfs_steps=5, max_neighbors_per_node=10):
    """
    Create DFS-based adjacency matrix with importance-guided exploration.

    Implements Equations 10-11 for multi-step DFS exploration.

    Args:
        adj_sparse: Sparse adjacency matrix
        importance_scores: Node importance scores from Equation 8
        high_importance_mask: High-importance node mask from Equation 9
        nb_nodes: Total number of nodes
        dfs_steps: Number of DFS steps T (default: 5 as per paper, Equation 11)
        max_neighbors_per_node: Maximum neighbors per node for computational efficiency

    Returns:
        dfs_indices: Indices for DFS adjacency matrix
    """
    print("Creating DFS adjacency with", dfs_steps, "steps (simplified)...")

    adj_dense = tf.sparse_tensor_to_dense(adj_sparse)

    # Initialize matrices for multi-step exploration
    accumulated = adj_dense
    result_matrix = adj_dense

    # Multi-step DFS exploration (Equation 10)
    for step in range(dfs_steps - 1):
        # Compute next hop: L^(t+1) = L^(t) × A
        accumulated = tf.matmul(accumulated, adj_dense)
        accumulated_binary = tf.cast(tf.greater(accumulated, 0.0), tf.float32)

        # Accumulate to result (Equation 11)
        result_matrix = result_matrix + accumulated_binary

    # Binarize final result
    result_matrix = tf.cast(tf.greater(result_matrix, 0.0), tf.float32)

    # Apply importance weighting
    importance_matrix = tf.expand_dims(importance_scores, 0)  # [1, nb_nodes]
    importance_matrix = tf.tile(importance_matrix, [nb_nodes, 1])  # [nb_nodes, nb_nodes]
    weighted_matrix = result_matrix * (1.0 + importance_matrix * 0.1)

    # Limit neighbors per node for computational efficiency
    target_edges = nb_nodes * max_neighbors_per_node
    flat_matrix = tf.reshape(weighted_matrix, [-1])
    num_nonzero = tf.reduce_sum(tf.cast(tf.greater(flat_matrix, 0.0), tf.int32))

    k = tf.minimum(target_edges, num_nonzero)
    k = tf.minimum(k, 10000000)

    top_values, _ = tf.nn.top_k(flat_matrix, k)
    threshold = top_values[-1]

    final_matrix = tf.cast(tf.greater_equal(weighted_matrix, threshold), tf.float32)
    nonzero_indices = tf.where(tf.greater(final_matrix, 0.0))

    print("DFS adjacency created")

    return nonzero_indices


def hyperbolic_gain_attention_head(input_seq, num_heads, out_sz, adj_mat, activation, nb_nodes,
                                   tr_c=1, pre_curvature=None, in_drop=0.0, coef_drop=0.0,
                                   model_size="small", name=None, walk_type="local", dfs_steps=5, max_neighbors_per_node=10):
    """
    Hyperbolic GAIN attention head with dual-pathway mechanism.

    Implements the complete dual-pathway hyperbolic attention described in
    Section 3.4 of the paper, including:
    - Feature transformation in hyperbolic space (Section 3.4.1, Equations 4-5)
    - Hyperbolic distance computation (Section 3.4.2, Equations 6-7)
    - Dual-pathway attention (Section 3.4.3, Equations 12-13)
    - Multi-layer aggregation with self-loops (Section 3.4.4, Equations 15-16)

    Args:
        input_seq: List of input feature tensors
        num_heads: Number of attention heads
        out_sz: Output dimension
        adj_mat: Sparse adjacency matrix
        activation: Activation function
        nb_nodes: Number of nodes
        tr_c: Curvature training flag (0: fixed, 1: trainable)
        pre_curvature: Pre-computed curvature value
        in_drop: Input dropout rate
        coef_drop: Attention coefficient dropout rate
        model_size: Model size ("small" or "big")
        name: Variable scope name
        walk_type: "local" for direct neighbors, "global" for DFS pathway
        dfs_steps: Number of DFS steps T (default: 5, as per Equation 11)
        max_neighbors_per_node: Maximum neighbors per node

    Returns:
        ret_list: List of attention outputs
        c: Learned curvature parameter
    """
    distance_list = []
    seq_list = []
    ret_list = []

    # Initialize curvature parameter
    is_curv_train = (tr_c == 1)
    with tf.name_scope("hyperbolic_gain") as scope:
        c = tf.Variable([1.0], trainable=is_curv_train, name="curvature")

    if pre_curvature is None:
        pre_curvature = c

    # Set MLP dimensions based on model size
    if model_size == "small":
        hidden_dim = 128
    elif model_size == "big":
        hidden_dim = 256
    else:
        hidden_dim = 128

    # Process input sequences
    for attn_num, seq in enumerate(input_seq):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq = tf.squeeze(seq, 0)
        seq = tf.transpose(seq)
        seq_size = seq.shape.as_list()

        # Create transformation weights
        scope_name = "transform_{}".format(attn_num) if name is None else "{}/transform_{}".format(name, attn_num)
        with tf.variable_scope(scope_name):
            W = tf.get_variable(
                name="weights",
                shape=[out_sz, seq_size[0]],
                initializer=tf.contrib.layers.xavier_initializer()
            )

            # Learnable self-loop parameter (Equation 16, initialized to 0.5)
            epsilon = tf.Variable(0.5, name="epsilon", trainable=True)

        # Map to hyperbolic space (Equation 4)
        seq_log = util.tf_my_prod_mat_log_map_zero(seq, pre_curvature)
        seq_fts_log = tf.matmul(W, seq_log)
        seq_fts_exp = tf.transpose(util.tf_my_prod_mat_exp_map_zero(tf.transpose(seq_fts_log), c))

        # Get adjacency matrix indices
        adj_indices = adj_mat.indices
        adj_idx_x = adj_indices[:, 0]
        adj_idx_y = adj_indices[:, 1]

        # Modify adjacency for global pathway with DFS exploration
        if walk_type == "global" and dfs_steps > 1:
            scope_name = "global_walk_{}".format(attn_num) if name is None else "{}/global_walk_{}".format(name,
                                                                                                           attn_num)
            with tf.variable_scope(scope_name):
                print("Computing importance-guided DFS...")

                # Compute importance scores (Equation 8)
                importance_scores, high_importance_mask = compute_importance_scores_simple(
                    adj_mat, nb_nodes, beta=0.3
                )

                # Create DFS adjacency (Equations 10-11)
                dfs_indices = create_dfs_adjacency_simple(
                    adj_mat,
                    importance_scores,
                    high_importance_mask,
                    nb_nodes,
                    dfs_steps=dfs_steps,  # Use configured DFS steps (default T=5 as per paper)
                    max_neighbors_per_node=max_neighbors_per_node if 'max_neighbors_per_node' in locals() else 10
                )

                adj_idx_x = dfs_indices[:, 0]
                adj_idx_y = dfs_indices[:, 1]
                adj_indices = dfs_indices

                # Global pathway weight (Equation 14)
                global_weight = tf.get_variable(
                    name="global_weight",
                    shape=[],
                    initializer=tf.constant_initializer(0.3),
                    trainable=True
                )

        # Gather node features
        fts_x = tf.gather(tf.transpose(seq_fts_exp), adj_idx_x)
        fts_y = tf.gather(tf.transpose(seq_fts_exp), adj_idx_y)

        # Compute hyperbolic distances (Equation 6)
        sparse_distance = util.tf_my_mobius_list_distance(fts_x, fts_y, c)

        # Adjust distances for global pathway (Equation 14)
        if walk_type == "global" and dfs_steps > 1:
            sparse_distance_modified = sparse_distance * (1.0 - global_weight)
            distance_list.append(sparse_distance_modified)
        else:
            distance_list.append(sparse_distance)

        seq_fts = tf.transpose(seq_fts_log)
        seq_list.append(seq_fts)

    # Compute attention coefficients (Equations 12-13)
    prod_dis = tf.stack(distance_list, axis=-1)
    if len(prod_dis.shape) > 1:
        prod_dis = tf.reduce_sum(prod_dis ** 2, axis=-1)
        prod_dis = tf.reshape(prod_dis, [-1])
    else:
        prod_dis = prod_dis ** 2

    # Create sparse attention tensor with negative distances
    lrelu = tf.SparseTensor(
        indices=adj_indices,
        values=-tf.sqrt(prod_dis + 1e-8),
        dense_shape=adj_mat.dense_shape
    )

    # Apply softmax to get attention coefficients
    coefs = tf.sparse_softmax(lrelu)

    # Apply attention to node features
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

            # Apply attention weights
            vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)

            vals = tf.expand_dims(vals, axis=0)
            vals.set_shape([1, nb_nodes, out_sz])

            ret_before = tf.contrib.layers.bias_add(vals)

            # GIN-style self-loop (Equations 15-16)
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

            # Adjust epsilon for global pathway
            if walk_type == "global":
                effective_epsilon = epsilon * 0.7
            else:
                effective_epsilon = epsilon

            # Combine neighbor and self information (Equation 16)
            output = from_neighs + (1.0 + effective_epsilon) * from_self

            # Apply MLP transformation
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

            # Map back to hyperbolic space
            ret = util.tf_my_prod_mat_exp_map_zero(activation(output), c)
            ret = tf.expand_dims(ret, 0)
            ret_list.append(ret)

    return ret_list, c