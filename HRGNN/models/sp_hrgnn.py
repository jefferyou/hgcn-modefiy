import tensorflow as tf
from utils import hrgnn_layer as layers
from utils import util
from models.base_hgattn import BaseHGAttN


class SpHRGNN(BaseHGAttN):
    """
    Sparse Hyperbolic Road Graph Neural Network (HRGNN).

    HRGNN is a specialized architecture for road network representation learning
    that leverages hyperbolic geometry to capture hierarchical road structures.

    Key components:
    1. Hyperbolic Feature Transformation (Section 3.4.1)
       - Maps Euclidean features to hyperbolic space using exponential map
       - Applies hyperbolic linear transformations

    2. Dual-Pathway Hyperbolic Attention (Section 3.4.2-3.4.3)
       - Local pathway: processes direct neighbors
       - DFS pathway: captures long-range dependencies via importance-guided DFS
       - Attention coefficients computed using hyperbolic distances (Eq. 12-13)

    3. Multi-layer Aggregation (Section 3.4.4)
       - GIN-style learnable self-loops (Eq. 15-16)
       - Multi-scale feature extraction

    4. Adaptive Fusion Strategy (Section 3.4.5)
       - Dynamically weights local and DFS pathways (Eq. 17-18)
       - Learns optimal combination based on node characteristics

    References:
        Paper: "HRGNN: Hyperbolic Graph Neural Networks with Dual-Pathway
                Attention for Learning Representation of Hierarchical Road Networks"

    Args:
        See inference() method for detailed parameter descriptions.
    """

    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat, hid_units, n_heads, activation=tf.nn.elu, c=1,
                  model_size="big", use_global_walks=True,fusion_type="adaptive",dfs_steps=5):
        """
        Build HRGNN model inference graph.

        This function constructs the complete HRGNN architecture, implementing
        the dual-pathway hyperbolic attention mechanism described in the paper.

        Args:
            inputs (tf.Tensor): Input node features [batch_size, nb_nodes, input_dim]
            nb_classes (int): Number of output classes for road type classification
            nb_nodes (int): Total number of nodes in the road network graph
            training (tf.placeholder): Boolean placeholder for training mode
            attn_drop (tf.placeholder): Attention coefficient dropout rate
            ffd_drop (tf.placeholder): Feature dropout rate
            bias_mat (tf.SparseTensor): Adjacency matrix defining graph structure
            hid_units (list): Hidden units per layer, e.g., [256] for 1-layer
            n_heads (list): Number of attention heads per layer, e.g., [8, 1]
            activation (callable): Activation function (default: tf.nn.elu)
            c (int): Curvature learning mode:
                - 0: Fixed curvature (c=1.0)
                - 1: Learnable curvature (initialized at 1.0)
            model_size (str): Model variant:
                - "small": 128-dim hidden MLP
                - "big": 256-dim hidden MLP
            use_dfs_pathway (bool): Whether to use DFS pathway in dual-pathway attention.
                - True: Use both local + DFS pathways (full HRGNN)
                - False: Use only local pathway (ablation study)
            fusion_type (str): Strategy for combining local and DFS pathways:
                - "simple": Uniform averaging (baseline)
                - "adaptive": Learnable fusion weights (Eq. 17-18, recommended)
            dfs_steps (int): Number of DFS exploration steps T (default: 5, Eq. 11)

        Returns:
            tuple: (logits, embeddings, curvature)
                - logits (tf.Tensor): Classification logits [nb_nodes, nb_classes]
                - embeddings (tf.Tensor): Learned node embeddings [nb_nodes, hid_units[-1]]
                - curvature (tf.Variable): Learned hyperbolic curvature parameter
        """

        # Process inputs
        attns = []
        inputs = tf.transpose(tf.squeeze(inputs, 0))

        # Map inputs to hyperbolic space
        print("Mapping inputs to hyperbolic space...")
        inputs = tf.transpose(util.tf_mat_exp_map_zero(inputs))
        inputs = tf.expand_dims(inputs, 0)

        # Prepare input for multi-head attention
        input_list = [inputs for _ in range(n_heads[0])]

        print("Applying dual-pathway hyperbolic attention (local + DFS)...")
        att_local, this_c = layers.hyperbolic_gain_attention_head(
            input_list,
            num_heads=n_heads[0],
            adj_mat=bias_mat,
            out_sz=hid_units[0],
            activation=activation,
            nb_nodes=nb_nodes,
            tr_c=c,
            in_drop=ffd_drop,
            coef_drop=attn_drop,
            model_size=model_size,
            name="hyperbolic_gain_local_layer",
            walk_type="local"
        )

        # Store local attention outputs
        attns_local = att_local

        # Incorporate global neighborhood if specified
        if use_global_walks:
            print("Applying Hyperbolic GAIN attention with global neighborhood...")
            att_global, _ = layers.hyperbolic_gain_attention_head(
                input_list,
                num_heads=n_heads[0],
                adj_mat=bias_mat,
                out_sz=hid_units[0],
                activation=activation,
                nb_nodes=nb_nodes,
                tr_c=c,
                pre_curvature=this_c,
                in_drop=ffd_drop,
                coef_drop=attn_drop,
                model_size=model_size,
                name="hyperbolic_gain_global_layer",
                walk_type="global",
                dfs_steps=dfs_steps
            )

            # Combine local and DFS pathway attention outputs
            attns = []
            if fusion_type == "simple":
                # Simple averaging fusion (baseline for comparison)
                attns = []
                for i in range(len(att_local)):
                    combined = tf.add(att_local[i], att_global[i]) / 2.0
                    attns.append(combined)
            elif fusion_type == "adaptive":
                # Adaptive fusion strategy (Equations 17-18 in paper)
                for i in range(len(att_local)):
                    with tf.variable_scope(f"fusion_layer_{i}"):
                        # Equation 17: Compute fusion features from local pathway
                        # f^(k) = mean(H^(k,local), axis=-1)
                        local_features = tf.reduce_mean(att_local[i], axis=-1, keepdims=True)
                        # Equation 18: Learn fusion weight via dense layer + sigmoid
                        # w^(k) = σ(W_fusion^(k) · f^(k) + b_fusion^(k))
                        fusion_weight = tf.layers.dense(
                            local_features,
                            units=1,
                            activation=tf.nn.sigmoid,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            name="fusion_weight"
                        )

                        # Broadcast fusion weight to match feature dimensions
                        fusion_weight = tf.tile(fusion_weight, [1, 1, att_local[i].shape[-1]])

                        # Adaptive weighted combination of pathways
                        # H_fused = w · H_local + (1-w) · H_dfs
                        combined = fusion_weight * att_local[i] + (1.0 - fusion_weight) * att_global[i]
                        attns.append(combined)

        else:
            # Use only local attention
            attns = attns_local

        # Add hidden layers if specified
        for i in range(1, len(hid_units)):
            attns_prev = attns
            attns = []

            print(f'Adding hidden layer {i}...')
            for _ in range(n_heads[i]):
                att, this_c = layers.hyperbolic_gain_attention_head(
                    attns_prev,
                    num_heads=n_heads[i],
                    pre_curvature=this_c,
                    adj_mat=bias_mat,
                    out_sz=hid_units[i],
                    activation=activation,
                    nb_nodes=nb_nodes,
                    tr_c=c,
                    in_drop=ffd_drop,
                    coef_drop=attn_drop,
                    model_size=model_size,
                    name=f"hyperbolic_gain_layer{i + 1}"
                )
                attns.append(att)

        # Output layer
        print("Creating output layer...")
        out = []

        # Final attention layer to produce class logits
        att, last_c = layers.hyperbolic_gain_attention_head(
            attns,
            num_heads=n_heads[-1],
            pre_curvature=this_c,
            adj_mat=bias_mat,
            out_sz=nb_classes,
            activation=lambda x: x,
            nb_nodes=nb_nodes,
            tr_c=0,  # Fix curvature in output layer
            in_drop=ffd_drop,
            coef_drop=attn_drop,
            model_size=model_size,
            name="output_layer"
        )

        out = att

        # Average outputs from multiple heads
        logits = tf.add_n(out) / n_heads[-1]

        # Return logits, embeddings and learned curvature
        emb_concat = tf.concat(attns, axis=-1)

        return logits, emb_concat, this_c