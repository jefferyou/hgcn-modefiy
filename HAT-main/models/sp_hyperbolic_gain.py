import numpy as np
import tensorflow as tf
from utils import hyperbolic_gain_layer as layers
from utils import util
from models.base_hgattn import BaseHGAttN


class SpHyperbolicGAIN(BaseHGAttN):
    """
    Sparse Hyperbolic Graph Attention Isomorphism Network (Hyperbolic GAIN)

    This model combines elements from:
    - HAT: Hyperbolic Graph Attention Network
    - GAIN: Graph Attention Isomorphism Network

    It leverages hyperbolic geometry for better representation of hierarchical data
    while using the improved attention mechanism from GAIN for better feature aggregation.
    """

    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
                  bias_mat, hid_units, n_heads, activation=tf.nn.elu, c=0,
                  model_size="small"):
        """
        Build the model inference graph.

        Args:
            inputs: Input features tensor [batch_size, nb_nodes, input_dim]
            nb_classes: Number of output classes
            nb_nodes: Number of nodes in the graph
            training: Boolean tensor for training mode
            attn_drop: Attention dropout rate
            ffd_drop: Feedforward dropout rate
            bias_mat: Bias matrix (adj matrix)
            hid_units: List of hidden units per layer
            n_heads: List of number of heads per layer
            activation: Activation function
            c: Curvature parameter (0: fixed, 1: trainable)
            model_size: Model size ('small' or 'big')

        Returns:
            logits: Output logits
            embeddings: Node embeddings
            curvature: Learned curvature value
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

        # Apply first Hyperbolic GAIN layer
        print("Applying Hyperbolic GAIN attention...")
        att, this_c = layers.hyperbolic_gain_attention_head(
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
            name="hyperbolic_gain_layer1"
        )

        # Store attention outputs
        attns = att

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