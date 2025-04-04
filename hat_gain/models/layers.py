import tensorflow as tf
import numpy as np
from utils import hyperbolic_utils as hutils

class Layer(object):
    """Base layer class for all model layers."""
    
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
            
        self.name = kwargs.get('name', None)
        if self.name is None:
            self.name = self.__class__.__name__.lower()
            
        self.logging = kwargs.get('logging', False)
        self.model_size = kwargs.get('model_size', 'small')
        self.vars = {}
        
    def _call(self, inputs):
        return inputs
        
    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs
            
    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer with optional batch normalization."""
    
    def __init__(self, input_dim, output_dim, dropout=0., 
                 act=tf.nn.relu, bias=True, batch_norm=False, **kwargs):
        super(Dense, self).__init__(**kwargs)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.bias = bias
        self.batch_norm = batch_norm
        
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.get_variable(
                'weights', 
                shape=[input_dim, output_dim],
                dtype=tf.float32,
                initializer=tf.glorot_uniform_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(0.0005)
            )
            
            if self.bias:
                self.vars['bias'] = tf.get_variable(
                    'bias',
                    shape=[output_dim],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer()
                )
                
        if self.batch_norm:
            self.bn = tf.layers.BatchNormalization(name=self.name+'_bn')
            
        if self.logging:
            self._log_vars()
            
    def _call(self, inputs):
        x = inputs
        
        # Apply dropout
        x = tf.cond(
            tf.greater(self.dropout, 0.0),
            lambda: tf.nn.dropout(x, 1.0 - self.dropout),
            lambda: x
        )
            
        # Apply linear transformation
        output = tf.matmul(x, self.vars['weights'])
        
        # Apply bias if specified
        if self.bias:
            output += self.vars['bias']
            
        # Apply batch normalization if specified
        if self.batch_norm:
            output = self.bn(output, training=True)
            
        # Apply activation function
        if self.act is not None:
            if self.act == tf.nn.leaky_relu:
                output = self.act(output, alpha=0.1)
            else:
                output = self.act(output)
                
        return output


class HyperbolicGAIN(Layer):
    """
    Hyperbolic Graph Attention Isomorphism Network layer.
    
    This layer combines elements from HAT (Hyperbolic Attention) and GAIN
    (Graph Attention Isomorphism Network) for improved graph representation.
    """
    
    def __init__(self, input_dim, output_dim, adj_mat, activation=tf.nn.elu,
                 num_heads=8, dropout=0.0, bias=True, curvature_trainable=True,
                 initial_curvature=1.0, model_size="small", epsilon_trainable=True,
                 initial_epsilon=0.5, concat=False, **kwargs):
        super(HyperbolicGAIN, self).__init__(**kwargs)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adj_mat = adj_mat
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        self.activation = activation
        self.concat = concat
        
        # Hyperbolic geometry parameters
        self.curvature_trainable = curvature_trainable
        self.initial_curvature = initial_curvature
        
        # GAIN parameters for self-attention
        self.epsilon_trainable = epsilon_trainable
        self.initial_epsilon = initial_epsilon
        
        # Set hidden dimension based on model size
        if model_size == "small":
            self.hidden_dim = 128
        elif model_size == "big":
            self.hidden_dim = 256
        else:
            self.hidden_dim = output_dim * 2
            
        # Initialize variables
        with tf.variable_scope(self.name + '_vars'):
            # Curvature parameter
            self.vars['curvature'] = tf.Variable(
                [initial_curvature], 
                trainable=curvature_trainable,
                name='curvature'
            )
            
            # GAIN self-attention epsilon parameter
            self.vars['epsilon'] = tf.Variable(
                initial_epsilon,
                trainable=epsilon_trainable,
                name='epsilon'
            )
            
            # Create variables for each attention head
            for head in range(num_heads):
                # Transform weights
                self.vars[f'transform_{head}'] = tf.get_variable(
                    f'transform_{head}',
                    shape=[input_dim, output_dim],
                    dtype=tf.float32,
                    initializer=tf.glorot_uniform_initializer()
                )
                
                # Attention weights
                self.vars[f'attention_{head}'] = tf.get_variable(
                    f'attention_{head}',
                    shape=[2 * output_dim, 1],
                    dtype=tf.float32,
                    initializer=tf.glorot_uniform_initializer()
                )
                
                # Self weights (for GAIN)
                self.vars[f'self_weight_{head}'] = tf.get_variable(
                    f'self_weight_{head}',
                    shape=[input_dim, output_dim],
                    dtype=tf.float32,
                    initializer=tf.glorot_uniform_initializer()
                )
                
                # Neighbor weights (for GAIN)
                self.vars[f'neigh_weight_{head}'] = tf.get_variable(
                    f'neigh_weight_{head}',
                    shape=[output_dim, output_dim],
                    dtype=tf.float32,
                    initializer=tf.glorot_uniform_initializer()
                )
                
                # Bias terms
                if self.bias:
                    self.vars[f'bias_{head}'] = tf.get_variable(
                        f'bias_{head}',
                        shape=[output_dim],
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer()
                    )
            
            # MLP weights for processing attentional outputs
            # MLP layer 1
            self.vars['mlp_w1'] = tf.get_variable(
                'mlp_w1',
                shape=[output_dim, self.hidden_dim],
                dtype=tf.float32,
                initializer=tf.glorot_uniform_initializer()
            )
            
            self.vars['mlp_b1'] = tf.get_variable(
                'mlp_b1',
                shape=[self.hidden_dim],
                dtype=tf.float32,
                initializer=tf.zeros_initializer()
            )
            
            # MLP layer 2
            self.vars['mlp_w2'] = tf.get_variable(
                'mlp_w2',
                shape=[self.hidden_dim, output_dim],
                dtype=tf.float32,
                initializer=tf.glorot_uniform_initializer()
            )
            
            self.vars['mlp_b2'] = tf.get_variable(
                'mlp_b2',
                shape=[output_dim],
                dtype=tf.float32,
                initializer=tf.zeros_initializer()
            )
            
        # Create batch normalization layers
        self.bn1 = tf.layers.BatchNormalization(name=self.name+'_bn1')
        self.bn2 = tf.layers.BatchNormalization(name=self.name+'_bn2')
        
        if self.logging:
            self._log_vars()
    
    def _call(self, inputs):
        """
        Apply hyperbolic graph attention mechanism.
        
        Args:
            inputs: Node features tensor of shape [batch_size, num_nodes, input_dim]
                   or [num_nodes, input_dim] if batch_size=1
        
        Returns:
            Output tensor of shape [batch_size, num_nodes, output_dim]
        """
        # Ensure inputs are 3D [batch_size, num_nodes, input_dim]
        x = inputs
        input_shape = tf.shape(x)
        
        if len(x.shape) == 2:
            x = tf.expand_dims(x, 0)  # Add batch dimension
            
        batch_size = tf.shape(x)[0]
        num_nodes = tf.shape(x)[1]
        
        # Map features to hyperbolic space if they aren't already
        # We first need to map them to tangent space at origin using log map
        # Reshape for processing
        x_flat = tf.reshape(x, [-1, self.input_dim])
        
        # Apply dropout
        x_flat = tf.cond(
            tf.greater(self.dropout, 0.0),
            lambda: tf.nn.dropout(x_flat, 1.0 - self.dropout),
            lambda: x_flat
        )

        # Multi-head attention outputs
        head_outputs = []
        
        # Process each attention head
        for head in range(self.num_heads):
            # Transform input features
            transformed = tf.matmul(x_flat, self.vars[f'transform_{head}'])
            transformed = tf.reshape(transformed, [batch_size, num_nodes, self.output_dim])
            
            # Map to hyperbolic space using exponential map
            # Reshape for hyperbolic operations
            transformed_flat = tf.reshape(transformed, [-1, self.output_dim])
            transformed_hyp = hutils.exponential_map_zero(transformed_flat, self.vars['curvature'])
            transformed_hyp = tf.reshape(transformed_hyp, [batch_size, num_nodes, self.output_dim])
            
            # Compute attention scores using hyperbolic distance
            # Get sparse adjacency indices
            adj_indices = self.adj_mat.indices
            
            # Gather node features for connected pairs
            node_i = tf.gather(transformed_hyp, adj_indices[:, 0], axis=1)
            node_j = tf.gather(transformed_hyp, adj_indices[:, 1], axis=1)
            
            # Compute hyperbolic distance
            node_i_flat = tf.reshape(node_i, [-1, self.output_dim])
            node_j_flat = tf.reshape(node_j, [-1, self.output_dim])
            
            # Compute squared distance for attention scores
            hyperbolic_dist = hutils.hyperbolic_distance(node_i_flat, node_j_flat, self.vars['curvature'])
            squared_dist = tf.square(hyperbolic_dist)
            
            # Create attention scores (negative distance for higher attention to closer nodes)
            attention_values = -tf.squeeze(squared_dist)
            
            # Create sparse attention tensor
            attention_sparse = tf.SparseTensor(
                indices=adj_indices,
                values=attention_values,
                dense_shape=[num_nodes, num_nodes]
            )
            
            # Apply softmax to get normalized attention coefficients
            attention_coeffs = tf.sparse_softmax(attention_sparse)
            
            # Apply self-transform (GAIN style)
            self_transform = tf.matmul(x_flat, self.vars[f'self_weight_{head}'])
            self_transform = tf.reshape(self_transform, [batch_size, num_nodes, self.output_dim])
            
            # Apply attention to transformed features
            # First create a dense tensor from sparse attention
            messages = tf.sparse_tensor_dense_matmul(attention_coeffs, transformed)
            
            # GAIN style: add weighted self-connections with epsilon
            output = messages + (1.0 + self.vars['epsilon']) * self_transform
            
            # Apply batch normalization
            output_flat = tf.reshape(output, [-1, self.output_dim])
            output_bn = self.bn1(output_flat, training=True)
            
            # Apply MLP (GAIN multiset function)
            # First hidden layer
            hidden = tf.matmul(output_bn, self.vars['mlp_w1']) + self.vars['mlp_b1']
            hidden = tf.nn.leaky_relu(hidden, alpha=0.1)
            hidden = self.bn2(hidden, training=True)
            
            # Second layer
            output_final = tf.matmul(hidden, self.vars['mlp_w2']) + self.vars['mlp_b2']
            
            # Apply activation
            if self.activation is not None:
                if self.activation == tf.nn.leaky_relu:
                    output_final = self.activation(output_final, alpha=0.1)
                else:
                    output_final = self.activation(output_final)
            
            # Reshape back
            output_reshaped = tf.reshape(output_final, [batch_size, num_nodes, self.output_dim])
            
            # Add to list of head outputs
            head_outputs.append(output_reshaped)
        
        # Combine outputs from all attention heads
        if self.concat and self.num_heads > 1:
            # Concatenate along feature dimension
            output = tf.concat(head_outputs, axis=-1)
        else:
            # Average the outputs from different heads
            output = tf.add_n(head_outputs) / self.num_heads
        
        # Return to original shape if needed
        if len(inputs.shape) == 2:
            output = tf.squeeze(output, axis=0)
            
        return output, self.vars['curvature']