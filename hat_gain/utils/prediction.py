import tensorflow as tf
from utils import util

class HyperbolicEdgePredLayer:
    """
    Edge prediction layer for hyperbolic embeddings, modified for combined HAT-GAIN model.
    """
    
    def __init__(self, input_dim1, input_dim2, placeholders, dropout=False, 
                 act=tf.nn.sigmoid, loss_fn='xent', neg_sample_weights=1.0,
                 bias=False, **kwargs):
        """
        Initialize the hyperbolic edge prediction layer.
        
        Args:
            input_dim1: Input dimension for first node set
            input_dim2: Input dimension for second node set
            placeholders: TensorFlow placeholders
            dropout: Whether to use dropout
            act: Activation function
            loss_fn: Loss function type (xent, hinge, or skipgram)
            neg_sample_weights: Weight for negative samples in loss
            bias: Whether to use bias
        """
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.act = act
        self.bias = bias
        self.eps = 1e-7
        
        # Margin for hinge loss
        self.margin = 0.1
        self.neg_sample_weights = neg_sample_weights
        
        # Dropout option
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.
            
        # Setup output shape
        self.output_dim = 1
        
        # Create variables
        with tf.variable_scope('edge_pred_vars'):
            # Create trainable curvature parameter
            self.c = tf.Variable([1.0], trainable=True, name="curvature")
            
            # Create transformation matrices
            self.weights1 = tf.get_variable(
                'weights1',
                shape=(input_dim1, input_dim2),
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer()
            )
            
            if self.bias:
                self.bias_var = tf.get_variable(
                    'bias',
                    shape=[self.output_dim],
                    initializer=tf.zeros_initializer()
                )
        
        # Select loss function
        if loss_fn == 'xent':
            self.loss_fn = self._xent_loss
        elif loss_fn == 'skipgram':
            self.loss_fn = self._skipgram_loss
        elif loss_fn == 'hinge':
            self.loss_fn = self._hinge_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
    
    def hyperbolic_distance(self, u, v):
        """
        Calculate hyperbolic distance between vectors u and v.
        
        Args:
            u: First set of vectors [batch_size, features]
            v: Second set of vectors [batch_size, features]
            
        Returns:
            Hyperbolic distances [batch_size]
        """
        # Use Poincaré distance formula
        return util.tf_my_mobius_list_distance(u, v, self.c)
    
    def affinity(self, inputs1, inputs2):
        """
        Calculate affinity between two sets of hyperbolic embeddings.
        
        Args:
            inputs1: First set of node embeddings [batch_size, features]
            inputs2: Second set of node embeddings [batch_size, features]
            
        Returns:
            Affinity scores (negative distance) [batch_size]
        """
        # Apply transformation if needed
        if self.dropout > 0:
            inputs1 = tf.nn.dropout(inputs1, 1 - self.dropout)
            inputs2 = tf.nn.dropout(inputs2, 1 - self.dropout)
        
        # Calculate hyperbolic distance
        distance = self.hyperbolic_distance(inputs1, inputs2)
        
        # Convert to affinity (negative distance so closer points have higher affinity)
        affinity = -distance
        
        # Add bias if necessary
        if self.bias:
            affinity += self.bias_var
        
        return affinity
    
    def neg_cost(self, inputs1, neg_samples):
        """
        Calculate affinity between inputs and negative samples.
        
        Args:
            inputs1: Node embeddings [batch_size, features]
            neg_samples: Negative sample embeddings [num_neg_samples, features]
            
        Returns:
            Affinity matrix [batch_size, num_neg_samples]
        """
        # Apply dropout if needed
        if self.dropout > 0:
            inputs1 = tf.nn.dropout(inputs1, 1 - self.dropout)
            neg_samples = tf.nn.dropout(neg_samples, 1 - self.dropout)
        
        # Calculate pairwise hyperbolic distances
        # Reshape for broadcasting
        expanded_inputs = tf.expand_dims(inputs1, 1)  # [batch_size, 1, features]
        expanded_samples = tf.expand_dims(neg_samples, 0)  # [1, num_neg_samples, features]
        
        # Calculate distances for all pairs
        # We need a custom implementation for batched distance calculation
        inputs_norm = tf.reduce_sum(expanded_inputs ** 2, axis=2, keepdims=True)
        samples_norm = tf.reduce_sum(expanded_samples ** 2, axis=2, keepdims=True)
        dot_products = tf.matmul(expanded_inputs, expanded_samples, transpose_b=True)
        
        # Compute the distances using the Poincaré formula
        numerator = 2 * dot_products
        denominator = 1 + inputs_norm + samples_norm + 2 * dot_products
        distances = tf.acosh(1 + 2 * numerator / (denominator + self.eps))
        
        # Convert to affinity (negative distance)
        neg_aff = -distances
        
        return tf.squeeze(neg_aff, axis=1)
    
    def loss(self, inputs1, inputs2, neg_samples):
        """
        Calculate the loss for link prediction.
        
        Args:
            inputs1: First set of node embeddings [batch_size, features]
            inputs2: Second set of node embeddings [batch_size, features]
            neg_samples: Negative sample embeddings [num_neg_samples, features]
            
        Returns:
            Loss value
        """
        return self.loss_fn(inputs1, inputs2, neg_samples)
    
    def _xent_loss(self, inputs1, inputs2, neg_samples):
        """
        Cross-entropy loss for link prediction.
        
        Args:
            inputs1: First set of node embeddings [batch_size, features]
            inputs2: Second set of node embeddings [batch_size, features]
            neg_samples: Negative sample embeddings [num_neg_samples, features]
            
        Returns:
            Cross-entropy loss
        """
        # Calculate positive affinity
        aff = self.affinity(inputs1, inputs2)
        
        # Calculate negative affinity
        neg_aff = self.neg_cost(inputs1, neg_samples)
        
        # Cross-entropy loss for positive examples
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(aff), logits=aff)
        
        # Cross-entropy loss for negative examples
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(neg_aff), logits=neg_aff)
        
        # Combine with negative sample weights
        loss = tf.reduce_sum(true_xent) + self.neg_sample_weights * tf.reduce_sum(negative_xent)
        
        return loss
    
    def _skipgram_loss(self, inputs1, inputs2, neg_samples):
        """
        Skipgram-style loss for link prediction.
        
        Args:
            inputs1: First set of node embeddings [batch_size, features]
            inputs2: Second set of node embeddings [batch_size, features]
            neg_samples: Negative sample embeddings [num_neg_samples, features]
            
        Returns:
            Skipgram loss
        """
        # Calculate positive affinity
        aff = self.affinity(inputs1, inputs2)
        
        # Calculate negative affinity
        neg_aff = self.neg_cost(inputs1, neg_samples)
        
        # Skipgram loss
        neg_cost = tf.log(tf.reduce_sum(tf.exp(neg_aff), axis=1))
        loss = tf.reduce_sum(aff - neg_cost)
        
        return loss
    
    def _hinge_loss(self, inputs1, inputs2, neg_samples):
        """
        Hinge loss for link prediction.
        
        Args:
            inputs1: First set of node embeddings [batch_size, features]
            inputs2: Second set of node embeddings [batch_size, features]
            neg_samples: Negative sample embeddings [num_neg_samples, features]
            
        Returns:
            Hinge loss
        """
        # Calculate positive affinity
        aff = self.affinity(inputs1, inputs2)
        
        # Calculate negative affinity
        neg_aff = self.neg_cost(inputs1, neg_samples)
        
        # Hinge loss with margin
        diff = tf.nn.relu(tf.subtract(neg_aff, tf.expand_dims(aff, 1) - self.margin), name='diff')
        loss = tf.reduce_sum(diff)
        
        return loss