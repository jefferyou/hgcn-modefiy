import tensorflow as tf
import numpy as np
from models.layers import HyperbolicGAIN, Dense
from utils import hyperbolic_utils as hutils
from models.base_model import Model

class HyperbolicGAINModel(Model):
    """
    Hyperbolic Graph Attention Isomorphism Network Model.
    
    This model combines elements from:
    - HAT (Hyperbolic Attention Network): Operating in hyperbolic space
    - GAIN (Graph Attention Isomorphism Network): Advanced attention mechanism
    
    It supports both supervised and unsupervised learning for node classification
    and link prediction tasks.
    """
    
    def __init__(self, placeholders, features, adj, degrees, is_supervised=True,
                 learning_rate=0.005, weight_decay=0.0005, hidden_dims=[64, 64],
                 num_heads=[8, 1], dropout=0.2, num_classes=None, sparse_inputs=True,
                 model_size="small", curvature_trainable=True, initial_curvature=1.0,
                 activation=tf.nn.elu, concat_heads=False, neg_sample_size=10, **kwargs):
        """
        Initialize the Hyperbolic GAIN model.
        
        Args:
            placeholders: Dictionary of placeholders
            features: Node features (dense or sparse)
            adj: Sparse adjacency matrix
            degrees: Node degrees
            is_supervised: Whether this is a supervised or unsupervised model
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization coefficient
            hidden_dims: List of hidden dimensions for each layer
            num_heads: List of number of attention heads for each layer
            dropout: Dropout rate
            num_classes: Number of output classes (for supervised learning)
            sparse_inputs: Whether input features are sparse
            model_size: Model size ("small" or "big")
            curvature_trainable: Whether to train the curvature parameter
            initial_curvature: Initial value for curvature
            activation: Activation function
            concat_heads: Whether to concatenate or average attention heads
            neg_sample_size: Number of negative samples (for unsupervised learning)
        """
        super(HyperbolicGAINModel, self).__init__(**kwargs)
        
        # Store model parameters
        self.sparse_inputs = sparse_inputs
        self.features = features
        self.adj = adj
        self.degrees = degrees
        self.is_supervised = is_supervised
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_classes = num_classes
        self.model_size = model_size
        self.curvature_trainable = curvature_trainable
        self.initial_curvature = initial_curvature
        self.activation = activation
        self.concat_heads = concat_heads
        self.neg_sample_size = neg_sample_size
        
        # Set input and output dimensions
        if sparse_inputs:
            self.input_dim = features[2][1]  # sparse_features[2] has the shape info
        else:
            self.input_dim = features.shape[1]
            
        # Extract placeholders
        self.placeholders = placeholders
        
        # Build the model
        self.curvature = None  # Will be set during model building
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.build()
    
    def _build_supervised(self):
        """Build the supervised learning model for node classification."""
        
        # Get placeholders
        self.batch = self.placeholders['batch']
        self.labels = self.placeholders['labels']
        self.dropout_ph = self.placeholders['dropout']
        
        # Create first hidden layer
        with tf.variable_scope('hgain_layer_1'):
            self.layer1, self.curvature = HyperbolicGAIN(
                input_dim=self.input_dim,
                output_dim=self.hidden_dims[0],
                adj_mat=self.adj,
                activation=self.activation,
                num_heads=self.num_heads[0],
                dropout=self.dropout_ph,
                curvature_trainable=self.curvature_trainable,
                initial_curvature=self.initial_curvature,
                model_size=self.model_size,
                concat=self.concat_heads,
                logging=self.logging
            )(self.features)
        
        # Create additional hidden layers
        current_input = self.layer1
        current_dim = self.hidden_dims[0]
        
        for i in range(1, len(self.hidden_dims)):
            with tf.variable_scope(f'hgain_layer_{i+1}'):
                current_input, self.curvature = HyperbolicGAIN(
                    input_dim=current_dim,
                    output_dim=self.hidden_dims[i],
                    adj_mat=self.adj,
                    activation=self.activation,
                    num_heads=self.num_heads[i if i < len(self.num_heads) else -1],
                    dropout=self.dropout_ph,
                    curvature_trainable=self.curvature_trainable,
                    initial_curvature=self.curvature,  # Use previous layer's curvature
                    model_size=self.model_size,
                    concat=self.concat_heads,
                    logging=self.logging
                )(current_input)
                
                current_dim = self.hidden_dims[i]
                
        # Output layer for classification
        with tf.variable_scope('output_layer'):
            # Map back to Euclidean space for classification
            current_input_flat = tf.reshape(current_input, [-1, current_dim])
            
            # Map from hyperbolic to Euclidean space
            euclidean_features = hutils.logarithmic_map_zero(current_input_flat, self.curvature)
            
            # Create dense output layer
            self.outputs = Dense(
                input_dim=current_dim,
                output_dim=self.num_classes,
                dropout=self.dropout_ph,
                act=lambda x: x,  # Identity activation for final layer
                logging=self.logging
            )(euclidean_features)
            
            # Reshape to original dimensions
            self.outputs = tf.reshape(self.outputs, [-1, self.num_classes])
            
        # Store node embeddings for analysis
        self.embeddings = current_input
    
    def _build_unsupervised(self):
        """Build the unsupervised learning model for link prediction."""
        
        # Get placeholders
        self.inputs1 = self.placeholders['batch1']
        self.inputs2 = self.placeholders['batch2']
        self.dropout_ph = self.placeholders['dropout']
        
        # Create sampling-based embeddings for input nodes
        with tf.variable_scope('hgain_layer_1'):
            self.layer1_1, self.curvature = HyperbolicGAIN(
                input_dim=self.input_dim,
                output_dim=self.hidden_dims[0],
                adj_mat=self.adj,
                activation=self.activation,
                num_heads=self.num_heads[0],
                dropout=self.dropout_ph,
                curvature_trainable=self.curvature_trainable,
                initial_curvature=self.initial_curvature,
                model_size=self.model_size,
                concat=self.concat_heads,
                logging=self.logging
            )(tf.nn.embedding_lookup(self.features, self.inputs1))
            
        # Process input nodes through the network
        current_input1 = self.layer1_1
        current_dim = self.hidden_dims[0]
        
        for i in range(1, len(self.hidden_dims)):
            with tf.variable_scope(f'hgain_layer_{i+1}', reuse=tf.AUTO_REUSE):
                current_input1, self.curvature = HyperbolicGAIN(
                    input_dim=current_dim,
                    output_dim=self.hidden_dims[i],
                    adj_mat=self.adj,
                    activation=self.activation,
                    num_heads=self.num_heads[i if i < len(self.num_heads) else -1],
                    dropout=self.dropout_ph,
                    curvature_trainable=self.curvature_trainable,
                    initial_curvature=self.curvature,
                    model_size=self.model_size,
                    concat=self.concat_heads,
                    logging=self.logging
                )(current_input1)
                
                current_dim = self.hidden_dims[i]
        
        # Process second set of nodes (positive samples)
        with tf.variable_scope('hgain_layer_1', reuse=True):
            self.layer1_2, _ = HyperbolicGAIN(
                input_dim=self.input_dim,
                output_dim=self.hidden_dims[0],
                adj_mat=self.adj,
                activation=self.activation,
                num_heads=self.num_heads[0],
                dropout=self.dropout_ph,
                curvature_trainable=self.curvature_trainable,
                initial_curvature=self.initial_curvature,
                model_size=self.model_size,
                concat=self.concat_heads,
                logging=self.logging
            )(tf.nn.embedding_lookup(self.features, self.inputs2))
            
        # Process through the network (with shared weights)
        current_input2 = self.layer1_2
        
        for i in range(1, len(self.hidden_dims)):
            with tf.variable_scope(f'hgain_layer_{i+1}', reuse=True):
                current_input2, _ = HyperbolicGAIN(
                    input_dim=current_dim,
                    output_dim=self.hidden_dims[i],
                    adj_mat=self.adj,
                    activation=self.activation,
                    num_heads=self.num_heads[i if i < len(self.num_heads) else -1],
                    dropout=self.dropout_ph,
                    curvature_trainable=self.curvature_trainable,
                    initial_curvature=self.curvature,
                    model_size=self.model_size,
                    concat=self.concat_heads,
                    logging=self.logging
                )(current_input2)
        
        # Sample negative examples
        self.neg_samples = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=tf.reshape(self.inputs2, [-1, 1]),
            num_true=1,
            num_sampled=self.neg_sample_size,
            unique=False,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist()
        )[0]
        
        # Process negative samples
        with tf.variable_scope('hgain_layer_1', reuse=True):
            self.layer1_neg, _ = HyperbolicGAIN(
                input_dim=self.input_dim,
                output_dim=self.hidden_dims[0],
                adj_mat=self.adj,
                activation=self.activation,
                num_heads=self.num_heads[0],
                dropout=self.dropout_ph,
                curvature_trainable=self.curvature_trainable,
                initial_curvature=self.initial_curvature,
                model_size=self.model_size,
                concat=self.concat_heads,
                logging=self.logging
            )(tf.nn.embedding_lookup(self.features, self.neg_samples))
            
        # Process through the network (with shared weights)
        current_input_neg = self.layer1_neg
        
        for i in range(1, len(self.hidden_dims)):
            with tf.variable_scope(f'hgain_layer_{i+1}', reuse=True):
                current_input_neg, _ = HyperbolicGAIN(
                    input_dim=current_dim,
                    output_dim=self.hidden_dims[i],
                    adj_mat=self.adj,
                    activation=self.activation,
                    num_heads=self.num_heads[i if i < len(self.num_heads) else -1],
                    dropout=self.dropout_ph,
                    curvature_trainable=self.curvature_trainable,
                    initial_curvature=self.curvature,
                    model_size=self.model_size,
                    concat=self.concat_heads,
                    logging=self.logging
                )(current_input_neg)
        
        # Store final embeddings
        self.outputs1 = current_input1
        self.outputs2 = current_input2
        self.outputs_neg = current_input_neg
        
        # Compute hyperbolic distance for positive samples
        pos_dist = hutils.hyperbolic_distance(
            tf.reshape(self.outputs1, [-1, current_dim]),
            tf.reshape(self.outputs2, [-1, current_dim]),
            self.curvature
        )
        
        # Compute hyperbolic distance for negative samples
        neg_dist = []
        for i in range(self.neg_sample_size):
            neg_sample = tf.gather(self.outputs_neg, i)
            dist = hutils.hyperbolic_distance(
                tf.reshape(self.outputs1, [-1, current_dim]),
                tf.reshape(neg_sample, [-1, current_dim]),
                self.curvature
            )
            neg_dist.append(dist)
            
        # Stack negative distances
        self.neg_dist = tf.stack(neg_dist, axis=1)
        
        # Prepare for loss calculation
        self.pos_dist = pos_dist
        self.all_dist = tf.concat([tf.expand_dims(self.pos_dist, 1), self.neg_dist], axis=1)
    
    def build(self):
        """Build the model based on supervised/unsupervised setting."""
        if self.is_supervised:
            self._build_supervised()
        else:
            self._build_unsupervised()
        
        # Define loss function
        self._loss()
        self._accuracy()
        
        # Create optimizer operation
        self.opt_op = self.optimizer.minimize(self.loss)
    
    def _loss(self):
        """Define the loss function based on supervised/unsupervised setting."""
        # Apply L2 regularization to all variables
        for var in tf.trainable_variables():
            if 'bias' not in var.name.lower():
                self.loss = self.loss + self.weight_decay * tf.nn.l2_loss(var)
                
        if self.is_supervised:
            # Supervised loss (cross-entropy)
            if self.num_classes > 1:
                # Multi-class classification
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=self.outputs,
                        labels=self.labels
                    )
                )
            else:
                # Binary classification
                self.loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.outputs,
                        labels=self.labels
                    )
                )
        else:
            # Unsupervised loss (based on distances)
            # Normalize distances to [0, 1] range
            max_dist = tf.reduce_max(self.all_dist) + 1e-6
            norm_pos_dist = self.pos_dist / max_dist
            norm_neg_dist = self.neg_dist / max_dist
            
            # Convert distances to probabilities (closer = higher probability)
            pos_probs = tf.exp(-norm_pos_dist)
            neg_probs = tf.exp(-norm_neg_dist)
            
            # Compute log-likelihood using max-margin loss
            self.loss = -tf.reduce_mean(
                tf.log(pos_probs + 1e-6) - 
                tf.log(tf.reduce_sum(neg_probs, axis=1) + pos_probs + 1e-6)
            )
            
        # Log loss value
        tf.summary.scalar('loss', self.loss)
    
    def _accuracy(self):
        """Define the accuracy metric based on supervised/unsupervised setting."""
        if self.is_supervised:
            # For multi-class classification
            if self.num_classes > 1:
                self.predictions = tf.argmax(self.outputs, 1)
                correct_predictions = tf.equal(
                    self.predictions,
                    tf.argmax(self.labels, 1)
                )
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            else:
                # For binary classification
                self.predictions = tf.cast(tf.greater_equal(tf.nn.sigmoid(self.outputs), 0.5), tf.int32)
                correct_predictions = tf.equal(self.predictions, tf.cast(self.labels, tf.int32))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        else:
            # For unsupervised learning, we use Mean Reciprocal Rank (MRR)
            # Compute rankings of positive samples among negative samples
            _, indices_of_ranks = tf.nn.top_k(
                self.all_dist, 
                k=self.neg_sample_size + 1
            )
            _, self.ranks = tf.nn.top_k(
                -indices_of_ranks, 
                k=self.neg_sample_size + 1
            )
            
            # Mean Reciprocal Rank (MRR)
            self.mrr = tf.reduce_mean(
                tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32))
            )
            self.accuracy = self.mrr  # Use MRR as accuracy for unsupervised model
            
        # Log accuracy value
        tf.summary.scalar('accuracy', self.accuracy)
