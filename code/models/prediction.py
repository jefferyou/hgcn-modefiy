import torch
import torch.nn as nn
import torch.nn.functional as F

class BipartiteEdgePredLayer(nn.Module):
    """
    Edge prediction layer for link prediction tasks.
    
    Applies dot product or bilinear form to calculate affinity between nodes.
    Used for skip-gram like unsupervised learning approach.
    """
    def __init__(self, input_dim1, input_dim2, act=torch.sigmoid, loss_fn='xent', 
                 neg_sample_weights=1.0, bias=False, bilinear_weights=False, dropout=0.0):
        """
        Args:
            input_dim1: Input dimension for the first set of features
            input_dim2: Input dimension for the second set of features
            act: Activation function
            loss_fn: Loss function type ('xent', 'skipgram', or 'hinge')
            neg_sample_weights: Weight for negative samples
            bias: Whether to use bias
            bilinear_weights: Whether to use bilinear form for affinity calculation
            dropout: Dropout rate
        """
        super(BipartiteEdgePredLayer, self).__init__()
        
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.act = act
        self.bias = bias
        self.dropout = dropout
        
        # Margin for hinge loss
        self.margin = 0.1
        self.neg_sample_weights = neg_sample_weights
        self.bilinear_weights = bilinear_weights
        
        # Initialize weights for bilinear form
        if bilinear_weights:
            self.weights = nn.Parameter(torch.Tensor(input_dim1, input_dim2))
            nn.init.xavier_uniform_(self.weights)
        
        # Initialize bias
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(1))
        
        # Set loss function
        if loss_fn == 'xent':
            self.loss_fn = self._xent_loss
        elif loss_fn == 'skipgram':
            self.loss_fn = self._skipgram_loss
        elif loss_fn == 'hinge':
            self.loss_fn = self._hinge_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
    
    def affinity(self, inputs1, inputs2):
        """
        Calculate affinity between two sets of inputs
        
        Args:
            inputs1: First input tensor [batch_size, input_dim1]
            inputs2: Second input tensor [batch_size, input_dim2]
            
        Returns:
            Affinity scores [batch_size]
        """
        if self.dropout > 0:
            inputs1 = F.dropout(inputs1, p=self.dropout, training=self.training)
            inputs2 = F.dropout(inputs2, p=self.dropout, training=self.training)
            
        if self.bilinear_weights:
            # Bilinear form: inputs1 * weights * inputs2^T
            prod = torch.mm(inputs2, self.weights.t())
            result = torch.sum(inputs1 * prod, dim=1)
        else:
            # Simple dot product
            result = torch.sum(inputs1 * inputs2, dim=1)
            
        if self.bias:
            result = result + self.bias_param
            
        return result
    
    def neg_cost(self, inputs1, neg_samples):
        """
        Calculate affinity of inputs1 to negative samples
        
        Args:
            inputs1: Input tensor [batch_size, input_dim1]
            neg_samples: Negative samples tensor [num_neg_samples, input_dim2]
            
        Returns:
            Affinity matrix [batch_size, num_neg_samples]
        """
        if self.dropout > 0:
            inputs1 = F.dropout(inputs1, p=self.dropout, training=self.training)
            neg_samples = F.dropout(neg_samples, p=self.dropout, training=self.training)
            
        if self.bilinear_weights:
            inputs1 = torch.mm(inputs1, self.weights)
            
        neg_aff = torch.mm(inputs1, neg_samples.t())
        return neg_aff
    
    def loss(self, inputs1, inputs2, neg_samples):
        """
        Calculate loss for link prediction
        
        Args:
            inputs1: Input tensor [batch_size, input_dim1]
            inputs2: Input tensor [batch_size, input_dim2]
            neg_samples: Negative samples tensor [num_neg_samples, input_dim2]
            
        Returns:
            Loss value
        """
        return self.loss_fn(inputs1, inputs2, neg_samples)
    
    def _xent_loss(self, inputs1, inputs2, neg_samples):
        """
        Cross entropy loss for link prediction
        """
        # Calculate affinity between positive samples
        aff = self.affinity(inputs1, inputs2)
        
        # Calculate affinity between inputs and negative samples
        neg_aff = self.neg_cost(inputs1, neg_samples)
        
        # Cross entropy for positive samples (label=1)
        true_xent = F.binary_cross_entropy_with_logits(
            aff,
            torch.ones_like(aff),
            reduction='sum'
        )
        
        # Cross entropy for negative samples (label=0)
        negative_xent = F.binary_cross_entropy_with_logits(
            neg_aff,
            torch.zeros_like(neg_aff),
            reduction='sum'
        )
        
        loss = true_xent + self.neg_sample_weights * negative_xent
        return loss
    
    def _skipgram_loss(self, inputs1, inputs2, neg_samples):
        """
        Skipgram loss for link prediction (word2vec style)
        """
        # Calculate affinity between positive samples
        aff = self.affinity(inputs1, inputs2)
        
        # Calculate affinity between inputs and negative samples
        neg_aff = self.neg_cost(inputs1, neg_samples)
        
        # Log-sum-exp trick for numerical stability
        neg_cost = torch.log(torch.sum(torch.exp(neg_aff), dim=1))
        
        # Loss is the negative of (positive affinity - negative cost)
        loss = -torch.sum(aff - neg_cost)
        return loss
    
    def _hinge_loss(self, inputs1, inputs2, neg_samples):
        """
        Hinge loss for link prediction
        """
        # Calculate affinity between positive samples
        aff = self.affinity(inputs1, inputs2)
        
        # Calculate affinity between inputs and negative samples
        neg_aff = self.neg_cost(inputs1, neg_samples)
        
        # Hinge loss
        diff = F.relu(neg_aff - aff.unsqueeze(1) + self.margin)
        loss = torch.sum(diff)
        return loss
