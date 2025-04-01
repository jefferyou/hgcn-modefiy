import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

class Aggregator(nn.Module):
    """Base class for aggregators"""
    def __init__(self, input_dim, output_dim, dropout=0., bias=False, act=F.relu, name=None, concat=False):
        super(Aggregator, self).__init__()
        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def forward(self, inputs):
        """
        Override in child classes.
        """
        raise NotImplementedError


class MeanAggregator(Aggregator):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """
    def __init__(self, input_dim, output_dim, dropout=0., bias=False, act=F.relu, name=None, concat=False):
        super(MeanAggregator, self).__init__(input_dim, output_dim, dropout, bias, act, name, concat)
        
        self.neigh_weights = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        self.self_weights = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        
        if self.bias:
            self.bias_param = nn.Parameter(torch.Tensor(self.output_dim))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.neigh_weights)
        nn.init.xavier_uniform_(self.self_weights)
        if self.bias:
            nn.init.zeros_(self.bias_param)
        
    def forward(self, inputs):
        """
        Args:
            inputs: tuple of (self_vectors, neighbor_vectors)
                self_vectors: tensor of shape [num_nodes, input_dim]
                neighbor_vectors: tensor of shape [num_nodes, num_samples, input_dim]
        """
        self_vecs, neigh_vecs = inputs
        
        # Apply dropout
        if self.training and self.dropout > 0:
            self_vecs = F.dropout(self_vecs, p=self.dropout, training=self.training)
            neigh_vecs = F.dropout(neigh_vecs, p=self.dropout, training=self.training)
        
        # Calculate mean of neighbor vectors
        neigh_means = neigh_vecs.mean(dim=1)  # [num_nodes, input_dim]
        
        # Apply weights
        from_neighs = torch.matmul(neigh_means, self.neigh_weights)  # [num_nodes, output_dim]
        from_self = torch.matmul(self_vecs, self.self_weights)  # [num_nodes, output_dim]
        
        if self.concat:
            output = torch.cat([from_self, from_neighs], dim=1)
        else:
            output = from_self + from_neighs
        
        # Apply bias and activation
        if self.bias:
            output = output + self.bias_param
            
        return self.act(output)


class GCNAggregator(Aggregator):
    """
    GCN: Graph Convolutional Network aggregator.
    """
    def __init__(self, input_dim, output_dim, dropout=0., bias=False, act=F.relu, name=None, concat=False):
        super(GCNAggregator, self).__init__(input_dim, output_dim, dropout, bias, act, name, concat)
        
        self.weights = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        
        if self.bias:
            self.bias_param = nn.Parameter(torch.Tensor(self.output_dim))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        if self.bias:
            nn.init.zeros_(self.bias_param)
        
    def forward(self, inputs):
        """
        GCN aggregation: combines self vectors with neighbor vectors using same weights
        """
        self_vecs, neigh_vecs = inputs
        
        # Apply dropout
        if self.training and self.dropout > 0:
            self_vecs = F.dropout(self_vecs, p=self.dropout, training=self.training)
            neigh_vecs = F.dropout(neigh_vecs, p=self.dropout, training=self.training)
        
        # Combine self and neighbor vectors
        extended_self_vecs = self_vecs.unsqueeze(1)  # [num_nodes, 1, input_dim]
        all_vecs = torch.cat([extended_self_vecs, neigh_vecs], dim=1)  # [num_nodes, 1+num_samples, input_dim]
        means = all_vecs.mean(dim=1)  # [num_nodes, input_dim]
        
        # Apply weights
        output = torch.matmul(means, self.weights)  # [num_nodes, output_dim]
        
        # Apply bias and activation
        if self.bias:
            output = output + self.bias_param
            
        return self.act(output)


class GAINAggregator(Aggregator):
    """
    Graph Attention Isomorphism Network (GAIN) Aggregator.
    """
    def __init__(self, input_dim, output_dim, neigh_input_dim=None,
                 transformed_dim=None, dropout=0., bias=False, num_multi_head=1,
                 act=F.relu, name=None, concat=False, model_size="small", **kwargs):
        
        super(GAINAggregator, self).__init__(input_dim, output_dim, dropout, bias, act, name, concat)
        
        self.dropout = dropout
        self.num_multi_head = num_multi_head
        self.bias = bias
        
        # Set dimensionality
        if neigh_input_dim is None:
            neigh_input_dim = input_dim
            
        if transformed_dim is None:
            transformed_dim = output_dim
        self.transformed_dim = transformed_dim
        
        # Set size of hidden layers in MLP
        if model_size == "small":
            hidden_dim = 128
        elif model_size == "big":
            hidden_dim = 256
        self.hidden_dim = hidden_dim
        
        # Non-linearity for attention
        self.non_linearity = False
        
        # Multi-head attention
        self.multi_head = False
        
        # Learnable epsilon parameter (default initialized to 0.5)
        self.epsilon = nn.Parameter(torch.tensor(0.5))
        
        # MLP for aggregation
        self.mlp = nn.Sequential(
            nn.Linear(transformed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention weights
        self.transform_weights = nn.ModuleList()
        self.attention_weights = nn.ModuleList()
        
        for _ in range(self.num_multi_head):
            # Feature transformation weights
            self.transform_weights.append(nn.Linear(input_dim, transformed_dim, bias=False))
            
            # Attention mechanism weights
            self.attention_weights.append(nn.Linear(2 * transformed_dim, 1, bias=False))
            
        # Multi-head attention weights
        if self.multi_head:
            self.neigh_weights = nn.Linear(transformed_dim, output_dim, bias=False)
            self.self_weights = nn.Linear(input_dim, output_dim, bias=False)
                
        self.reset_parameters()
        
    def reset_parameters(self):
        # Reset parameters for all sub-modules
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, inputs):
        """
        GAINAggregator forward pass
        """
        self_vecs, neigh_vecs = inputs
        
        # Apply dropout
        if self.training and self.dropout > 0:
            self_vecs = F.dropout(self_vecs, p=self.dropout, training=self.training)
            neigh_vecs = F.dropout(neigh_vecs, p=self.dropout, training=self.training)
        
        # Get shapes
        num_nodes, num_samples, _ = neigh_vecs.size()
        
        # Store aggregated vectors for all attention heads
        neigh_aggregated_heads = []
        
        for i in range(self.num_multi_head):
            # Transform neighbor vectors
            neigh_transformed = self.transform_weights[i](neigh_vecs.view(-1, self.input_dim))
            neigh_transformed = neigh_transformed.view(num_nodes, num_samples, self.transformed_dim)
            
            # Transform self vectors
            self_transformed = self.transform_weights[i](self_vecs)
            
            # Repeat self vectors for each neighbor
            self_transformed_tiled = self_transformed.unsqueeze(1).repeat(1, num_samples, 1)
            
            # Concatenate self and neighbor vectors for attention
            concat_transformed = torch.cat([self_transformed_tiled, neigh_transformed], dim=2)
            
            # Calculate attention scores
            attention_scores = self.attention_weights[i](concat_transformed.view(-1, 2 * self.transformed_dim))
            attention_scores = F.leaky_relu(attention_scores).view(num_nodes, num_samples)
            
            # Apply softmax to get attention weights
            attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(2)
            
            # Apply attention weights to transformed neighbor vectors
            neigh_weighted = neigh_transformed * attention_weights
            
            # Sum weighted neighbors
            neigh_aggregated = neigh_weighted.sum(dim=1)
            
            neigh_aggregated_heads.append(neigh_aggregated)
        
        # Combine multiple attention heads
        if len(neigh_aggregated_heads) > 1:
            stacked_heads = torch.stack(neigh_aggregated_heads, dim=1)
            final_neigh_aggregated = stacked_heads.sum(dim=1)
        else:
            final_neigh_aggregated = neigh_aggregated_heads[0]
        
        # Apply non-linearity if specified
        if self.non_linearity:
            final_neigh_aggregated = F.elu(final_neigh_aggregated)
        
        if self.multi_head:
            # Apply different weights for self and neighbor vectors
            from_neighs = self.neigh_weights(final_neigh_aggregated)
            from_self = self.self_weights(self_vecs)
            output = (1 + self.epsilon) * from_self + from_neighs
        else:
            # Use transformed vectors directly
            output = (1 + self.epsilon) * self_transformed + final_neigh_aggregated
        
        # Apply MLP
        output = self.mlp(output)
        
        return output


class GINAggregator(Aggregator):
    """
    Graph Isomorphism Network (GIN) Aggregator.
    """
    def __init__(self, input_dim, output_dim, dropout=0., bias=False, act=F.relu, name=None, concat=False, model_size="small", **kwargs):
        super(GINAggregator, self).__init__(input_dim, output_dim, dropout, bias, act, name, concat)
        
        # Set size of hidden layers in MLP
        if model_size == "small":
            hidden_dim = 128
        elif model_size == "big":
            hidden_dim = 256
        self.hidden_dim = hidden_dim
        
        # Learnable epsilon parameter (default initialized to 0.5)
        self.epsilon = nn.Parameter(torch.tensor(0.5))
        
        # MLP for feature transformation
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Reset parameters for all sub-modules
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, inputs):
        """
        GIN forward pass
        """
        self_vecs, neigh_vecs = inputs
        
        # Apply dropout
        if self.training and self.dropout > 0:
            self_vecs = F.dropout(self_vecs, p=self.dropout, training=self.training)
            neigh_vecs = F.dropout(neigh_vecs, p=self.dropout, training=self.training)
        
        # Sum neighbor vectors
        neigh_sum = neigh_vecs.sum(dim=1)
        
        # Combine self and neighbor vectors with epsilon
        combined = (1 + self.epsilon) * self_vecs + neigh_sum
        
        # Apply MLP
        output = self.mlp(combined)
        
        return output


class MaxPoolingAggregator(Aggregator):
    """
    Aggregates via max-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None, 
                 dropout=0., bias=False, act=F.relu, name=None, concat=False, **kwargs):
        super(MaxPoolingAggregator, self).__init__(input_dim, output_dim, dropout, bias, act, name, concat)
        
        # Set dimensionality
        if neigh_input_dim is None:
            neigh_input_dim = input_dim
        self.neigh_input_dim = neigh_input_dim
        
        # Set size of hidden layers in MLP
        if model_size == "small":
            hidden_dim = 512
        elif model_size == "big":
            hidden_dim = 1024
        self.hidden_dim = hidden_dim
        
        # MLP for neighbor transformation
        self.neigh_mlp = nn.Sequential(
            nn.Linear(neigh_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Weights for self and transformed neighbors
        self.neigh_weights = nn.Linear(hidden_dim, output_dim, bias=False)
        self.self_weights = nn.Linear(input_dim, output_dim, bias=False)
        
        if self.bias:
            self.bias_param = nn.Parameter(torch.Tensor(self.output_dim))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Reset parameters for all sub-modules
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        if self.bias:
            nn.init.zeros_(self.bias_param)
    
    def forward(self, inputs):
        """
        MaxPooling forward pass
        """
        self_vecs, neigh_vecs = inputs
        
        # Get shapes
        num_nodes, num_samples, _ = neigh_vecs.size()
        
        # Apply MLP to each neighbor vector
        neigh_h = self.neigh_mlp(neigh_vecs.view(-1, self.neigh_input_dim))
        neigh_h = neigh_h.view(num_nodes, num_samples, self.hidden_dim)
        
        # Max pooling across neighbors
        pooled = torch.max(neigh_h, dim=1)[0]  # [num_nodes, hidden_dim]
        
        # Apply weights
        from_neighs = self.neigh_weights(pooled)  # [num_nodes, output_dim]
        from_self = self.self_weights(self_vecs)  # [num_nodes, output_dim]
        
        if self.concat:
            output = torch.cat([from_self, from_neighs], dim=1)
        else:
            output = from_self + from_neighs
        
        # Apply bias and activation
        if self.bias:
            output = output + self.bias_param
            
        return self.act(output)


class AttentionAggregator(Aggregator):
    """
    Graph Attention Network (GAT) Aggregator.
    """
    def __init__(self, input_dim, output_dim, transformed_dim=None, num_multi_head=1,
                 neigh_input_dim=None, dropout=0., bias=False, act=F.relu, name=None, concat=False, **kwargs):
        super(AttentionAggregator, self).__init__(input_dim, output_dim, dropout, bias, act, name, concat)
        
        # Set dimensionality
        if neigh_input_dim is None:
            neigh_input_dim = input_dim
        
        if transformed_dim is None:
            transformed_dim = input_dim
        self.transformed_dim = transformed_dim
        
        self.num_multi_head = num_multi_head
        
        # Transformation and attention weights
        self.transform_weights = nn.ModuleList()
        self.attention_weights = nn.ModuleList()
        
        for _ in range(self.num_multi_head):
            # Feature transformation weights
            self.transform_weights.append(nn.Linear(input_dim, transformed_dim, bias=False))
            
            # Attention mechanism weights
            self.attention_weights.append(nn.Linear(2 * transformed_dim, 1, bias=False))
        
        # Output weights for self and neighbor vectors
        self.neigh_weights = nn.Linear(transformed_dim, output_dim, bias=False)
        self.self_weights = nn.Linear(input_dim, output_dim, bias=False)
        
        if self.bias:
            self.bias_param = nn.Parameter(torch.Tensor(self.output_dim))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Reset parameters for all sub-modules
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        if self.bias:
            nn.init.zeros_(self.bias_param)
    
    def forward(self, inputs):
        """
        GAT forward pass
        """
        self_vecs, neigh_vecs = inputs
        
        # Apply dropout
        if self.training and self.dropout > 0:
            self_vecs = F.dropout(self_vecs, p=self.dropout, training=self.training)
            neigh_vecs = F.dropout(neigh_vecs, p=self.dropout, training=self.training)
        
        # Get shapes
        num_nodes, num_samples, _ = neigh_vecs.size()
        
        # Store aggregated vectors for all attention heads
        neigh_aggregated_heads = []
        
        for i in range(self.num_multi_head):
            # Transform neighbor vectors
            neigh_transformed = self.transform_weights[i](neigh_vecs.view(-1, self.input_dim))
            neigh_transformed = neigh_transformed.view(num_nodes, num_samples, self.transformed_dim)
            
            # Transform self vectors
            self_transformed = self.transform_weights[i](self_vecs)
            
            # Repeat self vectors for each neighbor
            self_transformed_tiled = self_transformed.unsqueeze(1).repeat(1, num_samples, 1)
            
            # Concatenate self and neighbor vectors for attention
            concat_transformed = torch.cat([self_transformed_tiled, neigh_transformed], dim=2)
            
            # Calculate attention scores
            attention_scores = self.attention_weights[i](concat_transformed.view(-1, 2 * self.transformed_dim))
            attention_scores = F.leaky_relu(attention_scores).view(num_nodes, num_samples)
            
            # Apply softmax to get attention weights
            attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(2)
            
            # Apply attention weights to transformed neighbor vectors
            neigh_weighted = neigh_transformed * attention_weights
            
            # Sum weighted neighbors
            neigh_aggregated = neigh_weighted.sum(dim=1)
            
            neigh_aggregated_heads.append(neigh_aggregated)
        
        # Combine multiple attention heads (average)
        if len(neigh_aggregated_heads) > 1:
            stacked_heads = torch.stack(neigh_aggregated_heads, dim=1)
            final_neigh_aggregated = stacked_heads.mean(dim=1)
        else:
            final_neigh_aggregated = neigh_aggregated_heads[0]
        
        # Apply non-linearity
        final_neigh_aggregated = F.elu(final_neigh_aggregated)
        
        # Apply output weights
        from_neighs = self.neigh_weights(final_neigh_aggregated)
        from_self = self.self_weights(self_vecs)
        
        if self.concat:
            output = torch.cat([from_self, from_neighs], dim=1)
        else:
            output = from_self + from_neighs
        
        # Apply bias and activation
        if self.bias:
            output = output + self.bias_param
            
        return self.act(output)
