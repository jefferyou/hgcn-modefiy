import torch
import torch.nn as nn
import torch.nn.functional as F

from models.aggregators import (
    MeanAggregator, GCNAggregator, GAINAggregator, GINAggregator,
    MaxPoolingAggregator, AttentionAggregator
)


class SupervisedGraphsage(nn.Module):
    """
    Implementation of supervised GraphSAGE/GAIN for node classification
    """
    def __init__(self, num_classes, features, adj, degrees, layer_infos,
                 concat=True, aggregator_type="mean", model_size="small", 
                 sigmoid_loss=False, identity_dim=0, dropout=0.0, device="cpu"):
        """
        Args:
            num_classes: Number of output classes
            features: Tensor of node features
            adj: Adjacency information (for node neighbor sampling)
            degrees: Node degrees (for random walk sampling)
            layer_infos: List of dicts containing parameters for each layer
            concat: Whether to concatenate or average during aggregation
            aggregator_type: Type of aggregator to use
            model_size: Size of hidden layers
            sigmoid_loss: Whether to use sigmoid loss (multi-label) or softmax (multi-class)
            identity_dim: Set to positive int to use identity features
            dropout: Dropout rate
            device: Device to run the model on
        """
        super(SupervisedGraphsage, self).__init__()
        
        self.device = device
        self.features = features
        self.adj = adj
        self.degrees = degrees
        self.concat = concat
        self.model_size = model_size
        self.sigmoid_loss = sigmoid_loss
        self.dropout = dropout
        self.num_classes = num_classes
        
        # Get input dimension
        if features is None:
            if identity_dim == 0:
                raise ValueError("Must have positive identity feature dimension if no input features given")
            self.num_features = identity_dim
        else:
            self.num_features = features.shape[1]
        
        # Handle identity mappings
        if identity_dim > 0:
            self.embeds = nn.Parameter(torch.empty(adj.shape[0], identity_dim))
            nn.init.xavier_uniform_(self.embeds)
            if features is not None:
                self.features = torch.cat([self.embeds, self.features], dim=1)
                self.num_features = self.num_features + identity_dim
            else:
                self.features = self.embeds
        
        # Layer dimensions
        self.dims = [self.num_features]
        self.dims.extend([layer_info["output_dim"] for layer_info in layer_infos])
        
        self.layer_infos = layer_infos
        
        # Set up aggregators for each layer
        self.aggregators = self._build_aggregators(aggregator_type)
        
        # Node prediction layer
        if concat and (len(layer_infos) > 0):
            # For concatenation, the output dim is doubled
            final_dim = self.dims[-1] * 2
        else:
            final_dim = self.dims[-1]
        
        self.node_pred = nn.Linear(final_dim, num_classes)
        
    def _build_aggregators(self, aggregator_type):
        """
        Create aggregator modules for each layer
        """
        aggregators = nn.ModuleList()
        
        for layer in range(len(self.layer_infos)):
            # For the first layer, input dimension is the original feature dimension
            # For subsequent layers, it depends on whether concatenation is used
            dim_mult = 2 if self.concat and (layer != 0) else 1
            
            # Choose correct aggregator type
            if aggregator_type == "mean":
                aggregator_cls = MeanAggregator
            elif aggregator_type == "gcn":
                aggregator_cls = GCNAggregator
            elif aggregator_type == "maxpool":
                aggregator_cls = MaxPoolingAggregator
            elif aggregator_type == "gain":
                aggregator_cls = GAINAggregator
            elif aggregator_type == "gin":
                aggregator_cls = GINAggregator
            elif aggregator_type == "attn":
                aggregator_cls = AttentionAggregator
            else:
                raise ValueError(f"Unknown aggregator: {aggregator_type}")
            
            if layer == len(self.layer_infos) - 1:
                # Last layer has no activation
                act = lambda x: x
            else:
                act = F.relu
            
            aggregators.append(
                aggregator_cls(
                    input_dim=self.dims[layer] * dim_mult,
                    output_dim=self.dims[layer+1],
                    dropout=self.layer_infos[layer].get("dropout", self.dropout),
                    bias=self.layer_infos[layer].get("bias", False),
                    act=act,
                    concat=self.concat,
                    model_size=self.model_size
                ).to(self.device)
            )
            
        return aggregators
    
    def sample(self, nodes, layer_infos, batch_size=None):
        """
        Sample neighbors to be the supportive fields for multi-layer aggregation.
        
        Args:
            nodes: List of node indices to start with
            layer_infos: List of layer configuration dicts
            batch_size: Number of nodes in the batch
            
        Returns:
            samples: List of sampled nodes for each layer
            support_sizes: List of support sizes for each layer
        """
        if batch_size is None:
            batch_size = len(nodes)
            
        samples = [nodes]
        support_size = 1
        support_sizes = [support_size]
        
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1  # Work backwards from deepest layer
            support_size *= layer_infos[t]["num_samples"]
            
            # Sample neighbors for the current layer's nodes
            node_supports = []
            for node in samples[k]:
                # Get neighbors of the node
                if node < len(self.adj) and len(self.adj[node]) > 0:
                    neighbors = self.adj[node]
                    # Sample num_samples neighbors
                    if len(neighbors) >= layer_infos[t]["num_samples"]:
                        node_supports.extend(np.random.choice(neighbors, layer_infos[t]["num_samples"], replace=False))
                    else:
                        # If not enough neighbors, sample with replacement
                        node_supports.extend(np.random.choice(neighbors, layer_infos[t]["num_samples"], replace=True))
                else:
                    # If no neighbors, use random nodes from the graph
                    node_supports.extend(np.random.choice(len(self.adj), layer_infos[t]["num_samples"], replace=False))
            
            samples.append(node_supports)
            support_sizes.append(support_size)
            
        return samples, support_sizes
    
    def aggregate(self, samples, input_features, batch_size=None):
        """
        At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
        at next layer.
        
        Args:
            samples: List of sampled nodes for each layer
            input_features: Input features for all nodes
            batch_size: Number of nodes in the batch
            
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """
        if batch_size is None:
            batch_size = len(samples[0])
            
        # Get initial hidden representations
        hidden = [input_features[torch.tensor(sample_list, dtype=torch.long, device=self.device)] 
                  for sample_list in samples]
        
        for layer in range(len(self.layer_infos)):
            next_hidden = []
            dim_mult = 2 if self.concat and (layer != 0) else 1
            
            for hop in range(len(self.layer_infos) - layer):
                neigh_dims = [
                    batch_size * self.layer_infos[hop]["num_samples"] ** hop,
                    self.layer_infos[len(self.layer_infos) - hop - 1]["num_samples"],
                    dim_mult * self.dims[layer]
                ]
                
                # Reshape for aggregation
                h_self = hidden[hop].view(-1, dim_mult * self.dims[layer])
                h_neigh = hidden[hop + 1].view(neigh_dims)
                
                # Perform aggregation
                h = self.aggregators[layer]((h_self, h_neigh))
                next_hidden.append(h)
                
            hidden = next_hidden
            
        return hidden[0]
        
    def forward(self, nodes):
        """
        Forward pass for node classification
        
        Args:
            nodes: Node indices to classify
            
        Returns:
            Logits for node classification
        """
        # Sample neighbors and aggregate
        samples, support_sizes = self.sample(nodes, self.layer_infos)
        
        # Aggregate features
        node_features = self.aggregate(samples, self.features, len(nodes))
        
        # Normalize embeddings
        node_features = F.normalize(node_features, p=2, dim=1)
        
        # Apply prediction layer
        if self.sigmoid_loss:
            # For multi-label classification
            logits = self.node_pred(node_features)
        else:
            # For multi-class classification
            logits = self.node_pred(node_features)
            
        return logits
    
    def predict(self, logits):
        """
        Convert logits to predictions
        
        Args:
            logits: Output logits from forward pass
            
        Returns:
            Predictions (probabilities)
        """
        if self.sigmoid_loss:
            return torch.sigmoid(logits)
        else:
            return F.softmax(logits, dim=1)
    
    def loss(self, logits, labels):
        """
        Calculate loss
        
        Args:
            logits: Output logits from forward pass
            labels: Ground truth labels
            
        Returns:
            Loss value
        """
        if self.sigmoid_loss:
            return F.binary_cross_entropy_with_logits(logits, labels)
        else:
            return F.cross_entropy(logits, labels)
