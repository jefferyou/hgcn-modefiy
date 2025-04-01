import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.aggregators import (
    MeanAggregator, GCNAggregator, GAINAggregator, GINAggregator,
    MaxPoolingAggregator, AttentionAggregator
)
from models.prediction import BipartiteEdgePredLayer


class SampleAndAggregate(nn.Module):
    """
    Base implementation of GraphSAGE/GAIN for unsupervised learning
    """
    def __init__(self, features, adj, degrees, layer_infos, 
                 concat=True, aggregator_type="mean", model_size="small",
                 identity_dim=0, neg_sample_size=10, device="cpu"):
        """
        Args:
            features: Tensor of node features
            adj: Adjacency information (for node neighbor sampling)
            degrees: Node degrees (for negative sampling)
            layer_infos: List of dicts containing parameters for each layer
            concat: Whether to concatenate or average during aggregation
            aggregator_type: Type of aggregator to use
            model_size: Size of hidden layers
            identity_dim: Set to positive int to use identity features
            neg_sample_size: Number of negative samples to use
            device: Device to run the model on
        """
        super(SampleAndAggregate, self).__init__()
        
        self.device = device
        self.features = features
        self.adj = adj
        self.degrees = degrees
        self.concat = concat
        self.model_size = model_size
        self.neg_sample_size = neg_sample_size
        
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
        
        # Edge prediction layer
        if concat and (len(layer_infos) > 0):
            # For concatenation, the output dim is doubled for each layer
            self.link_pred_layer = BipartiteEdgePredLayer(
                self.dims[-1] * 2, self.dims[-1] * 2, act=torch.sigmoid, bilinear_weights=False
            )
        else:
            self.link_pred_layer = BipartiteEdgePredLayer(
                self.dims[-1], self.dims[-1], act=torch.sigmoid, bilinear_weights=False
            )
        
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
                    dropout=self.layer_infos[layer].get("dropout", 0.),
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
        
    def forward(self, inputs1, inputs2):
        """
        Forward pass of the model
        
        Args:
            inputs1: Indices of source nodes
            inputs2: Indices of target nodes (positive samples)
            
        Returns:
            loss: The unsupervised loss
            scores: Raw scores for link prediction
        """
        # Sample negative examples
        neg_samples = self._get_negative_samples(inputs1, self.neg_sample_size)
        
        # Sample neighbors and aggregate
        samples1, support_sizes1 = self.sample(inputs1, self.layer_infos)
        samples2, support_sizes2 = self.sample(inputs2, self.layer_infos)
        neg_samples_list, neg_support_sizes = self.sample(neg_samples, self.layer_infos, batch_size=self.neg_sample_size)
        
        # Aggregate features for positive and negative samples
        self_outputs1 = self.aggregate(samples1, self.features, len(inputs1))
        self_outputs2 = self.aggregate(samples2, self.features, len(inputs2))
        neg_outputs = self.aggregate(neg_samples_list, self.features, len(neg_samples))
        
        # Normalize embeddings
        self_outputs1 = F.normalize(self_outputs1, p=2, dim=1)
        self_outputs2 = F.normalize(self_outputs2, p=2, dim=1)
        neg_outputs = F.normalize(neg_outputs, p=2, dim=1)
        
        # Calculate loss
        loss = self.link_pred_layer.loss(self_outputs1, self_outputs2, neg_outputs)
        
        # Calculate scores for evaluation
        scores = self.link_pred_layer.affinity(self_outputs1, self_outputs2)
        neg_scores = self.link_pred_layer.neg_cost(self_outputs1, neg_outputs)
        
        return loss, scores, neg_scores, self_outputs1
    
    def _get_negative_samples(self, nodes, num_samples):
        """
        Sample negative nodes based on node degree distribution
        """
        # Convert degrees to probabilities
        degree_probs = self.degrees / np.sum(self.degrees)
        
        # Sample nodes based on degree distribution
        neg_samples = np.random.choice(
            len(self.degrees),
            size=num_samples,
            replace=True,
            p=degree_probs
        )
        
        return neg_samples