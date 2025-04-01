"""PyTorch implementation of SHGNN with Hyperbolic Convolution Layer for DRSD task."""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import scipy.sparse as sp
from torch.nn.modules.module import Module
import math
from collections import Counter

# 导入HGCN中的双曲几何相关组件
import manifolds
from layers.hyp_layers import HyperbolicGraphConvolution
from manifolds.base import ManifoldParameter
from manifolds.poincare import PoincareBall
from utils.math_utils import artanh, tanh

# Helper function to convert sparse matrix to torch sparse tensor (from HGCN)
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# Helper function for setting up random seeds
def seed_setup(seed):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class SectorWiseAgg(Module):
    """
    Sector-wise aggregation layer that partitions the graph based on angular sectors.
    """
    def __init__(self, g, in_dim, out_dim, num_sect, rotation, drop_rate, device='cpu'):
        super(SectorWiseAgg, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_sect = num_sect
        self.g = g
        self.drop_rate = drop_rate
        self.device = device

        # Linear transformations
        self.linear_self = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_sect_list = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(self.num_sect)
        ])

        # Parameters for interaction mechanisms
        self.sect_WC = nn.Parameter(torch.Tensor(out_dim, out_dim))
        self.sect_WD = nn.Parameter(torch.Tensor(out_dim, out_dim))
        self.att_gate = nn.Linear(2*(num_sect+1)*out_dim, 1)

        # Initialize parameters with xavier uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        nn.init.xavier_uniform_(self.sect_WC)
        nn.init.xavier_uniform_(self.sect_WD)

        # Compute sector-based subgraphs
        self.sector_subgraphs = self.sector_partition(rotation)

        # Calculate normalization factors
        self.compute_norm()

        # Dropout layer
        self.dropout = nn.Dropout(drop_rate)
        self.sigmoid = nn.Sigmoid()

    def to(self, device):
        """Move module to specified device."""
        self.device = device

        # Move all parameters and buffers to device
        for param in self.parameters():
            param.data = param.data.to(device)

        # Move all tensor attributes
        if hasattr(self, 'out_degree_norm'):
            self.out_degree_norm = self.out_degree_norm.to(device)
            self.in_degree_norm = self.in_degree_norm.to(device)

        # Move all sector subgraphs
        for i in range(len(self.sector_subgraphs)):
            if isinstance(self.sector_subgraphs[i], torch.Tensor):
                self.sector_subgraphs[i] = self.sector_subgraphs[i].to(device)

        return self

    def compute_norm(self):
        """Compute degree-based normalization factors."""
        # Calculate out and in degrees for normalization
        out_degree = torch.FloatTensor([self.g.out_degree(n) for n in range(self.g.number_of_nodes())])
        in_degree = torch.FloatTensor([self.g.in_degree(n) for n in range(self.g.number_of_nodes())])

        # Clamp to avoid division by zero and compute normalization
        self.out_degree_norm = torch.pow(out_degree.clamp(min=1.0), -0.5).view(-1, 1)
        self.in_degree_norm = torch.pow(in_degree.clamp(min=1.0), -0.5).view(-1, 1)

        # Move to the appropriate device if needed
        if self.device != 'cpu':
            self.out_degree_norm = self.out_degree_norm.to(self.device)
            self.in_degree_norm = self.in_degree_norm.to(self.device)

    def sector_partition(self, rotation):
        """
        Partition the graph into sectors based on angles between connected nodes.
        Returns a list of sparse adjacency matrices for each sector.
        """
        # Get node spatial coordinates
        node_x = nx.get_node_attributes(self.g, 'node_x')
        node_y = nx.get_node_attributes(self.g, 'node_y')

        # Initialize sector adjacency matrices
        num_nodes = self.g.number_of_nodes()
        sector_adjs = [sp.lil_matrix((num_nodes, num_nodes)) for _ in range(self.num_sect)]

        # Process each edge to determine its sector
        for src, dst in self.g.edges():
            src_x, src_y = node_x[src], node_y[src]
            dst_x, dst_y = node_x[dst], node_y[dst]

            delta_x = float(src_x - dst_x)
            delta_y = float(src_y - dst_y)

            # Handle special cases to avoid division by zero
            if delta_x == 0:
                if delta_y > 0:
                    delta_x = 1e-4
                elif delta_y < 0:
                    delta_x = -1e-4
            if delta_y == 0:
                if delta_x > 0:
                    delta_y = -1e-4
                elif delta_x < 0:
                    delta_y = 1e-4

            # Skip self-loops (zero distance)
            if delta_x == 0 and delta_y == 0:
                continue

            # Calculate angle in radians
            angle = math.atan2(delta_y, delta_x)
            if angle < 0:
                angle += math.pi
            if delta_y < 0:
                angle += math.pi

            # Apply rotation if specified
            if rotation != 0:
                rotation_rad = float(rotation) / 180 * math.pi
                angle = angle - rotation_rad
                if angle < 0 and angle >= -rotation_rad:
                    angle += 2*math.pi

            # Assign to the appropriate sector
            sec_id = int(angle / (2 * math.pi / self.num_sect))
            if sec_id >= self.num_sect:
                sec_id = 0

            # Add edge to the corresponding sector adjacency matrix
            sector_adjs[sec_id][src, dst] = 1.0

        # Convert to sparse tensors for efficient computation
        result = []
        for adj in sector_adjs:
            adj = adj.tocsr()
            adj_tensor = sparse_mx_to_torch_sparse_tensor(adj)
            result.append(adj_tensor)

        return result

    def forward_sector_agg(self, x):
        """Forward pass of sector-wise aggregation."""
        # Self transformation
        z_self_sect = self.linear_self(x)

        # Sector-wise aggregation
        z_list_sect = []
        for sect_id in range(self.num_sect):
            adj = self.sector_subgraphs[sect_id]
            h = self.linear_sect_list[sect_id](x) * self.out_degree_norm

            # Message passing
            if adj.is_sparse:
                z_sect = torch.sparse.mm(adj, h)
            else:
                z_sect = torch.mm(adj, h)

            z_list_sect.append(z_sect)

        # Concatenate all sector representations
        z_list = [z_self_sect] + z_list_sect
        z = torch.cat(z_list, dim=1)

        # Apply in-degree normalization
        z = z * self.in_degree_norm

        return z

    def heter_sen_interaction(self, z):
        """Heterogeneous sensitivity interaction mechanism."""
        # Reshape to separate sectors
        batch_size = z.shape[0]
        z = z.view(batch_size, self.num_sect+1, self.out_dim)

        # Commonality kernel function
        z_hat_com = torch.matmul(z, self.sect_WC)
        com_score = torch.bmm(z_hat_com, z_hat_com.transpose(1, 2))
        alpha_com = F.softmax(com_score, dim=2)
        z_com = torch.bmm(alpha_com, z_hat_com)
        z_com = z_com.view(batch_size, -1)

        # Discrepancy kernel function
        z_hat_dis = torch.matmul(z, self.sect_WD)
        z_hat_dis_sub = z_hat_dis.unsqueeze(2) - z_hat_dis.unsqueeze(1)
        dis_score = torch.matmul(z_hat_dis.unsqueeze(2), z_hat_dis_sub.transpose(2, 3))
        alpha_dis = F.softmax(dis_score, dim=3)
        z_dis = torch.matmul(alpha_dis, z_hat_dis_sub).squeeze(2)
        z_dis = z_dis.view(batch_size, -1)

        # Attentive component selection
        z_com_dis_cat = torch.cat([z_com, z_dis], dim=-1)
        beta_sect = self.att_gate(z_com_dis_cat)
        beta_sect = self.sigmoid(beta_sect)
        z_wave = beta_sect * z_com + (1-beta_sect) * z_dis

        return z_wave

    def forward(self, x):
        """Forward pass of the sector-wise aggregation layer."""
        x = self.dropout(x)
        z = self.forward_sector_agg(x)
        z = self.heter_sen_interaction(z)
        return z


class SectorWiseAgg_RotateMHead(Module):
    """
    Multi-head sector-wise aggregation with different rotation angles.
    """
    def __init__(self, g, in_dim, out_dim, num_sect, rotation, head_sect, drop_rate, device='cpu'):
        super(SectorWiseAgg_RotateMHead, self).__init__()
        self.head_sect = head_sect
        self.device = device

        # Create different rotations for each head
        list_of_rotation = [rotation*i for i in range(head_sect)]

        # Create multiple sector-wise aggregation layers with different rotations
        self.sector_wise_agg_list = nn.ModuleList()
        for rotation_ in list_of_rotation:
            self.sector_wise_agg_list.append(
                SectorWiseAgg(g, in_dim, out_dim, num_sect, rotation_, drop_rate, device)
            )

    def to(self, device):
        """Move module to specified device."""
        self.device = device
        for i in range(len(self.sector_wise_agg_list)):
            self.sector_wise_agg_list[i] = self.sector_wise_agg_list[i].to(device)
        return self

    def forward(self, x):
        """Forward pass aggregating across all rotated sector heads."""
        z_list = []
        for i in range(self.head_sect):
            z_list.append(self.sector_wise_agg_list[i](x))
        h = torch.cat(z_list, dim=1)
        return h


class RingWiseAgg(Module):
    """
    Ring-wise aggregation layer that partitions the graph based on distance rings.
    """
    def __init__(self, g, in_dim, out_dim, num_ring, distance_list, drop_rate, device='cpu'):
        super(RingWiseAgg, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_ring = num_ring
        self.g = g
        self.device = device

        # Linear transformations
        self.linear_self = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_ring_list = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(self.num_ring)
        ])

        # Parameters for interaction mechanisms
        self.ring_WC = nn.Parameter(torch.Tensor(out_dim, out_dim))
        self.ring_WD = nn.Parameter(torch.Tensor(out_dim, out_dim))
        self.att_gate = nn.Linear(2*(num_ring+1)*out_dim, 1)

        # Initialize parameters
        nn.init.xavier_uniform_(self.ring_WC)
        nn.init.xavier_uniform_(self.ring_WD)

        # Initialize all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        # Compute ring-based subgraphs
        self.ring_subgraphs = self.ring_partition(distance_list)

        # Calculate normalization factors
        self.compute_norm()

        # Dropout layer
        self.dropout = nn.Dropout(drop_rate)
        self.sigmoid = nn.Sigmoid()

    def to(self, device):
        """Move module to specified device."""
        self.device = device

        # Move all parameters and buffers to device
        for param in self.parameters():
            param.data = param.data.to(device)

        # Move all tensor attributes
        if hasattr(self, 'out_degree_norm'):
            self.out_degree_norm = self.out_degree_norm.to(device)
            self.in_degree_norm = self.in_degree_norm.to(device)

        # Move all ring subgraphs
        for i in range(len(self.ring_subgraphs)):
            if isinstance(self.ring_subgraphs[i], torch.Tensor):
                self.ring_subgraphs[i] = self.ring_subgraphs[i].to(device)

        return self

    def compute_norm(self):
        """Compute degree-based normalization factors."""
        # Calculate out and in degrees for normalization
        out_degree = torch.FloatTensor([self.g.out_degree(n) for n in range(self.g.number_of_nodes())])
        in_degree = torch.FloatTensor([self.g.in_degree(n) for n in range(self.g.number_of_nodes())])

        # Clamp to avoid division by zero and compute normalization
        self.out_degree_norm = torch.pow(out_degree.clamp(min=1.0), -0.5).view(-1, 1)
        self.in_degree_norm = torch.pow(in_degree.clamp(min=1.0), -0.5).view(-1, 1)

        # Move to the appropriate device if needed
        if self.device != 'cpu':
            self.out_degree_norm = self.out_degree_norm.to(self.device)
            self.in_degree_norm = self.in_degree_norm.to(self.device)

    def ring_partition(self, distance_list):
        """
        Partition the graph into rings based on distances between connected nodes.
        Returns a list of sparse adjacency matrices for each ring.
        """
        # Ensure the number of rings matches the distance list
        assert self.num_ring == len(distance_list)

        # Add a large number to handle the upper bound of the last ring
        distance_list_pad = distance_list + [999]

        # Get edge distances
        edge_len = nx.get_edge_attributes(self.g, 'edge_len')

        # Initialize ring adjacency matrices
        num_nodes = self.g.number_of_nodes()
        ring_adjs = [sp.lil_matrix((num_nodes, num_nodes)) for _ in range(self.num_ring)]

        # Process each edge to determine its ring
        for src, dst in self.g.edges():
            distance = edge_len.get((src, dst), 0)

            # Skip self-loops or edges with no distance
            if distance == 0:
                continue

            # Determine the ring based on distance thresholds
            for i in range(self.num_ring):
                if distance >= distance_list_pad[i]*1000 and distance < distance_list_pad[i+1]*1000:
                    ring_adjs[i][src, dst] = 1.0
                    break

        # Convert to sparse tensors for efficient computation
        result = []
        for adj in ring_adjs:
            adj = adj.tocsr()
            adj_tensor = sparse_mx_to_torch_sparse_tensor(adj)
            result.append(adj_tensor)

        return result

    def forward_ring_agg(self, x):
        """Forward pass of ring-wise aggregation."""
        # Self transformation
        z_self_ring = self.linear_self(x)

        # Ring-wise aggregation
        z_list_ring = []
        for ring_id in range(self.num_ring):
            adj = self.ring_subgraphs[ring_id]
            h = self.linear_ring_list[ring_id](x) * self.out_degree_norm

            # Message passing
            if adj.is_sparse:
                z_ring = torch.sparse.mm(adj, h)
            else:
                z_ring = torch.mm(adj, h)

            z_list_ring.append(z_ring)

        # Concatenate all ring representations
        z_list = [z_self_ring] + z_list_ring
        z = torch.cat(z_list, dim=1)

        # Apply in-degree normalization
        z = z * self.in_degree_norm

        return z

    def heter_sen_interaction(self, z):
        """Heterogeneous sensitivity interaction mechanism."""
        # Reshape to separate rings
        batch_size = z.shape[0]
        z = z.view(batch_size, self.num_ring+1, self.out_dim)

        # Commonality kernel function
        z_hat_com = torch.matmul(z, self.ring_WC)
        com_score = torch.bmm(z_hat_com, z_hat_com.transpose(1, 2))
        alpha_com = F.softmax(com_score, dim=2)
        z_com = torch.bmm(alpha_com, z_hat_com)
        z_com = z_com.view(batch_size, -1)

        # Discrepancy kernel function
        z_hat_dis = torch.matmul(z, self.ring_WD)
        z_hat_dis_sub = z_hat_dis.unsqueeze(2) - z_hat_dis.unsqueeze(1)
        dis_score = torch.matmul(z_hat_dis.unsqueeze(2), z_hat_dis_sub.transpose(2, 3))
        alpha_dis = F.softmax(dis_score, dim=3)
        z_dis = torch.matmul(alpha_dis, z_hat_dis_sub).squeeze(2)
        z_dis = z_dis.view(batch_size, -1)

        # Attentive component selection
        z_com_dis_cat = torch.cat([z_com, z_dis], dim=-1)
        beta_ring = self.att_gate(z_com_dis_cat)
        beta_ring = self.sigmoid(beta_ring)
        z_wave = beta_ring * z_com + (1-beta_ring) * z_dis

        return z_wave

    def forward(self, x):
        """Forward pass of the ring-wise aggregation layer."""
        x = self.dropout(x)
        z = self.forward_ring_agg(x)
        z = self.heter_sen_interaction(z)
        return z


class RingWiseAgg_ScaleMHead(Module):
    """
    Multi-head ring-wise aggregation with different distance scales.
    """
    def __init__(self, g, in_dim, out_dim, num_ring, bucket_interval, head_ring, drop_rate, device='cpu'):
        super(RingWiseAgg_ScaleMHead, self).__init__()
        self.head_ring = head_ring
        self.device = device

        # Parse bucket interval string to list of floats
        bucket_interval = [float(interval_) for interval_ in bucket_interval.split(',')]
        assert len(bucket_interval) == head_ring, "len(bucket_interval) != head_ring"

        # Create different distance lists for each head
        list_of_distance_list = []
        for interval_ in bucket_interval:
            distance_list = [interval_*i for i in range(num_ring)]
            list_of_distance_list.append(distance_list)

        # Create multiple ring-wise aggregation layers with different distance scales
        self.ring_wise_agg_list = nn.ModuleList()
        for distance_list in list_of_distance_list:
            self.ring_wise_agg_list.append(
                RingWiseAgg(g, in_dim, out_dim, num_ring, distance_list, drop_rate, device)
            )

    def to(self, device):
        """Move module to specified device."""
        self.device = device
        for i in range(len(self.ring_wise_agg_list)):
            self.ring_wise_agg_list[i] = self.ring_wise_agg_list[i].to(device)
        return self

    def forward(self, x):
        """Forward pass aggregating across all distance-scaled ring heads."""
        z_list = []
        for i in range(self.head_ring):
            z_list.append(self.ring_wise_agg_list[i](x))
        h = torch.cat(z_list, dim=1)
        return h


class SHGNN_Layer(Module):
    """
    The main SHGNN layer that integrates sector-wise and ring-wise aggregations.
    """
    def __init__(self, g, in_dim, out_dim, pool_dim, num_sect, rotation, num_ring,
                 bucket_interval, head_sect, head_ring, drop_rate, device='cpu'):
        super(SHGNN_Layer, self).__init__()
        self.device = device

        # Multi-head sector-wise and ring-wise aggregation
        self.sect_wise_agg_mh = SectorWiseAgg_RotateMHead(
            g, in_dim, out_dim, num_sect, rotation, head_sect, drop_rate, device)

        self.ring_wise_agg_mh = RingWiseAgg_ScaleMHead(
            g, in_dim, out_dim, num_ring, bucket_interval, head_ring, drop_rate, device)

        # Pooling layers
        self.sect_pool = nn.Linear((num_sect+1)*out_dim*head_sect, pool_dim)
        self.ring_pool = nn.Linear((num_ring+1)*out_dim*head_ring, pool_dim)

        # Fusion parameter
        self.gamma = nn.Parameter(torch.Tensor(1, 1))
        nn.init.xavier_uniform_(self.gamma)

        # Activation functions and dropout
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(drop_rate)

        # Initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def to(self, device):
        """Move module to specified device."""
        self.device = device
        self.sect_wise_agg_mh = self.sect_wise_agg_mh.to(device)
        self.ring_wise_agg_mh = self.ring_wise_agg_mh.to(device)
        return super(SHGNN_Layer, self).to(device)

    def fuse_two_views(self, h_sect, h_ring):
        """Fuse sector-wise and ring-wise representations."""
        gamma = self.sigmoid(self.gamma)
        h = gamma * h_sect + (1-gamma) * h_ring
        return h

    def forward(self, x):
        """Forward pass of the SHGNN layer."""
        # Get sector-wise and ring-wise representations
        h_sect = self.sect_wise_agg_mh(x)
        h_ring = self.ring_wise_agg_mh(x)

        # Apply dropout
        h_sect = self.dropout(h_sect)
        h_ring = self.dropout(h_ring)

        # Apply pooling
        h_sect = self.sect_pool(h_sect)
        h_ring = self.ring_pool(h_ring)

        # Apply dropout again
        h_sect = self.dropout(h_sect)
        h_ring = self.dropout(h_ring)

        # Fuse the two views
        h = self.fuse_two_views(h_sect, h_ring)

        return h


class SHGNN_DRSD_with_HyperConv(nn.Module):
    """
    Enhanced SHGNN model for Dangerous Road Section Detection (DRSD) task
    with Hyperbolic Convolution as feature extractor.
    """
    def __init__(self, g, in_dim, out_dim, pool_dim, num_sect, rotation, num_ring,
                 bucket_interval, head_sect, head_ring, drop_rate, device='cpu', c=1.0, hyp_layers=1):
        super(SHGNN_DRSD_with_HyperConv, self).__init__()
        self.device = device

        # Set up manifold and curvature
        self.manifold = PoincareBall()
        self.c = torch.tensor([c])
        if device != 'cpu':
            self.c = self.c.to(device)

        # Create adjacency matrix for the hyperbolic layer
        adj = nx.adjacency_matrix(g)
        self.adj = sparse_mx_to_torch_sparse_tensor(adj)
        if device != 'cpu':
            self.adj = self.adj.to(device)

        # Define hyperbolic graph convolution layers
        self.hyp_layers = hyp_layers
        self.hyp_conv_layers = nn.ModuleList()

        # First hyperbolic layer
        self.hyp_conv_layers.append(
            HyperbolicGraphConvolution(
                self.manifold,
                in_dim,                 # Input dimension
                in_dim,                 # Keep the same dimension
                c_in=self.c,            # Input curvature
                c_out=self.c,           # Output curvature
                dropout=drop_rate,      # Dropout rate
                act=F.relu,             # Activation function
                use_bias=True,          # Use bias
                use_att=False,          # No attention
                local_agg=False         # No local aggregation
            )
        )

        # Additional hyperbolic layers if needed
        for i in range(1, hyp_layers):
            self.hyp_conv_layers.append(
                HyperbolicGraphConvolution(
                    self.manifold,
                    in_dim,
                    in_dim,
                    c_in=self.c,
                    c_out=self.c,
                    dropout=drop_rate,
                    act=F.relu,
                    use_bias=True,
                    use_att=False,
                    local_agg=False
                )
            )

        # SHGNN layer remains the same
        self.shgnn_layer = SHGNN_Layer(
            g, in_dim, out_dim, pool_dim, num_sect, rotation, num_ring,
            bucket_interval, head_sect, head_ring, drop_rate, device)

        # Classification head
        self.linear_cls = nn.Linear(pool_dim, 1)

        # Activation and dropout
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(drop_rate)

        # Initialize parameters
        nn.init.xavier_uniform_(self.linear_cls.weight)
        nn.init.zeros_(self.linear_cls.bias)

    def to(self, device):
        """Move module to specified device."""
        self.device = device
        self.c = self.c.to(device)
        self.adj = self.adj.to(device)
        self.shgnn_layer = self.shgnn_layer.to(device)
        return super(SHGNN_DRSD_with_HyperConv, self).to(device)

    def forward(self, x):
        """Forward pass of the enhanced SHGNN_DRSD model."""
        # Convert input features to hyperbolic space
        x_tan = self.manifold.proj_tan0(x, self.c)
        x_hyp = self.manifold.expmap0(x_tan, c=self.c)
        x_hyp = self.manifold.proj(x_hyp, c=self.c)

        # Pass through hyperbolic convolution layers
        for i in range(self.hyp_layers):
            x_hyp, adj_out = self.hyp_conv_layers[i]((x_hyp, self.adj))

        # Map back to Euclidean space
        x_out = self.manifold.logmap0(x_hyp, c=self.c)
        x_out = self.manifold.proj_tan0(x_out, c=self.c)

        # Pass through original SHGNN architecture
        x_out = self.shgnn_layer(x_out)
        x_out = self.dropout(x_out)
        x_out = self.relu(x_out)
        x_out = self.linear_cls(x_out)
        x_out = self.sigmoid(x_out)

        return x_out


def create_shgnn_drsd_model(args, g):
    """
    Create an enhanced SHGNN model for DRSD task with hyperbolic convolution.

    Args:
        args: Command line arguments
        g: NetworkX graph

    Returns:
        Initialized model
    """
    # Parse device
    device = args.device if args.device != -1 else 'cpu'
    if device != 'cpu':
        device = f'cuda:{device}'

    # Parse bucket interval
    if isinstance(args.bucket_interval, str):
        bucket_interval = args.bucket_interval
    else:
        bucket_interval = ','.join([str(x) for x in args.bucket_interval])

    # Create model
    model = SHGNN_DRSD_with_HyperConv(
        g=g,
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        pool_dim=args.pool_dim,
        num_sect=args.num_sect,
        rotation=args.rotation,
        num_ring=args.num_ring,
        bucket_interval=bucket_interval,
        head_sect=args.head_sect,
        head_ring=args.head_ring,
        drop_rate=args.drop,
        device=device,
        c=args.c if hasattr(args, 'c') else 1.0,
        hyp_layers=args.hyp_layers if hasattr(args, 'hyp_layers') else 1
    )

    return model.to(device)


def load_drsd_data(data_dir):
    """
    Load DRSD task data

    Args:
        data_dir: Path to data directory

    Returns:
        Dictionary with loaded data
    """
    import os

    # Load graph
    g = nx.read_gpickle(os.path.join(data_dir, 'graph'))

    # Load features and labels
    features = np.load(os.path.join(data_dir, 'features.npy'))
    labels = np.load(os.path.join(data_dir, 'label.npy'))

    # Load train/val/test masks
    with open(os.path.join(data_dir, 'mask.json'), 'r') as f:
        mask_dict = json.load(f)

    # Convert to PyTorch tensors
    features_tensor = torch.FloatTensor(features)
    labels_tensor = torch.FloatTensor(labels)

    return {
        'g': g,
        'features': features_tensor,
        'labels': labels_tensor,
        'train_ids': mask_dict['train'],
        'val_ids': mask_dict['val'],
        'test_ids': mask_dict['test']
    }

def train_drsd_model(model, data, args):
    """
    Train DRSD model

    Args:
        model: Initialized model
        data: Dictionary with loaded data
        args: Arguments

    Returns:
        trained model and metrics
    """
    print("Starting training...")
    # Set device
    device = args.device if args.device != -1 else 'cpu'
    if device != 'cpu':
        device = f'cuda:{device}'

    # Extract data
    features = data['features'].to(device)
    labels = data['labels'].to(device)
    train_ids = torch.LongTensor(data['train_ids']).to(device)
    val_ids = torch.LongTensor(data['val_ids']).to(device)
    test_ids = torch.LongTensor(data['test_ids']).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.999
    )

    # Loss function for binary classification
    loss_fn = nn.BCELoss()

    # Best metrics tracking
    best_epoch = 0
    best_auc = 0
    best_test_auc = 0

    # Create log directory
    import os
    log_dir = os.path.join('logs', 'shgnn_hyp')
    os.makedirs(log_dir, exist_ok=True)

    # Training loop
    for epoch in range(args.epoch_num):
        # Training
        model.train()
        optimizer.zero_grad()

        # Forward pass
        pred = model(features).squeeze(1)

        # Compute loss
        pred_train = pred[train_ids]
        label_train = labels[train_ids]
        loss = loss_fn(pred_train, label_train)

        # Backward and optimize
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Evaluation
        model.eval()
        with torch.no_grad():
            pred = model(features).squeeze(1)

            # Compute metrics
            from sklearn.metrics import roc_curve, auc

            # Training set
            pred_train_np = pred[train_ids].cpu().numpy()
            label_train_np = labels[train_ids].cpu().numpy()
            fpr, tpr, _ = roc_curve(label_train_np, pred_train_np, pos_label=1)
            train_auc = auc(fpr, tpr)

            # Validation set
            pred_val_np = pred[val_ids].cpu().numpy()
            label_val_np = labels[val_ids].cpu().numpy()
            fpr, tpr, _ = roc_curve(label_val_np, pred_val_np, pos_label=1)
            val_auc = auc(fpr, tpr)

            # Test set
            pred_test_np = pred[test_ids].cpu().numpy()
            label_test_np = labels[test_ids].cpu().numpy()
            fpr, tpr, _ = roc_curve(label_test_np, pred_test_np, pos_label=1)
            test_auc = auc(fpr, tpr)

            # Print metrics
            print(f"Epoch {epoch:4d}: Loss = {loss.item():.4f}, Train AUC = {train_auc:.4f}, "
                  f"Val AUC = {val_auc:.4f}, Test AUC = {test_auc:.4f}")

            # Check for improvement
            if val_auc > best_auc:
                best_auc = val_auc
                best_test_auc = test_auc
                best_epoch = epoch

                # Save best model
                torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pt'))

                print(f"New best model! Val AUC = {val_auc:.4f}, Test AUC = {test_auc:.4f}")

    # Print final results
    print("\nTraining completed!")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Best test AUC: {best_test_auc:.4f}")

    return model, {
        'best_epoch': best_epoch,
        'best_val_auc': best_auc,
        'best_test_auc': best_test_auc
    }

def main(args):
    """
    Main function

    Args:
        args: Command line arguments
    """
    print("Setting up...")
    # Set random seed
    seed_setup(args.seed)

    # Set device
    device = args.device if args.device != -1 else 'cpu'
    if device != 'cpu':
        device = f'cuda:{device}'
        torch.cuda.set_device(args.device)

    # Load data
    print("Loading data...")
    data_dir = os.path.join('data', 'DRSD')
    data = load_drsd_data(data_dir)

    # Create model
    print("Creating model...")
    model = create_shgnn_drsd_model(args, data['g'])
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Train model
    model, metrics = train_drsd_model(model, data, args)

    return model, metrics

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='DRSD', help='Task name (should be DRSD)')
    parser.add_argument('--in_dim', type=int, default=64, help='Input feature dimension')
    parser.add_argument('--out_dim', type=int, default=32, help='Output embedding dimension')
    parser.add_argument('--pool_dim', type=int, default=32, help='Pooling dimension')
    parser.add_argument('--num_sect', type=int, default=4, help='Number of sectors')
    parser.add_argument('--rotation', type=float, default=45, help='Rotation angle in degrees')
    parser.add_argument('--num_ring', type=int, default=2, help='Number of rings')
    parser.add_argument('--bucket_interval', type=str, default='0.1,0.2', help='Bucket intervals for rings')
    parser.add_argument('--head_sect', type=int, default=2, help='Number of sector heads')
    parser.add_argument('--head_ring', type=int, default=2, help='Number of ring heads')
    parser.add_argument('--drop', type=float, default=0.7, help='Dropout rate')
    parser.add_argument('--epoch_num', type=int, default=10000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=int, default=0, help='GPU ID (use -1 for CPU)')
    parser.add_argument('--c', type=float, default=1.0, help='Hyperbolic curvature')
    parser.add_argument('--hyp_layers', type=int, default=1, help='Number of hyperbolic layers')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')

    args = parser.parse_args()

    # Run main function
    main(args)

    # Example usage:
    # python shgnn_with_hyper.py --in_dim 64 --out_dim 32 --pool_dim 32 --num_sect 4 --rotation 45
    # --num_ring 2 --bucket_interval 0.1,0.2 --head_sect 2 --head_ring 2 --drop 0.7 --device 0
    # --c 1.0 --hyp_layers 1