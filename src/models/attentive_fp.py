"""
AttentiveFP Implementation
==========================
Graph attention-based fingerprint for molecular property prediction.

Reference:
    Xiong et al. (2020) J. Med. Chem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import softmax


class AttentiveLayer(MessagePassing):
    """Single AttentiveFP layer with attention mechanism."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__(aggr='add')

        self.node_dim = node_dim
        self.hidden_dim = hidden_dim

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # GRU for node update
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr, index, size_i):
        # Concatenate source, target, and edge features
        cat_feat = torch.cat([x_i, x_j, edge_attr], dim=-1)

        # Compute attention weights
        alpha = self.attention(cat_feat)
        alpha = softmax(alpha, index, num_nodes=size_i)

        # Message
        msg = torch.cat([x_j, edge_attr], dim=-1)
        msg = self.node_mlp(msg)

        return alpha * msg

    def update(self, aggr_out, x):
        return self.gru(aggr_out, x)


class AttentiveFP(nn.Module):
    """
    AttentiveFP model for molecular property prediction.

    Parameters
    ----------
    node_feat_dim : int
        Input node feature dimension.
    edge_feat_dim : int
        Input edge feature dimension.
    hidden_dim : int
        Hidden dimension.
    num_layers : int
        Number of message passing layers.
    num_timesteps : int
        Number of attention readout timesteps.
    dropout : float
        Dropout rate.
    """

    def __init__(
            self,
            node_feat_dim: int = 39,
            edge_feat_dim: int = 10,
            hidden_dim: int = 96,  # CHANGED from 256 (Hödl et al. 2025)
            num_layers: int = 2,  # CHANGED from 3 (Hödl et al. 2025)
            num_timesteps: int = 2,
            dropout: float = 0.1  # CHANGED from 0.2 (Hödl et al. 2025)
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps

        # Initial projection
        self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)

        # Message passing layers
        self.layers = nn.ModuleList([
            AttentiveLayer(hidden_dim, edge_feat_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Graph-level attention readout
        self.graph_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # GRU for graph-level readout
        self.graph_gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Initial node embedding
        x = self.node_embedding(x)
        x = F.relu(x)

        # Message passing
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # Graph-level readout with attention
        graph_embed = global_add_pool(x, batch)

        for _ in range(self.num_timesteps):
            # Compute attention over nodes
            att_scores = self.graph_attention(x)
            att_scores = softmax(att_scores, batch)

            # Weighted sum
            context = global_add_pool(att_scores * x, batch)

            # Update graph embedding
            graph_embed = self.graph_gru(context, graph_embed)

        return graph_embed