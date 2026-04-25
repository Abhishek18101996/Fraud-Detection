"""
src/models/gnn_model.py
─────────────────────────────────────────────────────────────
GraphSAGE model for transaction graph fraud detection.

Architecture:
  Input: node features (dim=4) + edge features (dim=30)
  GraphSAGE layers: 3 × message passing with mean aggregation
  Output: 128-dim node embeddings → edge classification head

Reference:
  Hamilton et al., "Inductive Representation Learning on Large Graphs"
  https://arxiv.org/abs/1706.02216
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import (
        SAGEConv,
        BatchNorm,
        global_mean_pool,
        global_max_pool,
    )
    from torch_geometric.utils import dropout_edge
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class EdgeAwareGraphSAGE(nn.Module):
    """
    GraphSAGE with edge-feature injection for transaction fraud detection.

    The key insight: transaction (edge) features are projected into
    the message-passing scheme by conditioning node updates on
    incoming edge attributes.

    Workflow:
      1. Project edge features → node message space
      2. Run GraphSAGE layers (3 hops capture counterparty patterns)
      3. Pool edge embeddings = concat(src_emb, dst_emb, edge_feat)
      4. MLP head → fraud logit
    """

    def __init__(
        self,
        node_feat_dim: int = 4,
        edge_feat_dim: int = 30,
        hidden_channels: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        embedding_dim: int = 128,
    ):
        super().__init__()

        if not HAS_PYG:
            raise RuntimeError("PyTorch Geometric required for GNN model.")

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim

        # ── Edge feature projection ────────────────────────
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # ── GraphSAGE convolution layers ───────────────────
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input projection (node features → hidden dim)
        self.input_proj = nn.Linear(node_feat_dim + hidden_channels, hidden_channels)

        for _ in range(num_layers):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, aggr="mean")
            )
            self.batch_norms.append(BatchNorm(hidden_channels))

        # ── Node embedding projection ──────────────────────
        self.node_embedding_proj = nn.Linear(hidden_channels, embedding_dim)

        # ── Edge classifier head ───────────────────────────
        # edge_repr = concat(src_embed, dst_embed, edge_feat_proj)
        edge_repr_dim = embedding_dim * 2 + hidden_channels
        self.edge_classifier = nn.Sequential(
            nn.Linear(edge_repr_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),  # Binary: fraud probability
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,         # [n_nodes, node_feat_dim]
        edge_index: torch.Tensor, # [2, n_edges]
        edge_attr: torch.Tensor,  # [n_edges, edge_feat_dim]
        batch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            edge_logits: [n_edges] — raw fraud logits per transaction
            node_embeddings: [n_nodes, embedding_dim] — for ensemble
        """
        # ── Project edge features ──────────────────────────
        edge_feat_proj = self.edge_proj(edge_attr)  # [n_edges, hidden]

        # ── Enrich node features with incoming edge info ───
        # Average incoming edge features per node
        n_nodes = x.size(0)
        node_edge_agg = torch.zeros(n_nodes, self.hidden_channels, device=x.device)
        dst_nodes = edge_index[1]  # destination = merchant nodes
        node_edge_agg.index_add_(0, dst_nodes, edge_feat_proj)
        count = torch.bincount(dst_nodes, minlength=n_nodes).float().unsqueeze(1).clamp(min=1)
        node_edge_agg = node_edge_agg / count

        # Concatenate and project
        x_combined = torch.cat([x, node_edge_agg], dim=1)
        h = F.relu(self.input_proj(x_combined))

        # ── GraphSAGE message passing ──────────────────────
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            h_new = conv(h, edge_index)
            h_new = bn(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            # Residual connection after first layer
            h = h_new + h if i > 0 else h_new

        # ── Node embeddings ────────────────────────────────
        node_embeddings = self.node_embedding_proj(h)  # [n_nodes, emb_dim]

        # ── Edge representations ───────────────────────────
        src_nodes = edge_index[0]
        dst_nodes_ = edge_index[1]

        src_emb = node_embeddings[src_nodes]       # [n_edges, emb_dim]
        dst_emb = node_embeddings[dst_nodes_]      # [n_edges, emb_dim]
        edge_repr = torch.cat([src_emb, dst_emb, edge_feat_proj], dim=1)

        # ── Edge fraud logits ──────────────────────────────
        edge_logits = self.edge_classifier(edge_repr).squeeze(-1)

        return edge_logits, node_embeddings

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Extract node embeddings only (for ensemble features)."""
        with torch.no_grad():
            _, node_embeddings = self.forward(x, edge_index, edge_attr)
        return node_embeddings

    def get_edge_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract per-transaction (edge) embeddings for XGBoost ensemble.
        Returns [n_edges, 2 * embedding_dim] tensor.
        """
        with torch.no_grad():
            _, node_emb = self.forward(x, edge_index, edge_attr)
            src = node_emb[edge_index[0]]
            dst = node_emb[edge_index[1]]
            return torch.cat([src, dst], dim=1)  # [n_edges, 2*emb_dim]


class FraudGNNLoss(nn.Module):
    """
    Focal loss for fraud detection (handles extreme class imbalance).

    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    """

    def __init__(
        self,
        alpha: float = 0.25,     # Weight for positive class
        gamma: float = 2.0,      # Focusing parameter
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none",
            pos_weight=self.pos_weight
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


def build_gnn_model(config=None) -> EdgeAwareGraphSAGE:
    """Factory function — reads from config or uses defaults."""
    if config is None:
        return EdgeAwareGraphSAGE()

    return EdgeAwareGraphSAGE(
        hidden_channels=config.gnn_hidden_channels,
        num_layers=config.gnn_num_layers,
        dropout=config.gnn_dropout,
        embedding_dim=config.gnn_embedding_dim,
    )