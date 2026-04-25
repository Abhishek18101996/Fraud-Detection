"""
src/data/graph_builder.py
─────────────────────────────────────────────────────────────
Construct bipartite transaction graphs for GNN training.

Graph schema:
  Nodes:  accounts (cardholders) + merchant categories
  Edges:  transactions (directed: account → merchant)
  Edge features: amount_log, time_delta, pca_features (V1..V28)
  Node features: aggregated transaction statistics

The ULB dataset doesn't have explicit account IDs, so we
use a heuristic: cluster transactions by (Time // 3600, V1_bin)
to approximate accounts.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger

# PyTorch Geometric imports (graceful fallback for environments without PyG)
try:
    from torch_geometric.data import Data, HeteroData
    from torch_geometric.utils import to_undirected
    HAS_PYG = True
except ImportError:
    logger.warning("PyTorch Geometric not installed. GNN features disabled.")
    HAS_PYG = False


PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
N_MERCHANT_CATEGORIES = 20  # Synthetic merchant bins for ULB


class TransactionGraphBuilder:
    """
    Builds a bipartite transaction graph from the ULB dataset.

    Since ULB is fully anonymised (no account/merchant IDs),
    we proxy them:
      - Accounts:  Cluster by (Time-bin, V1-bin) → ~10K pseudo-accounts
      - Merchants: Bin by Amount range → 20 merchant categories

    For real data (PaySim / production), replace the clustering
    with actual account_id and merchant_id columns.
    """

    def __init__(
        self,
        n_time_bins: int = 48,       # 48 × 1-hour bins for 2-day dataset
        n_v1_bins: int = 10,          # V1 discretisation buckets
        n_merchant_bins: int = N_MERCHANT_CATEGORIES,
        device: str = "cpu",
    ):
        self.n_time_bins = n_time_bins
        self.n_v1_bins = n_v1_bins
        self.n_merchant_bins = n_merchant_bins
        self.device = torch.device(device)

        # Populated during build()
        self.account_encoder_: Optional[Dict[tuple, int]] = None
        self.n_accounts_: int = 0
        self.n_merchants_: int = n_merchant_bins

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def build(self, df: pd.DataFrame) -> "Data":
        """
        Build a PyG Data object from a transaction DataFrame.

        Args:
            df: DataFrame with columns V1..V28, Amount, Time, Class

        Returns:
            torch_geometric.data.Data with:
              x           — node features (accounts + merchants stacked)
              edge_index  — COO format edge indices
              edge_attr   — edge (transaction) features
              y           — edge labels (fraud/legit)
              n_accounts  — number of account nodes
              n_merchants — number of merchant nodes
        """
        if not HAS_PYG:
            raise RuntimeError("PyTorch Geometric required for graph construction.")

        logger.info(f"Building transaction graph from {len(df):,} transactions...")

        df = df.copy().reset_index(drop=True)

        # ── 1. Assign pseudo node IDs ──────────────────────
        account_ids = self._assign_account_ids(df)
        merchant_ids = self._assign_merchant_ids(df)

        self.n_accounts_ = len(set(account_ids))
        self.account_encoder_ = {v: i for i, v in enumerate(sorted(set(account_ids)))}

        account_node_ids = torch.tensor(
            [self.account_encoder_[a] for a in account_ids], dtype=torch.long
        )
        merchant_node_ids = torch.tensor(merchant_ids, dtype=torch.long)

        # Merchants are offset by n_accounts in the unified node space
        merchant_node_ids_offset = merchant_node_ids + self.n_accounts_

        # ── 2. Build edge_index (COO) ──────────────────────
        edge_index = torch.stack(
            [account_node_ids, merchant_node_ids_offset], dim=0
        )  # shape [2, n_txns]

        # ── 3. Build edge features ─────────────────────────
        edge_attr = self._build_edge_features(df)  # shape [n_txns, n_edge_feats]

        # ── 4. Build node features ─────────────────────────
        x = self._build_node_features(
            df, account_node_ids, merchant_node_ids
        )  # shape [n_accounts + n_merchants, n_node_feats]

        # ── 5. Edge labels ─────────────────────────────────
        y_edges = torch.tensor(df["Class"].values, dtype=torch.long)

        # ── 6. Assemble PyG Data ───────────────────────────
        data = Data(
            x=x.to(self.device),
            edge_index=edge_index.to(self.device),
            edge_attr=edge_attr.to(self.device),
            y=y_edges.to(self.device),
            n_accounts=self.n_accounts_,
            n_merchants=self.n_merchants_,
            num_nodes=self.n_accounts_ + self.n_merchants_,
        )

        logger.info(
            f"Graph built — Nodes: {data.num_nodes:,} "
            f"(accounts={self.n_accounts_:,}, merchants={self.n_merchants_}) | "
            f"Edges: {data.num_edges:,} | "
            f"Fraud edges: {y_edges.sum().item():,}"
        )
        return data

    def build_mini_batch(
        self,
        df: pd.DataFrame,
        batch_size: int = 2048,
    ) -> List["Data"]:
        """Build multiple smaller graphs for mini-batch training."""
        graphs = []
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i : i + batch_size].reset_index(drop=True)
            if len(batch_df) < 10:
                continue
            graphs.append(self.build(batch_df))
        logger.info(f"Built {len(graphs)} mini-batch graphs (batch_size={batch_size})")
        return graphs

    # ──────────────────────────────────────────────────────────
    # Node / Edge construction
    # ──────────────────────────────────────────────────────────

    def _assign_account_ids(self, df: pd.DataFrame) -> List[tuple]:
        """
        Proxy account ID = (time_bin, V1_bin).
        This approximates the idea that the same cardholder
        transacts within a time window with similar PCA signature.
        """
        time_max = df["Time"].max() + 1e-8
        time_bins = (df["Time"] / time_max * self.n_time_bins).astype(int).clip(
            0, self.n_time_bins - 1
        )
        v1_bins = pd.cut(df["V1"], bins=self.n_v1_bins, labels=False).fillna(0).astype(int)
        return list(zip(time_bins.tolist(), v1_bins.tolist()))

    def _assign_merchant_ids(self, df: pd.DataFrame) -> List[int]:
        """
        Proxy merchant category = Amount decile bin.
        Higher amounts → higher-numbered merchant categories.
        """
        return (
            pd.cut(df["Amount"], bins=self.n_merchant_bins, labels=False)
            .fillna(0)
            .astype(int)
            .tolist()
        )

    def _build_edge_features(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Edge features = transaction-level attributes.

        Features per edge:
          - amount_log (1)
          - time_normalised (1)
          - V1..V28 (28)  ← full PCA features
          Total: 30 features
        """
        amount_log = np.log1p(df["Amount"].values).reshape(-1, 1)
        time_norm = (
            df["Time"].values / (df["Time"].max() + 1e-8)
        ).reshape(-1, 1)
        pca = df[PCA_FEATURES].values  # shape [n, 28]

        edge_feat = np.concatenate([amount_log, time_norm, pca], axis=1)
        return torch.tensor(edge_feat, dtype=torch.float32)

    def _build_node_features(
        self,
        df: pd.DataFrame,
        account_ids: torch.Tensor,
        merchant_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Node features = aggregated transaction statistics.

        Account nodes (per pseudo-account):
          - mean_amount, std_amount, txn_count, fraud_rate
        Merchant nodes (per category):
          - mean_amount, txn_count, fraud_rate
        Total node feature dim = 4 (unified, padded to same width)
        """
        n_nodes = self.n_accounts_ + self.n_merchants_
        node_feat_dim = 4
        node_features = np.zeros((n_nodes, node_feat_dim), dtype=np.float32)

        # ── Account node features ──────────────────────────
        acc_arr = account_ids.numpy()
        for acc_id in np.unique(acc_arr):
            mask = acc_arr == acc_id
            amounts = df["Amount"].values[mask]
            labels = df["Class"].values[mask]
            node_features[acc_id] = [
                np.log1p(amounts.mean()),
                np.log1p(amounts.std() + 1e-8),
                np.log1p(mask.sum()),
                labels.mean(),
            ]

        # ── Merchant node features ─────────────────────────
        merch_arr = merchant_ids.numpy()
        for merch_id in np.unique(merch_arr):
            mask = merch_arr == merch_id
            amounts = df["Amount"].values[mask]
            labels = df["Class"].values[mask]
            node_idx = self.n_accounts_ + merch_id
            node_features[node_idx] = [
                np.log1p(amounts.mean()),
                0.0,  # std not meaningful for merchant categories
                np.log1p(mask.sum()),
                labels.mean(),
            ]

        return torch.tensor(node_features, dtype=torch.float32)


# ──────────────────────────────────────────────────────────────
# Utility: single transaction → local subgraph
# ──────────────────────────────────────────────────────────────

def transaction_to_subgraph(
    txn_features: dict,
    context_df: Optional[pd.DataFrame] = None,
) -> Optional["Data"]:
    """
    Build a tiny local subgraph for a single inference transaction.
    Optionally include recent context transactions for richer neighbourhood.
    """
    if not HAS_PYG:
        return None

    rows = [txn_features]
    if context_df is not None:
        rows = context_df.to_dict("records") + rows

    df = pd.DataFrame(rows)
    for col in PCA_FEATURES + ["Amount", "Time", "Class"]:
        if col not in df.columns:
            df[col] = 0.0

    builder = TransactionGraphBuilder()
    return builder.build(df)