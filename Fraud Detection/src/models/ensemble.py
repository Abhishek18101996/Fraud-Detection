"""
src/models/ensemble.py
─────────────────────────────────────────────────────────────
GNN + XGBoost ensemble for production fraud detection.

Strategy:
  1. GNN generates 256-dim edge embeddings (concat src+dst node embs)
  2. Tabular features + GNN embeddings → XGBoost final classifier
  3. Optional: stacking with logistic regression meta-learner

This fusion captures:
  - Graph structure: who the counterparties are and their history
  - Tabular signals: amount anomaly, time patterns, PCA features
  - Cross-modal: interactions between graph and tabular signals
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from src.models.gnn_model import EdgeAwareGraphSAGE
from src.models.xgboost_model import XGBoostFraudClassifier


class FraudEnsemble:
    """
    Production ensemble: GNN embeddings + XGBoost classifier.

    Inference flow:
      transaction payload
        ├─ tabular features → preprocessor → X_tab [n, 42]
        ├─ graph features   → graph builder → Data
        │     └─ GNN forward → edge_emb [n, 256]
        └─ concat [X_tab | edge_emb] → XGBoost → fraud_score
    """

    MODEL_VERSION = "ensemble-v2.1.0"

    def __init__(
        self,
        gnn: Optional[EdgeAwareGraphSAGE] = None,
        xgb_model: Optional[XGBoostFraudClassifier] = None,
        device: str = "cpu",
        use_gnn: bool = True,
    ):
        self.gnn = gnn
        self.xgb_model = xgb_model
        self.device = torch.device(device)
        self.use_gnn = use_gnn
        self._is_ready = False

        if self.gnn:
            self.gnn = self.gnn.to(self.device).eval()

    # ──────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────

    def predict(
        self,
        X_tabular: np.ndarray,
        graph_data=None,         # PyG Data object or None
    ) -> np.ndarray:
        """
        Predict fraud probability for a batch of transactions.

        Args:
            X_tabular: [n, n_tab_features] preprocessed tabular features
            graph_data: PyG Data object (optional; None → tabular only)

        Returns:
            fraud_probs: [n] float array of fraud probabilities
        """
        X = self._build_ensemble_features(X_tabular, graph_data)
        return self.xgb_model.predict_proba(X)

    def predict_with_confidence(
        self,
        X_tabular: np.ndarray,
        graph_data=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (fraud_scores, confidence_scores).
        Confidence = distance from 0.5 decision boundary, normalised.
        """
        probs = self.predict(X_tabular, graph_data)
        confidence = np.abs(probs - 0.5) * 2  # [0, 1] scale
        return probs, confidence

    def classify(
        self,
        fraud_scores: np.ndarray,
        threshold: float = 0.5,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Binary classification + risk tier assignment.

        Returns:
            (is_fraud_binary, risk_tiers)
        """
        is_fraud = (fraud_scores >= threshold).astype(int)
        risk_tiers = [self._score_to_tier(s) for s in fraud_scores]
        return is_fraud, risk_tiers

    # ──────────────────────────────────────────────────────────
    # Feature construction
    # ──────────────────────────────────────────────────────────

    def _build_ensemble_features(
        self,
        X_tabular: np.ndarray,
        graph_data=None,
    ) -> np.ndarray:
        """Concatenate tabular + GNN embeddings into ensemble feature matrix."""
        if not self.use_gnn or graph_data is None or self.gnn is None:
            return X_tabular

        gnn_emb = self._extract_gnn_embeddings(graph_data)
        n = min(X_tabular.shape[0], gnn_emb.shape[0])

        X_combined = np.concatenate(
            [X_tabular[:n], gnn_emb[:n]], axis=1
        )
        return X_combined

    def _extract_gnn_embeddings(self, graph_data) -> np.ndarray:
        """Run GNN forward pass and extract edge embeddings."""
        self.gnn.eval()
        with torch.no_grad():
            data = graph_data.to(self.device)
            edge_emb = self.gnn.get_edge_embeddings(
                data.x, data.edge_index, data.edge_attr
            )
        return edge_emb.cpu().numpy()

    # ──────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        X_tabular: np.ndarray,
        y: np.ndarray,
        graph_data=None,
        split_name: str = "test",
    ) -> Dict[str, float]:
        """Evaluate ensemble on a labelled dataset."""
        probs = self.predict(X_tabular, graph_data)
        preds = (probs >= self.xgb_model.threshold_).astype(int)

        metrics = {
            "auc_roc": roc_auc_score(y, probs),
            "auc_pr": average_precision_score(y, probs),
            "f1": f1_score(y, preds, zero_division=0),
            "model_version": self.MODEL_VERSION,
        }

        logger.info(f"\n{'='*50}")
        logger.info(f"ENSEMBLE Evaluation — {split_name.upper()}")
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:.4f}")

        return metrics

    # ──────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────

    def save(self, checkpoint_dir: str | Path) -> None:
        """Save all ensemble components."""
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save GNN weights
        if self.gnn is not None:
            torch.save(
                self.gnn.state_dict(),
                checkpoint_dir / "graphsage_best.pt"
            )

        # Save XGBoost model
        if self.xgb_model is not None:
            self.xgb_model.save(checkpoint_dir / "xgboost_ensemble.pkl")

        # Save ensemble metadata
        meta = {
            "model_version": self.MODEL_VERSION,
            "use_gnn": self.use_gnn,
            "device": str(self.device),
        }
        with open(checkpoint_dir / "ensemble_meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        logger.info(f"Ensemble saved → {checkpoint_dir}")

    @classmethod
    def load(
        cls,
        checkpoint_dir: str | Path,
        device: str = "cpu",
    ) -> "FraudEnsemble":
        """Load ensemble from checkpoint directory."""
        checkpoint_dir = Path(checkpoint_dir)

        # Load metadata
        meta_path = checkpoint_dir / "ensemble_meta.pkl"
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
        else:
            meta = {"use_gnn": True}

        # Load XGBoost
        xgb_path = checkpoint_dir / "xgboost_ensemble.pkl"
        xgb_model = XGBoostFraudClassifier.load(xgb_path) if xgb_path.exists() else None

        # Load GNN
        gnn_path = checkpoint_dir / "graphsage_best.pt"
        gnn = None
        if gnn_path.exists() and meta.get("use_gnn"):
            try:
                from src.models.gnn_model import build_gnn_model
                gnn = build_gnn_model()
                gnn.load_state_dict(torch.load(gnn_path, map_location=device))
                gnn.eval()
            except Exception as e:
                logger.warning(f"Could not load GNN: {e}. Using tabular-only mode.")

        ensemble = cls(gnn=gnn, xgb_model=xgb_model, device=device, use_gnn=meta.get("use_gnn", True))
        ensemble._is_ready = True
        logger.info(f"Ensemble loaded ← {checkpoint_dir} (version={meta.get('model_version', 'unknown')})")
        return ensemble

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _score_to_tier(score: float) -> str:
        if score >= 0.90:
            return "CRITICAL"
        elif score >= 0.75:
            return "HIGH"
        elif score >= 0.50:
            return "MEDIUM"
        return "LOW"