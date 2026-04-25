"""
src/training/train_gnn.py
─────────────────────────────────────────────────────────────
GraphSAGE training loop with MLflow experiment tracking.

Features:
  - Focal loss for class imbalance
  - Cosine LR schedule with warm-up
  - Early stopping on validation AUC-PR
  - Full MLflow logging: params, metrics, model artifacts
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.optim as optim
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.models.gnn_model import EdgeAwareGraphSAGE, FraudGNNLoss, build_gnn_model


class GNNTrainer:
    """
    Training orchestrator for GraphSAGE fraud detection model.

    Training loop handles:
    - Mini-batch graph training (memory efficient)
    - Gradient clipping (stability for deep GNNs)
    - LR scheduling (cosine annealing)
    - Early stopping with patience
    - MLflow checkpoint saving
    """

    def __init__(
        self,
        model: Optional[EdgeAwareGraphSAGE] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 15,
        max_epochs: int = 100,
        checkpoint_dir: str = "checkpoints",
        device: str = "cpu",
        pos_weight: float = 578.0,   # Fraud ratio in ULB
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.model = model or build_gnn_model()
        self.model = self.model.to(self.device)

        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.max_epochs = max_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Scheduler: cosine annealing with warm restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2
        )

        # Loss: focal loss for imbalance
        pos_w = torch.tensor([pos_weight], device=self.device)
        self.criterion = FraudGNNLoss(
            alpha=0.25, gamma=2.0, pos_weight=pos_w
        )

        self.best_val_auc_pr: float = 0.0
        self.epochs_without_improvement: int = 0
        self.history: Dict[str, list] = {
            "train_loss": [], "val_loss": [],
            "val_auc_roc": [], "val_auc_pr": [],
        }

    # ──────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────

    def train(
        self,
        train_graphs: list,       # List of PyG Data objects
        val_graphs: list,
        run_name: str = "gnn-training",
        mlflow_experiment: str = "fraud-detection",
    ) -> EdgeAwareGraphSAGE:
        """
        Full training run with MLflow tracking.

        Returns best model (by validation AUC-PR).
        """
        mlflow.set_experiment(mlflow_experiment)

        with mlflow.start_run(run_name=run_name) as run:
            self._log_mlflow_params()
            logger.info(
                f"Starting GNN training — device={self.device} | "
                f"train_graphs={len(train_graphs)} | val_graphs={len(val_graphs)}"
            )

            for epoch in range(1, self.max_epochs + 1):
                t0 = time.time()

                train_loss = self._train_epoch(train_graphs)
                val_metrics = self._validate(val_graphs)

                epoch_time = time.time() - t0

                # Record history
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_auc_roc"].append(val_metrics["auc_roc"])
                self.history["val_auc_pr"].append(val_metrics["auc_pr"])

                # LR step
                self.scheduler.step(epoch - 1 + len(train_graphs) / max(len(train_graphs), 1))
                current_lr = self.optimizer.param_groups[0]["lr"]

                # MLflow logging
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_metrics["loss"],
                        "val_auc_roc": val_metrics["auc_roc"],
                        "val_auc_pr": val_metrics["auc_pr"],
                        "lr": current_lr,
                    },
                    step=epoch,
                )

                logger.info(
                    f"Epoch {epoch:03d}/{self.max_epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_auc_pr={val_metrics['auc_pr']:.4f} | "
                    f"val_auc_roc={val_metrics['auc_roc']:.4f} | "
                    f"lr={current_lr:.6f} | "
                    f"time={epoch_time:.1f}s"
                )

                # Early stopping + checkpoint
                if val_metrics["auc_pr"] > self.best_val_auc_pr:
                    self.best_val_auc_pr = val_metrics["auc_pr"]
                    self.epochs_without_improvement = 0
                    self._save_checkpoint(epoch, val_metrics)
                    mlflow.pytorch.log_model(
                        self.model, artifact_path="gnn_model_best"
                    )
                    logger.info(f"  ✓ New best model saved (AUC-PR={self.best_val_auc_pr:.4f})")
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= self.patience:
                        logger.info(
                            f"Early stopping at epoch {epoch} "
                            f"(patience={self.patience})"
                        )
                        break

            # Load best checkpoint
            best_model = self._load_best_checkpoint()
            mlflow.log_metric("best_val_auc_pr", self.best_val_auc_pr)
            logger.info(
                f"Training complete. Best val AUC-PR: {self.best_val_auc_pr:.4f}"
            )
            return best_model

    # ──────────────────────────────────────────────────────────
    # Epoch routines
    # ──────────────────────────────────────────────────────────

    def _train_epoch(self, graphs: list) -> float:
        """One training epoch over all mini-batch graphs."""
        self.model.train()
        total_loss, n_batches = 0.0, 0

        for data in graphs:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            edge_logits, _ = self.model(
                data.x, data.edge_index, data.edge_attr
            )

            loss = self.criterion(edge_logits, data.y.float())
            loss.backward()

            # Gradient clipping — important for GNN stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, graphs: list) -> Dict[str, float]:
        """Validate over all val graphs."""
        self.model.eval()
        all_logits, all_labels = [], []
        total_loss = 0.0

        for data in graphs:
            data = data.to(self.device)
            edge_logits, _ = self.model(
                data.x, data.edge_index, data.edge_attr
            )
            loss = self.criterion(edge_logits, data.y.float())
            total_loss += loss.item()

            probs = torch.sigmoid(edge_logits).cpu().numpy()
            labels = data.y.cpu().numpy()
            all_logits.extend(probs.tolist())
            all_labels.extend(labels.tolist())

        all_logits = np.array(all_logits)
        all_labels = np.array(all_labels)

        if all_labels.sum() == 0:
            return {"loss": total_loss / len(graphs), "auc_roc": 0.5, "auc_pr": 0.0}

        return {
            "loss": total_loss / len(graphs),
            "auc_roc": roc_auc_score(all_labels, all_logits),
            "auc_pr": average_precision_score(all_labels, all_logits),
        }

    # ──────────────────────────────────────────────────────────
    # Checkpointing
    # ──────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, metrics: Dict) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_auc_pr": metrics["auc_pr"],
            "val_auc_roc": metrics["auc_roc"],
        }
        path = self.checkpoint_dir / "graphsage_best.pt"
        torch.save(checkpoint, path)

    def _load_best_checkpoint(self) -> EdgeAwareGraphSAGE:
        path = self.checkpoint_dir / "graphsage_best.pt"
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
        return self.model

    def _log_mlflow_params(self) -> None:
        mlflow.log_params({
            "model_type": "GraphSAGE",
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "max_epochs": self.max_epochs,
            "hidden_channels": self.model.hidden_channels,
            "num_layers": self.model.num_layers,
            "dropout": self.model.dropout,
            "embedding_dim": self.model.embedding_dim,
            "device": str(self.device),
        })