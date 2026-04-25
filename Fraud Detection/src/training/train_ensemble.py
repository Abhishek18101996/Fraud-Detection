"""
src/training/train_ensemble.py
─────────────────────────────────────────────────────────────
Full GNN + XGBoost ensemble training pipeline.

Steps:
  1. Load & preprocess data
  2. Train XGBoost baseline (tabular only)
  3. Build transaction graphs
  4. Train GraphSAGE on graphs
  5. Extract GNN edge embeddings
  6. Concatenate tabular + GNN features
  7. Retrain XGBoost on enriched features
  8. Fit SHAP explainer
  9. Save all artifacts + log to MLflow
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import numpy as np
import torch
from loguru import logger

from src.config import get_settings
from src.data.graph_builder import TransactionGraphBuilder
from src.data.loader import FraudDataLoader, generate_synthetic_dataset
from src.data.preprocessor import FraudPreprocessor
from src.explainability.shap_explainer import FraudSHAPExplainer
from src.models.ensemble import FraudEnsemble
from src.models.gnn_model import build_gnn_model
from src.models.xgboost_model import XGBoostFraudClassifier
from src.training.train_gnn import GNNTrainer

settings = get_settings()


class EnsembleTrainer:
    """
    Orchestrates the full multi-stage training pipeline.

    Stage 1 — Tabular baseline:
        Preprocessed features → XGBoost → baseline AUC-PR

    Stage 2 — Graph training:
        Transaction graph → GraphSAGE → 256-dim edge embeddings

    Stage 3 — Ensemble fusion:
        [tabular | GNN embeddings] → XGBoost → ensemble AUC-PR

    Stage 4 — Explainability:
        SHAP TreeExplainer fitted on training background
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        mlflow_experiment: str = "fraud-detection",
        device: str = "cpu",
        use_synthetic: bool = False,
        data_dir: Optional[str] = None,
        run_optuna: bool = False,
        gnn_epochs: int = 50,
        gnn_batch_size: int = 4096,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_experiment = mlflow_experiment
        self.device = device
        self.use_synthetic = use_synthetic
        self.data_dir = data_dir
        self.run_optuna = run_optuna
        self.gnn_epochs = gnn_epochs
        self.gnn_batch_size = gnn_batch_size

        # Will be populated during training
        self.preprocessor: Optional[FraudPreprocessor] = None
        self.gnn_model = None
        self.xgb_baseline: Optional[XGBoostFraudClassifier] = None
        self.xgb_ensemble: Optional[XGBoostFraudClassifier] = None
        self.shap_explainer: Optional[FraudSHAPExplainer] = None
        self.graph_builder: Optional[TransactionGraphBuilder] = None

    # ──────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────

    def run(self) -> Dict[str, float]:
        """
        Execute full training pipeline.
        Returns final ensemble metrics dict.
        """
        mlflow.set_experiment(self.mlflow_experiment)

        with mlflow.start_run(run_name="ensemble-full-pipeline") as run:
            run_id = run.info.run_id
            logger.info(f"MLflow run started: {run_id}")

            # ── Stage 0: Load data ─────────────────────────
            train_df, val_df, test_df = self._load_data()

            # ── Stage 1: Tabular preprocessing + baseline ──
            X_train_tab, y_train = self._stage1_tabular(train_df, val_df)
            X_val_tab, y_val = self.preprocessor.transform(val_df)
            X_test_tab, y_test = self.preprocessor.transform(test_df)

            baseline_metrics = self._train_baseline(
                X_train_tab, y_train, X_val_tab, y_val, X_test_tab, y_test
            )
            self._log_stage_metrics("baseline", baseline_metrics)

            # ── Stage 2: GNN training ──────────────────────
            gnn_emb_train, gnn_emb_val, gnn_emb_test = self._stage2_gnn(
                train_df, val_df, test_df
            )

            # ── Stage 3: Ensemble fusion ───────────────────
            ensemble_metrics = self._stage3_ensemble(
                X_train_tab, gnn_emb_train, y_train,
                X_val_tab, gnn_emb_val, y_val,
                X_test_tab, gnn_emb_test, y_test,
            )
            self._log_stage_metrics("ensemble", ensemble_metrics)

            # ── Stage 4: SHAP explainer ────────────────────
            self._stage4_shap(X_train_tab, gnn_emb_train)

            # ── Save artifacts ─────────────────────────────
            self._save_all_artifacts(run_id)

            # Log improvement
            improvement = ensemble_metrics["auc_pr"] - baseline_metrics.get("auc_pr", 0)
            mlflow.log_metric("auc_pr_improvement_vs_baseline", round(improvement, 4))
            logger.info(
                f"\n{'='*60}\n"
                f"TRAINING COMPLETE\n"
                f"  Baseline AUC-PR:  {baseline_metrics.get('auc_pr', 0):.4f}\n"
                f"  Ensemble AUC-PR:  {ensemble_metrics['auc_pr']:.4f}\n"
                f"  Improvement:      +{improvement:.4f}\n"
                f"  MLflow run ID:    {run_id}\n"
                f"{'='*60}"
            )

            return {**ensemble_metrics, "run_id": run_id}

    # ──────────────────────────────────────────────────────────
    # Stage implementations
    # ──────────────────────────────────────────────────────────

    def _load_data(self):
        logger.info("━━━ Stage 0: Loading data ━━━")
        loader = FraudDataLoader(data_dir=self.data_dir)

        if self.use_synthetic:
            logger.info("Using synthetic dataset (demo mode)")
            df = generate_synthetic_dataset(n_samples=50_000)
            loader._df = df

        train_df, val_df, test_df = loader.get_splits()
        logger.info(
            f"Data loaded — train={len(train_df):,} | val={len(val_df):,} | test={len(test_df):,}"
        )
        return train_df, val_df, test_df

    def _stage1_tabular(self, train_df, val_df):
        """Feature engineering + SMOTE on training data."""
        logger.info("━━━ Stage 1: Tabular feature engineering ━━━")
        self.preprocessor = FraudPreprocessor(apply_smote=True)
        X_train, y_train = self.preprocessor.fit_transform(train_df)

        # Save preprocessor immediately
        pp_path = self.checkpoint_dir / "preprocessor.pkl"
        with open(pp_path, "wb") as f:
            pickle.dump(self.preprocessor, f)
        logger.info(f"Preprocessor saved → {pp_path}")

        return X_train, y_train

    def _train_baseline(
        self,
        X_train, y_train, X_val, y_val, X_test, y_test
    ) -> Dict:
        """Train XGBoost on tabular features only (performance floor)."""
        logger.info("━━━ Stage 1b: XGBoost baseline ━━━")
        t0 = time.time()

        self.xgb_baseline = XGBoostFraudClassifier(
            n_estimators=300,
            scale_pos_weight=float(sum(y_train == 0)) / max(sum(y_train == 1), 1),
        )
        self.xgb_baseline.fit(
            X_train, y_train, X_val, y_val,
            feature_names=self.preprocessor.get_feature_names(),
        )
        self.xgb_baseline.optimise_threshold(X_val, y_val)
        metrics = self.xgb_baseline.evaluate(X_test, y_test, split_name="baseline_test")

        logger.info(f"Baseline trained in {time.time() - t0:.1f}s")
        self.xgb_baseline.save(self.checkpoint_dir / "xgboost_baseline.pkl")
        return metrics

    def _stage2_gnn(self, train_df, val_df, test_df):
        """Build graphs, train GNN, extract embeddings."""
        logger.info("━━━ Stage 2: GNN training ━━━")

        try:
            from torch_geometric.data import Data
            HAS_PYG = True
        except ImportError:
            logger.warning("PyTorch Geometric not available. Skipping GNN stage.")
            n_train = len(train_df)
            n_val = len(val_df)
            n_test = len(test_df)
            return (
                np.zeros((n_train, 256), dtype=np.float32),
                np.zeros((n_val, 256), dtype=np.float32),
                np.zeros((n_test, 256), dtype=np.float32),
            )

        # Build mini-batch graphs
        logger.info("Building transaction graphs...")
        self.graph_builder = TransactionGraphBuilder(device=self.device)
        train_graphs = self.graph_builder.build_mini_batch(train_df, batch_size=self.gnn_batch_size)
        val_graphs = self.graph_builder.build_mini_batch(val_df, batch_size=self.gnn_batch_size)
        test_graphs = self.graph_builder.build_mini_batch(test_df, batch_size=self.gnn_batch_size)

        # Train GNN
        self.gnn_model = build_gnn_model(settings.model)
        trainer = GNNTrainer(
            model=self.gnn_model,
            max_epochs=self.gnn_epochs,
            checkpoint_dir=str(self.checkpoint_dir),
            device=self.device,
        )
        self.gnn_model = trainer.train(
            train_graphs, val_graphs,
            run_name="graphsage-training",
            mlflow_experiment=self.mlflow_experiment,
        )
        self.gnn_model.eval()

        # Extract edge embeddings for each split
        def extract_embeddings(graphs, total_rows):
            all_emb = []
            for g in graphs:
                g = g.to(torch.device(self.device))
                emb = self.gnn_model.get_edge_embeddings(g.x, g.edge_index, g.edge_attr)
                all_emb.append(emb.cpu().numpy())
            if not all_emb:
                return np.zeros((total_rows, 256), dtype=np.float32)
            combined = np.concatenate(all_emb, axis=0)
            # Pad / truncate to match tabular rows
            if len(combined) < total_rows:
                pad = np.zeros((total_rows - len(combined), combined.shape[1]), dtype=np.float32)
                combined = np.vstack([combined, pad])
            return combined[:total_rows]

        logger.info("Extracting GNN edge embeddings...")
        gnn_emb_train = extract_embeddings(train_graphs, len(train_df))
        gnn_emb_val = extract_embeddings(val_graphs, len(val_df))
        gnn_emb_test = extract_embeddings(test_graphs, len(test_df))

        logger.info(
            f"GNN embeddings — train={gnn_emb_train.shape} | "
            f"val={gnn_emb_val.shape} | test={gnn_emb_test.shape}"
        )
        return gnn_emb_train, gnn_emb_val, gnn_emb_test

    def _stage3_ensemble(
        self,
        X_train_tab, gnn_emb_train, y_train,
        X_val_tab, gnn_emb_val, y_val,
        X_test_tab, gnn_emb_test, y_test,
    ) -> Dict:
        """Fuse tabular + GNN features and train final XGBoost."""
        logger.info("━━━ Stage 3: Ensemble fusion training ━━━")

        # Align sample counts (GNN batch size may differ from tabular)
        n_train = min(len(X_train_tab), len(gnn_emb_train))
        n_val = min(len(X_val_tab), len(gnn_emb_val))
        n_test = min(len(X_test_tab), len(gnn_emb_test))

        X_train_ens = np.concatenate([X_train_tab[:n_train], gnn_emb_train[:n_train]], axis=1)
        X_val_ens = np.concatenate([X_val_tab[:n_val], gnn_emb_val[:n_val]], axis=1)
        X_test_ens = np.concatenate([X_test_tab[:n_test], gnn_emb_test[:n_test]], axis=1)
        y_train_ens = y_train[:n_train]

        logger.info(f"Ensemble feature matrix: {X_train_ens.shape[1]} features")

        # Feature names = tabular + GNN embedding names
        tab_names = self.preprocessor.get_feature_names()
        gnn_names = [f"gnn_emb_{i}" for i in range(gnn_emb_train.shape[1])]
        all_feature_names = tab_names + gnn_names

        self.xgb_ensemble = XGBoostFraudClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.75,
            scale_pos_weight=float(sum(y_train_ens == 0)) / max(sum(y_train_ens == 1), 1),
        )
        self.xgb_ensemble.fit(
            X_train_ens, y_train_ens,
            X_val_ens, y_val[:n_val],
            feature_names=all_feature_names,
        )
        self.xgb_ensemble.optimise_threshold(X_val_ens, y_val[:n_val])
        metrics = self.xgb_ensemble.evaluate(X_test_ens, y_test[:n_test], split_name="ensemble_test")

        # Save as primary ensemble model
        self.xgb_ensemble.save(self.checkpoint_dir / "xgboost_ensemble.pkl")
        return metrics

    def _stage4_shap(self, X_train_tab, gnn_emb_train):
        """Fit SHAP TreeExplainer on ensemble model."""
        logger.info("━━━ Stage 4: SHAP explainer fitting ━━━")

        if self.xgb_ensemble is None:
            logger.warning("No ensemble model — skipping SHAP.")
            return

        n = min(len(X_train_tab), len(gnn_emb_train))
        X_background = np.concatenate([X_train_tab[:n], gnn_emb_train[:n]], axis=1)

        tab_names = self.preprocessor.get_feature_names()
        gnn_names = [f"gnn_emb_{i}" for i in range(gnn_emb_train.shape[1])]

        self.shap_explainer = FraudSHAPExplainer(
            model=self.xgb_ensemble,
            feature_names=tab_names + gnn_names,
            n_background_samples=200,
        )
        self.shap_explainer.fit(X_background)

        # Save SHAP explainer
        shap_path = self.checkpoint_dir / "shap_explainer.pkl"
        with open(shap_path, "wb") as f:
            pickle.dump(self.shap_explainer, f)
        logger.info(f"SHAP explainer saved → {shap_path}")

    # ──────────────────────────────────────────────────────────
    # Artifact saving
    # ──────────────────────────────────────────────────────────

    def _save_all_artifacts(self, run_id: str) -> None:
        """Save full model ensemble and log to MLflow."""
        logger.info("Saving all artifacts...")

        # Bundle ensemble metadata
        ensemble = FraudEnsemble(
            gnn=self.gnn_model,
            xgb_model=self.xgb_ensemble,
            device=self.device,
            use_gnn=(self.gnn_model is not None),
        )
        ensemble.save(self.checkpoint_dir)

        # Log artifacts to MLflow
        mlflow.log_artifacts(str(self.checkpoint_dir), artifact_path="checkpoints")
        mlflow.log_params({
            "checkpoint_dir": str(self.checkpoint_dir),
            "model_version": settings.model.version,
            "run_id": run_id,
        })

        logger.info(f"All artifacts saved → {self.checkpoint_dir}")

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    def _log_stage_metrics(self, stage: str, metrics: Dict) -> None:
        loggable = {
            f"{stage}_{k}": v
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        mlflow.log_metrics(loggable)


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────

def train_ensemble(
    checkpoint_dir: str = "checkpoints",
    use_synthetic: bool = False,
    data_dir: Optional[str] = None,
    gnn_epochs: int = 50,
    run_optuna: bool = False,
) -> Dict:
    """Convenience function — train full ensemble end-to-end."""
    trainer = EnsembleTrainer(
        checkpoint_dir=checkpoint_dir,
        use_synthetic=use_synthetic,
        data_dir=data_dir,
        gnn_epochs=gnn_epochs,
        run_optuna=run_optuna,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return trainer.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train fraud detection ensemble")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--gnn-epochs", type=int, default=50)
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (no Kaggle)")
    parser.add_argument("--optuna", action="store_true", help="Run hyperparameter search")
    args = parser.parse_args()

    metrics = train_ensemble(
        checkpoint_dir=args.checkpoint_dir,
        use_synthetic=args.synthetic,
        data_dir=args.data_dir,
        gnn_epochs=args.gnn_epochs,
        run_optuna=args.optuna,
    )
    logger.info(f"Final metrics: {metrics}")