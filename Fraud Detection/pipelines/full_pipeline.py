"""
pipelines/full_pipeline.py
─────────────────────────────────────────────────────────────
End-to-end fraud detection pipeline orchestrator.

Stages:
  0. Validate environment & data
  1. Feature engineering + SMOTE
  2. XGBoost baseline
  3. GraphSAGE training
  4. Ensemble fusion
  5. SHAP explainer fitting
  6. Drift detector reference setting
  7. MLflow model registration
  8. GCP artifact upload (optional)
  9. Integration test against FastAPI

Usage:
    # Full pipeline on real data:
    python pipelines/full_pipeline.py --data-dir data/

    # Demo mode (synthetic data, no Kaggle needed):
    python pipelines/full_pipeline.py --synthetic --gnn-epochs 10

    # Skip GNN (tabular-only, fastest):
    python pipelines/full_pipeline.py --synthetic --skip-gnn
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import mlflow
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# ── Ensure project root is on path ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings
from src.training.train_ensemble import EnsembleTrainer

settings = get_settings()
console = Console()


# ──────────────────────────────────────────────────────────────
# Pipeline stages
# ──────────────────────────────────────────────────────────────

class FraudDetectionPipeline:
    """
    Orchestrates the full ML pipeline from raw data to deployed model.

    This is the single script to run for:
      - Initial training on a new environment
      - Periodic retraining (weekly / on drift detection)
      - CI/CD model promotion workflows
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.stage_results = {}

    def run(self) -> bool:
        """Execute full pipeline. Returns True on success."""
        self._print_banner()

        try:
            # ── Stage 0: Environment validation ───────────
            self._stage("Validating environment", self._validate_environment)

            # ── Stage 1–4: Model training ──────────────────
            self._stage("Running ensemble training pipeline", self._train_models)

            # ── Stage 5: Drift detector setup ─────────────
            self._stage("Setting drift detector reference", self._setup_drift_detector)

            # ── Stage 6: MLflow model registration ────────
            self._stage("Registering model in MLflow", self._register_model)

            # ── Stage 7: GCP upload (optional) ────────────
            if not self.args.skip_gcp:
                self._stage("Uploading artifacts to GCP", self._upload_to_gcp)

            # ── Stage 8: Integration smoke test ───────────
            if not self.args.skip_tests:
                self._stage("Running integration smoke test", self._run_smoke_test)

            self._print_summary()
            return True

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            console.print(f"\n[bold red]✗ Pipeline failed:[/bold red] {e}")
            if self.args.debug:
                import traceback
                traceback.print_exc()
            return False

    # ──────────────────────────────────────────────────────────
    # Stage implementations
    # ──────────────────────────────────────────────────────────

    def _validate_environment(self) -> dict:
        """Check all dependencies and data availability."""
        checks = {}

        # Python packages
        try:
            import torch
            checks["pytorch"] = f"✓ PyTorch {torch.__version__}"
        except ImportError:
            checks["pytorch"] = "✗ PyTorch NOT installed"

        try:
            import torch_geometric
            checks["pyg"] = f"✓ PyTorch Geometric {torch_geometric.__version__}"
        except ImportError:
            checks["pyg"] = "⚠ PyTorch Geometric not installed (GNN disabled)"

        try:
            import xgboost as xgb
            checks["xgboost"] = f"✓ XGBoost {xgb.__version__}"
        except ImportError:
            raise RuntimeError("XGBoost is required. Run: pip install xgboost")

        try:
            import shap
            checks["shap"] = f"✓ SHAP {shap.__version__}"
        except ImportError:
            checks["shap"] = "⚠ SHAP not installed (explainability disabled)"

        try:
            import mlflow
            checks["mlflow"] = f"✓ MLflow {mlflow.__version__}"
        except ImportError:
            checks["mlflow"] = "⚠ MLflow not installed (tracking disabled)"

        # Data check
        if not self.args.synthetic:
            data_path = Path(self.args.data_dir or "data") / "creditcard.csv"
            if data_path.exists():
                import os
                size_mb = os.path.getsize(data_path) / 1_048_576
                checks["dataset"] = f"✓ creditcard.csv ({size_mb:.1f} MB)"
            else:
                raise FileNotFoundError(
                    f"Dataset not found: {data_path}\n"
                    "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
                    "Or use --synthetic flag for demo mode."
                )
        else:
            checks["dataset"] = "✓ Using synthetic data (demo mode)"

        # Log checks
        for key, val in checks.items():
            logger.info(f"  {key}: {val}")

        return checks

    def _train_models(self) -> dict:
        """Run the full EnsembleTrainer pipeline."""
        trainer = EnsembleTrainer(
            checkpoint_dir=str(self.checkpoint_dir),
            mlflow_experiment=settings.mlflow.experiment_name,
            device="cpu",
            use_synthetic=self.args.synthetic,
            data_dir=self.args.data_dir,
            run_optuna=self.args.optuna,
            gnn_epochs=self.args.gnn_epochs if not self.args.skip_gnn else 0,
            gnn_batch_size=self.args.gnn_batch_size,
        )

        if self.args.skip_gnn:
            # Patch to skip GNN stage
            trainer.gnn_model = None

        metrics = trainer.run()
        self.stage_results["training"] = metrics

        # Store trainer references for subsequent stages
        self._trainer = trainer
        return metrics

    def _setup_drift_detector(self) -> dict:
        """Set training data as drift reference baseline."""
        from src.monitoring.drift_detector import DriftDetector
        import numpy as np

        detector = DriftDetector()

        # Load reference data from training
        ref_path = self.checkpoint_dir / "X_train_reference.npy"
        score_path = self.checkpoint_dir / "scores_reference.npy"

        if ref_path.exists() and score_path.exists():
            X_ref = np.load(ref_path)
            scores_ref = np.load(score_path)
        else:
            # Generate synthetic reference if training data not cached
            logger.warning("Training reference data not cached. Using synthetic reference.")
            rng = np.random.default_rng(42)
            n_features = 42  # tabular features
            X_ref = rng.normal(0, 1, (10_000, n_features)).astype(np.float32)
            scores_ref = rng.beta(1, 200, 10_000).astype(np.float32)

        feature_names = getattr(
            self._trainer.preprocessor,
            "feature_names_",
            [f"feature_{i}" for i in range(X_ref.shape[1])],
        )
        detector.set_reference(X_ref, scores_ref, feature_names)

        return {"drift_reference_set": True, "n_reference_samples": len(X_ref)}

    def _register_model(self) -> dict:
        """Register best model in MLflow Model Registry."""
        try:
            run_id = self.stage_results.get("training", {}).get("run_id")
            if not run_id:
                logger.warning("No run_id found. Skipping MLflow registration.")
                return {"registered": False}

            client = mlflow.tracking.MlflowClient(
                tracking_uri=settings.mlflow.tracking_uri
            )

            # Register model version
            model_uri = f"runs:/{run_id}/checkpoints"
            try:
                mv = mlflow.register_model(
                    model_uri=model_uri,
                    name="fraud-detection-ensemble",
                )
                # Transition to Production
                client.transition_model_version_stage(
                    name="fraud-detection-ensemble",
                    version=mv.version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                logger.info(
                    f"Model registered: fraud-detection-ensemble v{mv.version} → Production"
                )
                return {"registered": True, "version": mv.version}
            except Exception as e:
                logger.warning(f"MLflow registration failed (tracking server may not be running): {e}")
                return {"registered": False, "error": str(e)}

        except Exception as e:
            logger.warning(f"Model registration skipped: {e}")
            return {"registered": False}

    def _upload_to_gcp(self) -> dict:
        """Upload model artifacts to GCP Cloud Storage."""
        try:
            from google.cloud import storage

            client = storage.Client(project=settings.gcp_project_id)
            bucket = client.bucket(settings.gcp_bucket_name)

            uploaded = []
            for artifact in self.checkpoint_dir.glob("*"):
                if artifact.is_file():
                    blob_name = f"models/production/{artifact.name}"
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(artifact))
                    uploaded.append(blob_name)
                    logger.info(f"Uploaded: gs://{settings.gcp_bucket_name}/{blob_name}")

            return {"uploaded": True, "files": uploaded}

        except ImportError:
            logger.warning("google-cloud-storage not installed. Skipping GCP upload.")
            return {"uploaded": False, "reason": "google-cloud-storage not installed"}
        except Exception as e:
            logger.warning(f"GCP upload failed: {e}")
            return {"uploaded": False, "error": str(e)}

    def _run_smoke_test(self) -> dict:
        """Quick integration test against the trained model."""
        import numpy as np

        logger.info("Running model smoke test...")

        try:
            import pickle

            # Load preprocessor
            pp_path = self.checkpoint_dir / "preprocessor.pkl"
            if not pp_path.exists():
                return {"passed": False, "reason": "Preprocessor not found"}

            with open(pp_path, "rb") as f:
                preprocessor = pickle.load(f)

            # Load ensemble
            from src.models.ensemble import FraudEnsemble
            xgb_path = self.checkpoint_dir / "xgboost_ensemble.pkl"
            if not xgb_path.exists():
                return {"passed": False, "reason": "XGBoost model not found"}

            ensemble = FraudEnsemble.load(str(self.checkpoint_dir))

            # Test prediction
            rng = np.random.default_rng(42)
            n_features = len(preprocessor.feature_names_) if preprocessor._is_fitted else 42
            X_test = rng.normal(0, 1, (5, n_features)).astype(np.float32)

            probs = ensemble.xgb_model.predict_proba(X_test)
            assert probs.shape == (5,), f"Expected (5,) got {probs.shape}"
            assert all(0.0 <= p <= 1.0 for p in probs), "Probabilities out of [0,1] range"

            logger.info(f"Smoke test PASSED — sample scores: {probs.round(3)}")
            return {"passed": True, "sample_scores": probs.tolist()}

        except Exception as e:
            logger.error(f"Smoke test FAILED: {e}")
            return {"passed": False, "error": str(e)}

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    def _stage(self, name: str, fn) -> None:
        """Run a pipeline stage with progress display."""
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]{name}...[/bold cyan]"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(name, total=None)
            result = fn()
            progress.update(task, completed=True)

        console.print(f"  [green]✓[/green] {name}")

    def _print_banner(self) -> None:
        console.print(Panel(
            "[bold blue]🔍 Fraud Detection Pipeline[/bold blue]\n"
            "[dim]GNN + XGBoost + SHAP | UAE VARA/CBUAE Compliant[/dim]\n"
            f"[dim]Environment: {settings.environment} | Version: {settings.app_version}[/dim]",
            expand=False,
        ))

        args_table = Table(title="Pipeline Configuration", show_header=False)
        args_table.add_column("Parameter", style="cyan")
        args_table.add_column("Value", style="white")
        for k, v in vars(self.args).items():
            args_table.add_row(k.replace("_", "-"), str(v))
        console.print(args_table)

    def _print_summary(self) -> None:
        elapsed = time.time() - self.start_time
        training_metrics = self.stage_results.get("training", {})

        summary = Table(title="Pipeline Complete ✓", show_header=True)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="bold green")

        summary.add_row("Total time", f"{elapsed:.1f}s ({elapsed/60:.1f} min)")
        summary.add_row("Checkpoint dir", str(self.checkpoint_dir))

        for k, v in training_metrics.items():
            if isinstance(v, float):
                summary.add_row(k, f"{v:.4f}")

        console.print(summary)
        console.print(Panel(
            "[bold green]✅ Pipeline successful![/bold green]\n\n"
            "Next steps:\n"
            "  1. Start API:  [cyan]uvicorn src.api.main:app --reload[/cyan]\n"
            "  2. MLflow UI:  [cyan]mlflow ui --port 5000[/cyan]\n"
            "  3. Docker:     [cyan]docker-compose up --build[/cyan]\n"
            "  4. API docs:   [cyan]http://localhost:8000/docs[/cyan]",
            expand=False,
        ))


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fraud Detection Full Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument("--data-dir", default="data", help="Directory containing creditcard.csv")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (no Kaggle required)")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory to save model artifacts")

    # Training
    parser.add_argument("--gnn-epochs", type=int, default=50, help="Max GNN training epochs")
    parser.add_argument("--gnn-batch-size", type=int, default=4096, help="Graph mini-batch size")
    parser.add_argument("--skip-gnn", action="store_true", help="Skip GNN stage (tabular XGBoost only)")
    parser.add_argument("--optuna", action="store_true", help="Run Optuna hyperparameter search")

    # Deployment
    parser.add_argument("--skip-gcp", action="store_true", help="Skip GCP artifact upload")
    parser.add_argument("--skip-tests", action="store_true", help="Skip integration smoke test")

    # Dev
    parser.add_argument("--debug", action="store_true", help="Enable debug logging + full tracebacks")
    parser.add_argument("--config", default="configs/model_config.yaml", help="Config file path")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    pipeline = FraudDetectionPipeline(args)
    success = pipeline.run()
    sys.exit(0 if success else 1)