"""
src/training/train_xgboost.py
─────────────────────────────────────────────────────────────
XGBoost baseline training with full MLflow tracking.
Logs: params, metrics, model artifact, SHAP plots, confusion matrix.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.loader import FraudDataLoader
from src.data.preprocessor import FraudPreprocessor
from src.models.xgboost_model import XGBoostFraudClassifier, optimise_xgboost


def run_xgboost_training(
    data_dir: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    run_optuna: bool = False,
    n_optuna_trials: int = 30,
    mlflow_experiment: str = "fraud-detection",
    run_name: str = "xgboost-baseline",
    random_state: int = 42,
    use_synthetic: bool = False,
) -> Tuple[XGBoostFraudClassifier, Dict]:
    """
    End-to-end XGBoost training run.

    Returns:
        (trained_classifier, metrics_dict)
    """
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(mlflow_experiment)

    # ── 1. Load & split data ───────────────────────────────
    logger.info("Loading dataset...")
    loader = FraudDataLoader(data_dir=data_dir)

    if use_synthetic:
        from src.data.loader import generate_synthetic_dataset
        df = generate_synthetic_dataset(n_samples=50_000, random_state=random_state)
        loader._df = df

    train_df, val_df, test_df = loader.get_splits(random_state=random_state)
    class_weights = loader.get_class_weights()
    scale_pos_weight = class_weights[1]  # neg/pos ratio

    # ── 2. Feature engineering ─────────────────────────────
    logger.info("Engineering features...")
    preprocessor = FraudPreprocessor(apply_smote=True)
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_val, y_val = preprocessor.transform(val_df)
    X_test, y_test = preprocessor.transform(test_df)

    feature_names = preprocessor.get_feature_names()
    logger.info(f"Feature matrix: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # ── 3. Hyperparameter optimisation ─────────────────
        if run_optuna:
            logger.info(f"Running Optuna hyperparameter search ({n_optuna_trials} trials)...")
            best_params = optimise_xgboost(
                X_train, y_train, X_val, y_val, n_trials=n_optuna_trials
            )
            mlflow.log_params({"optuna_trials": n_optuna_trials, **best_params})
        else:
            best_params = {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": scale_pos_weight,
            }

        # ── 4. Train classifier ────────────────────────────
        logger.info("Training XGBoost classifier...")
        t0 = time.time()

        clf = XGBoostFraudClassifier(**best_params)
        clf.fit(X_train, y_train, X_val, y_val, feature_names=feature_names)
        train_time = time.time() - t0

        # ── 5. Optimise decision threshold ─────────────────
        clf.optimise_threshold(X_val, y_val, metric="f1")

        # ── 6. Evaluate ────────────────────────────────────
        val_metrics = clf.evaluate(X_val, y_val, split_name="val")
        test_metrics = clf.evaluate(X_test, y_test, split_name="test")

        # ── 7. Log to MLflow ───────────────────────────────
        mlflow.log_params({
            "model_type": "XGBoost",
            "n_features": X_train.shape[1],
            "n_train": len(y_train),
            "n_val": len(y_val),
            "n_test": len(y_test),
            "fraud_rate_train": float(y_train.mean()),
            "fraud_rate_test": float(y_test.mean()),
            "applied_smote": True,
            "train_time_seconds": round(train_time, 2),
            **best_params,
        })

        mlflow.log_metrics({
            "val_auc_roc": val_metrics["auc_roc"],
            "val_auc_pr": val_metrics["auc_pr"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "test_auc_roc": test_metrics["auc_roc"],
            "test_auc_pr": test_metrics["auc_pr"],
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "optimal_threshold": clf.threshold_,
        })

        # Log feature importance
        if clf.model_ is not None:
            fi_dict = dict(zip(
                feature_names,
                clf.model_.feature_importances_.tolist()
            ))
            mlflow.log_dict(fi_dict, "feature_importances.json")
            _log_top_features(fi_dict)

        # Log model artifact
        mlflow.xgboost.log_model(
            clf.model_,
            artifact_path="xgboost_model",
            registered_model_name="fraud-xgboost-baseline",
        )

        # ── 8. Save locally ────────────────────────────────
        clf.save(checkpoint_path / "xgboost_baseline.pkl")

        # Save preprocessor
        import pickle
        with open(checkpoint_path / "preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)

        logger.info(f"XGBoost training complete. MLflow run_id: {run_id}")
        logger.info(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f} | AUC-PR: {test_metrics['auc_pr']:.4f}")

        return clf, {**test_metrics, "run_id": run_id, "model_type": "xgboost"}


def _log_top_features(feature_importance: Dict, top_n: int = 20) -> None:
    """Log top-N most important features."""
    sorted_features = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    )[:top_n]
    logger.info(f"\nTop {top_n} features by XGBoost importance:")
    for i, (feat, score) in enumerate(sorted_features, 1):
        logger.info(f"  {i:2d}. {feat:<35} {score:.4f}")