"""
src/models/xgboost_model.py
─────────────────────────────────────────────────────────────
XGBoost baseline fraud classifier.

Establishes performance floor before GNN embedding injection.
Optimised via Optuna with MLflow experiment tracking.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
import xgboost as xgb
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class XGBoostFraudClassifier:
    """
    XGBoost classifier optimised for imbalanced fraud detection.

    Key design decisions:
    - scale_pos_weight: handles 578:1 imbalance in ULB dataset
    - hist tree method: fastest for tabular data at this scale
    - eval_metric: aucpr (precision-recall AUC — better than ROC for imbalance)
    - Optuna: 50 trials of Bayesian hyperparameter search
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 578.0,
        min_child_weight: int = 5,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
        n_jobs: int = -1,
        device: str = "cpu",
    ):
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method="hist",
            eval_metric=["logloss", "aucpr"],
            use_label_encoder=False,
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.model_: Optional[xgb.XGBClassifier] = None
        self.feature_names_: list[str] = []
        self.threshold_: float = 0.5

    # ──────────────────────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> "XGBoostFraudClassifier":
        """
        Train XGBoost with early stopping on validation set.

        Returns self for method chaining.
        """
        if feature_names:
            self.feature_names_ = feature_names

        self.model_ = xgb.XGBClassifier(**self.params)
        self.model_.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=100,
        )

        logger.info(
            f"XGBoost trained — best iteration: {self.model_.best_iteration} | "
            f"best score: {self.model_.best_score:.4f}"
        )
        return self

    def optimise_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = "f1",
    ) -> float:
        """
        Find the optimal decision threshold maximising F1 (or recall/precision).
        Returns optimal threshold.
        """
        probs = self.predict_proba(X_val)
        best_thresh, best_score = 0.5, 0.0

        for thresh in np.arange(0.1, 0.95, 0.01):
            preds = (probs >= thresh).astype(int)
            if metric == "f1":
                score = f1_score(y_val, preds, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_val, preds, zero_division=0)
            else:
                score = precision_score(y_val, preds, zero_division=0)

            if score > best_score:
                best_score, best_thresh = score, thresh

        self.threshold_ = best_thresh
        logger.info(f"Optimised threshold: {best_thresh:.2f} (val {metric}: {best_score:.4f})")
        return best_thresh

    # ──────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fraud probability for each transaction."""
        return self.model_.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Return binary fraud predictions."""
        thresh = threshold if threshold is not None else self.threshold_
        return (self.predict_proba(X) >= thresh).astype(int)

    # ──────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: Optional[float] = None,
        split_name: str = "test",
    ) -> Dict[str, float]:
        """Full evaluation report with all fraud-detection metrics."""
        probs = self.predict_proba(X)
        preds = self.predict(X, threshold)

        metrics = {
            "auc_roc": roc_auc_score(y, probs),
            "auc_pr": average_precision_score(y, probs),
            "f1": f1_score(y, preds, zero_division=0),
            "precision": precision_score(y, preds, zero_division=0),
            "recall": recall_score(y, preds, zero_division=0),
            "threshold": self.threshold_,
        }

        logger.info(f"\n{'='*50}")
        logger.info(f"XGBoost Evaluation — {split_name.upper()}")
        logger.info(f"  AUC-ROC:    {metrics['auc_roc']:.4f}")
        logger.info(f"  AUC-PR:     {metrics['auc_pr']:.4f}")
        logger.info(f"  F1 Score:   {metrics['f1']:.4f}")
        logger.info(f"  Precision:  {metrics['precision']:.4f}")
        logger.info(f"  Recall:     {metrics['recall']:.4f}")
        logger.info(f"\n{classification_report(y, preds, target_names=['Legit', 'Fraud'])}")

        return metrics

    # ──────────────────────────────────────────────────────────
    # Serialisation
    # ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"XGBoost model saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostFraudClassifier":
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"XGBoost model loaded ← {path}")
        return model


# ──────────────────────────────────────────────────────────────
# Optuna hyperparameter optimisation
# ──────────────────────────────────────────────────────────────

def optimise_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    study_name: str = "xgboost-fraud",
) -> Dict:
    """
    Bayesian hyperparameter search with Optuna.
    Returns best hyperparameters dict.
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 100.0, 1000.0),
        }

        clf = XGBoostFraudClassifier(**params)
        clf.fit(X_train, y_train, X_val, y_val)
        probs = clf.predict_proba(X_val)
        return average_precision_score(y_val, probs)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    logger.info(f"Optuna best AUC-PR: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    return study.best_params