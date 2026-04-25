"""
src/data/preprocessor.py
─────────────────────────────────────────────────────────────
Feature engineering, scaling, and SMOTE oversampling.

Pipeline:
  raw df → feature engineering → scaling → SMOTE (train only)
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from loguru import logger
from sklearn.preprocessing import RobustScaler


# ──────────────────────────────────────────────────────────────
# Feature columns
# ──────────────────────────────────────────────────────────────

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES = PCA_FEATURES + ["Amount", "Time"]
LABEL_COL = "Class"
TRANSACTION_ID_COL = "transaction_id"

ENGINEERED_FEATURES = [
    "amount_log",
    "amount_zscore",
    "hour_of_day",
    "is_night",
    "is_weekend_approx",
    "time_since_last_txn",
    "txn_velocity_1h",
    "txn_velocity_6h",
    "txn_velocity_24h",
    "amount_rolling_mean_10",
    "amount_rolling_std_10",
    "v1_v2_interaction",
    "v3_v4_interaction",
    "v14_abs",
    "v17_abs",
]

ALL_TABULAR_FEATURES = PCA_FEATURES + ["Amount", "Time"] + ENGINEERED_FEATURES


class FraudPreprocessor:
    """
    End-to-end preprocessing pipeline for fraud detection.

    Handles:
    - Feature engineering (velocity, log transforms, interactions)
    - RobustScaler (outlier-resistant for fraud data)
    - SMOTE oversampling (training only, never on val/test)
    """

    def __init__(
        self,
        smote_k_neighbors: int = 5,
        smote_random_state: int = 42,
        apply_smote: bool = True,
    ):
        self.smote_k_neighbors = smote_k_neighbors
        self.smote_random_state = smote_random_state
        self.apply_smote = apply_smote
        self.scaler = RobustScaler()
        self._is_fitted = False
        self.feature_names_: list[str] = []

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Engineer features, fit scaler, apply SMOTE.
        Call on TRAINING data only.

        Returns:
            (X_resampled, y_resampled)
        """
        df = self._engineer_features(df)
        X, y = self._extract_Xy(df)

        logger.info(f"Before SMOTE — shape: {X.shape}, fraud: {y.sum()} ({y.mean():.4f})")

        # Fit and transform scaler
        X_scaled = self.scaler.fit_transform(X)
        self._is_fitted = True

        if self.apply_smote and y.sum() < len(y) * 0.1:
            # SMOTE only when imbalance ratio > 10:1
            smote = SMOTE(
                k_neighbors=self.smote_k_neighbors,
                random_state=self.smote_random_state,
            )
            X_res, y_res = smote.fit_resample(X_scaled, y)
            logger.info(
                f"After SMOTE — shape: {X_res.shape}, "
                f"fraud: {y_res.sum()} ({y_res.mean():.4f})"
            )
        else:
            X_res, y_res = X_scaled, y

        return X_res, y_res

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Engineer features and apply fitted scaler.
        Call on VAL / TEST data.

        Returns:
            (X_scaled, y)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_transform() on training data first.")

        df = self._engineer_features(df)
        X, y = self._extract_Xy(df)
        X_scaled = self.scaler.transform(X)
        return X_scaled, y

    def transform_single(self, features: dict) -> np.ndarray:
        """
        Transform a single transaction dict for real-time inference.

        Returns:
            X_scaled shape (1, n_features)
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor not fitted.")

        df = pd.DataFrame([features])
        df = self._engineer_features(df)

        # Fill any missing engineered features with 0
        for col in ALL_TABULAR_FEATURES:
            if col not in df.columns:
                df[col] = 0.0

        X = df[self.feature_names_].values
        return self.scaler.transform(X)

    def get_feature_names(self) -> list[str]:
        return self.feature_names_

    # ──────────────────────────────────────────────────────────
    # Feature Engineering
    # ──────────────────────────────────────────────────────────

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── Amount transforms ──────────────────────────────
        df["amount_log"] = np.log1p(df["Amount"])
        df["amount_zscore"] = (
            (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1e-8)
        )

        # ── Time features (ULB Time is seconds since first txn) ──
        df["hour_of_day"] = (df["Time"] / 3600) % 24
        df["is_night"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] > 22)).astype(int)
        df["is_weekend_approx"] = ((df["Time"] // 86400) % 7 >= 5).astype(int)

        # ── Velocity features ──────────────────────────────
        df = df.sort_values("Time").reset_index(drop=True)
        df["time_since_last_txn"] = df["Time"].diff().fillna(0).clip(lower=0)
        df["txn_velocity_1h"] = self._rolling_count(df["Time"], window=3600)
        df["txn_velocity_6h"] = self._rolling_count(df["Time"], window=21600)
        df["txn_velocity_24h"] = self._rolling_count(df["Time"], window=86400)

        # ── Rolling statistics ─────────────────────────────
        df["amount_rolling_mean_10"] = (
            df["Amount"].rolling(10, min_periods=1).mean()
        )
        df["amount_rolling_std_10"] = (
            df["Amount"].rolling(10, min_periods=1).std().fillna(0)
        )

        # ── Feature interactions (domain-informed) ─────────
        # V1 and V2 are most predictive PCA components in ULB
        df["v1_v2_interaction"] = df["V1"] * df["V2"]
        df["v3_v4_interaction"] = df["V3"] * df["V4"]
        df["v14_abs"] = df["V14"].abs()  # V14 highly negative in fraud
        df["v17_abs"] = df["V17"].abs()  # V17 highly negative in fraud

        return df

    def _extract_Xy(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract feature matrix and label vector."""
        feature_cols = [c for c in ALL_TABULAR_FEATURES if c in df.columns]
        self.feature_names_ = feature_cols

        X = df[feature_cols].values.astype(np.float32)
        y = df[LABEL_COL].values.astype(np.int32) if LABEL_COL in df.columns else np.zeros(len(df), dtype=np.int32)
        return X, y

    @staticmethod
    def _rolling_count(times: pd.Series, window: float) -> pd.Series:
        """Count transactions in a rolling time window."""
        counts = []
        time_arr = times.values
        for i, t in enumerate(time_arr):
            count = np.sum((time_arr[:i] >= t - window) & (time_arr[:i] < t))
            counts.append(count)
        return pd.Series(counts, index=times.index, dtype=np.float32)