"""
src/data/loader.py
─────────────────────────────────────────────────────────────
Dataset loading for ULB Credit Card Fraud Detection dataset
(284,807 transactions, 492 fraud cases).

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedShuffleSplit


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
ULB_FILENAME = "creditcard.csv"
EXPECTED_ROWS = 284_807
EXPECTED_COLS = 31
LABEL_COL = "Class"


# ──────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────

class FraudDataLoader:
    """
    Loads and validates the ULB Credit Card Fraud Detection dataset.

    The dataset has:
    - 28 PCA-anonymised features (V1–V28)
    - Time: seconds elapsed since first transaction
    - Amount: transaction amount in EUR
    - Class: 0=legitimate, 1=fraud (492 fraud out of 284,807)
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        filename: str = ULB_FILENAME,
    ):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.filepath = self.data_dir / filename
        self._df: Optional[pd.DataFrame] = None

    # ──────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────

    def load(self, force_reload: bool = False) -> pd.DataFrame:
        """Load dataset with validation and caching."""
        if self._df is not None and not force_reload:
            return self._df

        if not self.filepath.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.filepath}.\n"
                f"Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
                f"Place creditcard.csv in: {self.data_dir}/"
            )

        logger.info(f"Loading dataset from {self.filepath}")
        df = pd.read_csv(self.filepath)

        self._validate(df)
        df = self._add_transaction_ids(df)

        self._df = df
        logger.info(
            f"Loaded {len(df):,} transactions | "
            f"Fraud: {df[LABEL_COL].sum():,} ({df[LABEL_COL].mean() * 100:.3f}%)"
        )
        return df

    def get_splits(
        self,
        test_size: float = 0.20,
        val_size: float = 0.10,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Stratified train/val/test split preserving fraud ratio.

        Returns:
            (train_df, val_df, test_df)
        """
        df = self.load()

        # First split: train+val vs test
        sss1 = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_val_idx, test_idx = next(sss1.split(df, df[LABEL_COL]))
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # Second split: train vs val
        effective_val = val_size / (1 - test_size)
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=effective_val, random_state=random_state
        )
        train_idx, val_idx = next(
            sss2.split(train_val_df, train_val_df[LABEL_COL])
        )
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

        logger.info(
            f"Splits — Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}"
        )
        self._log_fraud_ratios(train_df, val_df, test_df)

        return train_df, val_df, test_df

    def get_feature_names(self) -> list[str]:
        """Return the base feature column names (before engineering)."""
        df = self.load()
        return [c for c in df.columns if c not in [LABEL_COL, "transaction_id"]]

    def get_class_weights(self) -> dict:
        """Return class weights for handling imbalance."""
        df = self.load()
        n_neg = (df[LABEL_COL] == 0).sum()
        n_pos = (df[LABEL_COL] == 1).sum()
        return {0: 1.0, 1: n_neg / n_pos}

    # ──────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> None:
        """Validate dataset shape and schema."""
        assert df.shape[1] == EXPECTED_COLS, (
            f"Expected {EXPECTED_COLS} columns, got {df.shape[1]}"
        )
        assert LABEL_COL in df.columns, f"Missing label column '{LABEL_COL}'"
        assert df[LABEL_COL].isin([0, 1]).all(), "Label column must be binary"
        assert df.isnull().sum().sum() == 0, "Dataset contains null values"

        missing_pca = [f"V{i}" for i in range(1, 29) if f"V{i}" not in df.columns]
        assert not missing_pca, f"Missing PCA features: {missing_pca}"

        logger.info(f"Dataset validation passed: {df.shape}")

    def _add_transaction_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add deterministic transaction IDs based on row hash."""
        def make_id(row) -> str:
            content = f"{row['Time']:.0f}_{row['Amount']:.4f}_{row.name}"
            return "TXN-" + hashlib.md5(content.encode()).hexdigest()[:12].upper()

        df = df.copy()
        df["transaction_id"] = [
            f"TXN-{i:08d}" for i in range(len(df))
        ]
        return df

    def _log_fraud_ratios(self, train, val, test) -> None:
        for name, split in [("Train", train), ("Val", val), ("Test", test)]:
            rate = split[LABEL_COL].mean() * 100
            logger.info(f"  {name} fraud rate: {rate:.3f}%")


# ──────────────────────────────────────────────────────────────
# Synthetic PaySim-style generator (for demo without Kaggle)
# ──────────────────────────────────────────────────────────────

def generate_synthetic_dataset(
    n_samples: int = 10_000,
    fraud_ratio: float = 0.002,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic dataset mimicking the ULB schema.
    Use this for testing without the real dataset.
    """
    rng = np.random.default_rng(random_state)
    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    def make_txns(n: int, label: int) -> pd.DataFrame:
        rows = {}
        rows["Time"] = rng.uniform(0, 172_800, n)  # 48 hours
        rows["Amount"] = np.where(
            label == 1,
            rng.exponential(scale=300, size=n),
            rng.exponential(scale=85, size=n),
        )
        for i in range(1, 29):
            mean = rng.uniform(-2, 2) if label == 1 else 0.0
            rows[f"V{i}"] = rng.normal(mean, 1.5 if label == 1 else 1.0, n)
        rows["Class"] = label
        return pd.DataFrame(rows)

    df = pd.concat(
        [make_txns(n_legit, 0), make_txns(n_fraud, 1)],
        ignore_index=True,
    ).sample(frac=1, random_state=random_state).reset_index(drop=True)

    df["transaction_id"] = [f"TXN-SYN-{i:07d}" for i in range(len(df))]

    logger.info(
        f"Generated synthetic dataset: {len(df):,} txns | "
        f"Fraud: {df['Class'].sum()} ({df['Class'].mean() * 100:.2f}%)"
    )
    return df