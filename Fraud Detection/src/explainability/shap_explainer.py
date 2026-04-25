"""
src/explainability/shap_explainer.py
─────────────────────────────────────────────────────────────
SHAP-based explainability for UAE VARA/CBUAE compliance.

Generates:
  - Force plots for individual flagged transactions
  - Summary plots for model audit reports
  - Top-3 SHAP reasons per prediction (API response)
  - Compliance explanation JSON (FATF-ready)

UAE VARA requirement: every automated decision must include
a human-readable justification traceable to input features.
"""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
import numpy as np
import shap
from loguru import logger


@dataclass
class SHAPExplanation:
    """Structured SHAP explanation for a single transaction."""
    transaction_id: str
    fraud_score: float
    base_value: float
    top_features: List[Dict]          # top-3 sorted by |SHAP value|
    all_shap_values: Dict[str, float]  # full feature → SHAP value map
    feature_values: Dict[str, float]   # raw feature values
    compliance_ref: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_version: str = "ensemble-v2.1.0"
    jurisdiction: str = "UAE"

    def to_api_response(self) -> Dict:
        """Format for FastAPI /predict response."""
        return {
            "top_3_shap_reasons": self.top_features[:3],
            "compliance_reference": self.compliance_ref,
            "model_version": self.model_version,
            "explanation_timestamp": self.timestamp,
        }

    def to_compliance_json(self) -> Dict:
        """Full compliance-grade explanation for audit trail."""
        return {
            "transaction_id": self.transaction_id,
            "fraud_score": round(self.fraud_score, 4),
            "base_fraud_rate": round(self.base_value, 4),
            "decision_justification": [
                {
                    "rank": i + 1,
                    "feature": f["feature"],
                    "observed_value": f["value"],
                    "shap_impact": f["impact"],
                    "direction": f["direction"],
                    "plain_language": self._plain_language(f),
                }
                for i, f in enumerate(self.top_features[:5])
            ],
            "model_version": self.model_version,
            "jurisdiction": self.jurisdiction,
            "compliance_reference": self.compliance_ref,
            "timestamp": self.timestamp,
            "regulatory_framework": "UAE VARA / CBUAE AML Guidelines 2024",
        }

    @staticmethod
    def _plain_language(feature_info: Dict) -> str:
        """Convert SHAP feature info to human-readable explanation."""
        feat = feature_info["feature"]
        direction = feature_info["direction"]
        impact_pct = abs(feature_info["impact"]) * 100

        direction_text = "increased" if direction == "increases_risk" else "decreased"

        FEATURE_DESCRIPTIONS = {
            "amount_log": "Transaction amount",
            "Amount": "Transaction amount",
            "V14": "Spending pattern anomaly (V14)",
            "V17": "Velocity pattern anomaly (V17)",
            "V12": "Geographic pattern anomaly (V12)",
            "time_since_last_txn": "Time since previous transaction",
            "txn_velocity_1h": "Number of transactions in past hour",
            "txn_velocity_24h": "Number of transactions in past 24 hours",
            "counterparty_fraud_rate": "Historical fraud rate for this counterparty",
            "merchant_fraud_rate": "Historical fraud rate for this merchant category",
            "is_night": "Night-time transaction",
        }

        feat_desc = FEATURE_DESCRIPTIONS.get(feat, f"Feature '{feat}'")
        return (
            f"{feat_desc} {direction_text} the fraud risk score by "
            f"{impact_pct:.1f} percentage points."
        )


class FraudSHAPExplainer:
    """
    SHAP TreeExplainer wrapper for the XGBoost fraud classifier.

    Designed for:
    - Real-time inference: cached TreeExplainer (~2ms per explanation)
    - Compliance reporting: structured JSON with plain-language reasons
    - Audit trail: full SHAP value vector stored per prediction
    """

    COMPLIANCE_PREFIX = "CBUAE-AML-2024"

    def __init__(
        self,
        model,                           # Trained XGBoostFraudClassifier
        feature_names: Optional[List[str]] = None,
        n_background_samples: int = 100,
    ):
        self.model = model
        self.feature_names = feature_names or []
        self._explainer: Optional[shap.TreeExplainer] = None
        self._background_data: Optional[np.ndarray] = None
        self.n_background = n_background_samples

    # ──────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────

    def fit(self, X_background: np.ndarray) -> "FraudSHAPExplainer":
        """
        Initialise SHAP TreeExplainer with background dataset.
        The background approximates E[f(X)] — the base rate.

        Call once after model training.
        """
        # Sample background for computational efficiency
        if len(X_background) > self.n_background:
            idx = np.random.choice(len(X_background), self.n_background, replace=False)
            self._background_data = X_background[idx]
        else:
            self._background_data = X_background

        logger.info(f"Initialising SHAP TreeExplainer (background={len(self._background_data)} samples)...")
        self._explainer = shap.TreeExplainer(
            self.model.model_,
            data=self._background_data,
            model_output="probability",
            feature_perturbation="interventional",
        )
        logger.info("SHAP explainer ready.")
        return self

    # ──────────────────────────────────────────────────────────
    # Single transaction explanation
    # ──────────────────────────────────────────────────────────

    def explain_transaction(
        self,
        X: np.ndarray,              # Shape (1, n_features) or (n_features,)
        transaction_id: str = "UNKNOWN",
        fraud_score: Optional[float] = None,
        feature_values: Optional[Dict] = None,
    ) -> SHAPExplanation:
        """
        Explain a single transaction prediction.

        Returns SHAPExplanation with top features and compliance JSON.
        """
        if self._explainer is None:
            raise RuntimeError("Call fit() before explaining.")

        X = X.reshape(1, -1) if X.ndim == 1 else X

        # Compute SHAP values
        shap_values = self._explainer.shap_values(X)

        # For binary classifiers, shap_values may be list [neg_class, pos_class]
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]  # Fraud class
        else:
            shap_vals = shap_values[0]

        base_value = self._explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[1])

        # Build feature → SHAP dict
        shap_dict = {
            name: float(val)
            for name, val in zip(self.feature_names, shap_vals)
        }

        # Sort by absolute SHAP value
        sorted_features = sorted(
            shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
        )

        # Format top features
        top_features = []
        for feat_name, shap_val in sorted_features[:10]:
            raw_val = (
                feature_values.get(feat_name, float("nan"))
                if feature_values
                else float(X[0, self.feature_names.index(feat_name)]
                           if feat_name in self.feature_names else "nan")
            )
            top_features.append({
                "feature": feat_name,
                "impact": round(abs(shap_val), 4),
                "direction": "increases_risk" if shap_val > 0 else "decreases_risk",
                "value": round(float(raw_val), 4) if not np.isnan(raw_val) else None,
                "shap_value": round(float(shap_val), 4),
            })

        # Generate compliance reference ID
        compliance_ref = self._generate_compliance_ref(transaction_id)

        return SHAPExplanation(
            transaction_id=transaction_id,
            fraud_score=fraud_score or float(self.model.predict_proba(X)[0]),
            base_value=float(base_value),
            top_features=top_features,
            all_shap_values=shap_dict,
            feature_values=feature_values or {},
            compliance_ref=compliance_ref,
        )

    # ──────────────────────────────────────────────────────────
    # Batch explanations
    # ──────────────────────────────────────────────────────────

    def explain_batch(
        self,
        X: np.ndarray,
        transaction_ids: Optional[List[str]] = None,
        fraud_scores: Optional[np.ndarray] = None,
    ) -> List[SHAPExplanation]:
        """Explain a batch of transactions (for offline audit reports)."""
        if transaction_ids is None:
            transaction_ids = [f"TXN-{i:07d}" for i in range(len(X))]

        explanations = []
        for i, (row, txn_id) in enumerate(zip(X, transaction_ids)):
            score = float(fraud_scores[i]) if fraud_scores is not None else None
            exp = self.explain_transaction(row, txn_id, score)
            explanations.append(exp)

        return explanations

    # ──────────────────────────────────────────────────────────
    # Visualisations
    # ──────────────────────────────────────────────────────────

    def plot_force(
        self,
        explanation: SHAPExplanation,
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generate SHAP force plot for a single transaction.
        Returns base64-encoded PNG string (for API embedding) or saves to file.
        """
        if self._explainer is None:
            return None

        try:
            shap_vals = np.array([
                explanation.all_shap_values.get(f, 0.0)
                for f in self.feature_names
            ])
            feat_vals = np.array([
                explanation.feature_values.get(f, 0.0)
                for f in self.feature_names
            ])

            # Waterfall chart (cleaner than force plot for dashboards)
            fig, ax = plt.subplots(figsize=(12, 6))
            top_n = 10
            sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]
            top_vals = shap_vals[sorted_idx]
            top_names = [self.feature_names[i] for i in sorted_idx]

            colors = ["#E74C3C" if v > 0 else "#2ECC71" for v in top_vals]
            bars = ax.barh(range(top_n), top_vals[::-1], color=colors[::-1], alpha=0.85)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_names[::-1], fontsize=10)
            ax.axvline(x=0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP value (impact on fraud probability)", fontsize=11)
            ax.set_title(
                f"SHAP Explanation — {explanation.transaction_id}\n"
                f"Fraud Score: {explanation.fraud_score:.3f} | "
                f"Compliance Ref: {explanation.compliance_ref}",
                fontsize=12, fontweight="bold"
            )
            ax.text(
                0.98, 0.02,
                "🇦🇪 UAE VARA Compliant | CBUAE AML 2024",
                transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="gray",
            )
            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close()
                return output_path
            else:
                # Return as base64 for API embedding
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                plt.close()
                return base64.b64encode(buf.getvalue()).decode("utf-8")

        except Exception as e:
            logger.error(f"Force plot failed: {e}")
            return None

    def plot_summary(
        self,
        X: np.ndarray,
        output_path: str = "shap_summary.png",
        max_display: int = 20,
    ) -> None:
        """SHAP beeswarm summary plot for model audit reports."""
        if self._explainer is None:
            return

        shap_values = self._explainer.shap_values(X[:500])
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            sv, X[:500],
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
            plot_type="dot",
        )
        plt.title("SHAP Feature Importance — Fraud Detection Model", fontsize=13)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot saved → {output_path}")

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    def _generate_compliance_ref(self, transaction_id: str) -> str:
        """Generate a traceable compliance reference ID."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        short_id = transaction_id.replace("TXN-", "")[-8:].upper()
        return f"{self.COMPLIANCE_PREFIX}-{ts}-{short_id}"