"""
src/monitoring/drift_detector.py
─────────────────────────────────────────────────────────────
Population Stability Index (PSI) based model drift detection.

Monitors:
  - Feature distribution drift (PSI per feature)
  - Prediction score drift (score distribution shift)
  - Fraud rate trend deviation
  - Latency degradation

PSI thresholds (industry standard):
  PSI < 0.10  → Insignificant change (green)
  PSI 0.10–0.25 → Some change (yellow — monitor)
  PSI > 0.25  → Significant change (red — retrain)

Alerting:
  - Slack webhook notification
  - MLflow metric logging
  - Database status record
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.api.schemas import ModelDriftResponse
from src.config import get_settings

settings = get_settings()


# ──────────────────────────────────────────────────────────────
# PSI computation
# ──────────────────────────────────────────────────────────────

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Compute Population Stability Index between two distributions.

    PSI = Σ (P_current - P_reference) × ln(P_current / P_reference)

    Args:
        reference: Reference distribution (training data)
        current:   Current distribution (recent predictions)
        n_bins:    Number of histogram bins
        eps:       Epsilon to avoid log(0)

    Returns:
        PSI score (float)
    """
    # Create bins from reference distribution
    breakpoints = np.nanpercentile(
        reference,
        np.linspace(0, 100, n_bins + 1)
    )
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    def to_distribution(data: np.ndarray) -> np.ndarray:
        counts, _ = np.histogram(data, bins=breakpoints)
        pct = counts / (len(data) + eps)
        return pct + eps  # avoid log(0)

    p_ref = to_distribution(reference)
    p_cur = to_distribution(current)

    psi = np.sum((p_cur - p_ref) * np.log(p_cur / p_ref))
    return float(psi)


def compute_js_divergence(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-8,
) -> float:
    """
    Jensen-Shannon divergence as an alternative drift metric.
    Bounded in [0, 1] — more numerically stable than KL divergence.
    """
    breakpoints = np.nanpercentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    def hist(data):
        c, _ = np.histogram(data, bins=breakpoints)
        p = c / (c.sum() + eps)
        return p + eps

    p = hist(reference)
    q = hist(current)
    m = (p + q) / 2

    js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return float(np.clip(js, 0, 1))


# ──────────────────────────────────────────────────────────────
# Drift detector
# ──────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Monitors model input/output drift using PSI.

    Usage:
        detector = DriftDetector()
        detector.set_reference(X_train, fraud_scores_train)
        report = detector.detect(X_current, fraud_scores_current)
    """

    REFERENCE_PATH = Path("checkpoints/drift_reference.npz")
    STATUS_PATH = Path("checkpoints/drift_status.json")

    # Features to monitor (most predictive + high drift risk)
    MONITORED_FEATURES = [
        "amount_log",
        "V14",
        "V17",
        "V12",
        "time_since_last_txn",
        "txn_velocity_1h",
        "counterparty_fraud_rate",
        "amount_zscore",
    ]

    def __init__(
        self,
        psi_threshold_warning: float = 0.10,
        psi_threshold_critical: float = 0.25,
        window_hours: int = 24,
    ):
        self.psi_threshold_warning = psi_threshold_warning
        self.psi_threshold_critical = psi_threshold_critical
        self.window_hours = window_hours

        self._reference_data: Optional[Dict[str, np.ndarray]] = None
        self._reference_scores: Optional[np.ndarray] = None
        self._load_reference()

    # ──────────────────────────────────────────────────────────
    # Reference management
    # ──────────────────────────────────────────────────────────

    def set_reference(
        self,
        X_reference: np.ndarray,
        fraud_scores: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Store reference distributions (from training data).
        Call once after model training.
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_reference.shape[1])]

        self._reference_data = {}
        for i, name in enumerate(feature_names):
            if name in self.MONITORED_FEATURES or i < 8:
                self._reference_data[name] = X_reference[:, i].copy()

        self._reference_data["fraud_score"] = fraud_scores.copy()
        self._reference_scores = fraud_scores.copy()

        # Persist to disk
        self.REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.REFERENCE_PATH,
            **{k: v for k, v in self._reference_data.items()}
        )
        logger.info(
            f"Drift reference set — {len(fraud_scores):,} samples | "
            f"monitoring {len(self._reference_data)} features"
        )

    def _load_reference(self) -> None:
        """Load reference distributions from disk if available."""
        if self.REFERENCE_PATH.exists():
            try:
                data = np.load(self.REFERENCE_PATH)
                self._reference_data = {k: data[k] for k in data.files}
                self._reference_scores = self._reference_data.get("fraud_score")
                logger.info(f"Drift reference loaded from {self.REFERENCE_PATH}")
            except Exception as e:
                logger.warning(f"Could not load drift reference: {e}")

    # ──────────────────────────────────────────────────────────
    # Drift detection
    # ──────────────────────────────────────────────────────────

    def detect(
        self,
        X_current: np.ndarray,
        fraud_scores_current: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> ModelDriftResponse:
        """
        Compute PSI between reference and current distributions.

        Returns:
            ModelDriftResponse with status, per-feature PSI, recommendation.
        """
        if self._reference_data is None:
            return self._no_reference_response()

        if feature_names is None:
            feature_names = list(self._reference_data.keys())

        feature_results = []
        max_psi = 0.0

        # Per-feature PSI
        for i, name in enumerate(feature_names):
            if name not in self._reference_data:
                continue
            if i >= X_current.shape[1]:
                break

            ref_dist = self._reference_data[name]
            cur_dist = X_current[:, i]

            psi = compute_psi(ref_dist, cur_dist)
            max_psi = max(max_psi, psi)

            if psi >= self.psi_threshold_critical:
                feat_status = "critical"
            elif psi >= self.psi_threshold_warning:
                feat_status = "warning"
            else:
                feat_status = "stable"

            feature_results.append({
                "feature": name,
                "psi": round(psi, 4),
                "status": feat_status,
                "js_divergence": round(compute_js_divergence(ref_dist, cur_dist), 4),
            })

        # Score distribution drift
        if self._reference_scores is not None:
            score_psi = compute_psi(self._reference_scores, fraud_scores_current)
            max_psi = max(max_psi, score_psi)
            feature_results.append({
                "feature": "fraud_score_distribution",
                "psi": round(score_psi, 4),
                "status": "critical" if score_psi >= self.psi_threshold_critical
                          else "warning" if score_psi >= self.psi_threshold_warning
                          else "stable",
                "js_divergence": round(
                    compute_js_divergence(self._reference_scores, fraud_scores_current), 4
                ),
            })

        # Overall status
        if max_psi >= self.psi_threshold_critical:
            overall_status = "critical"
            drift_detected = True
            recommendation = (
                f"CRITICAL: PSI={max_psi:.3f} exceeds threshold {self.psi_threshold_critical}. "
                "Immediate model retraining required. "
                "Notify ML team and CBUAE compliance officer."
            )
        elif max_psi >= self.psi_threshold_warning:
            overall_status = "warning"
            drift_detected = True
            recommendation = (
                f"WARNING: PSI={max_psi:.3f} indicates moderate drift. "
                "Schedule model review within 48 hours. "
                "Increase monitoring frequency."
            )
        else:
            overall_status = "healthy"
            drift_detected = False
            recommendation = (
                f"No significant drift detected (max PSI={max_psi:.4f}). "
                "Model performing within baseline thresholds."
            )

        # Alert if critical
        if drift_detected and overall_status == "critical":
            self._send_alert(max_psi, feature_results)

        # Persist status
        report = ModelDriftResponse(
            status=overall_status,
            psi_score=round(max_psi, 4),
            drift_detected=drift_detected,
            monitored_features=feature_results,
            recommendation=recommendation,
            timestamp=datetime.now(timezone.utc),
        )
        self._save_status(report)
        return report

    def get_current_status(self) -> Dict:
        """Return last computed drift status from disk cache."""
        if self.STATUS_PATH.exists():
            with open(self.STATUS_PATH) as f:
                return json.load(f)
        return {
            "status": "not_initialised",
            "message": "No drift detection run found. Run monitoring pipeline first.",
            "psi_score": None,
            "drift_detected": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_drift_report(self) -> ModelDriftResponse:
        """Return last drift report from disk cache."""
        status = self.get_current_status()
        return ModelDriftResponse(
            status=status.get("status", "unavailable"),
            psi_score=status.get("psi_score", 0.0),
            drift_detected=status.get("drift_detected", False),
            monitored_features=status.get("monitored_features", []),
            recommendation=status.get("recommendation", "No data available."),
        )

    # ──────────────────────────────────────────────────────────
    # Alerting
    # ──────────────────────────────────────────────────────────

    def _send_alert(self, psi: float, features: List[Dict]) -> None:
        """Send Slack/webhook alert on critical drift."""
        webhook_url = settings.monitoring.drift_alert_webhook
        if not webhook_url:
            logger.warning("No drift alert webhook configured.")
            return

        try:
            import urllib.request

            critical_features = [
                f"{f['feature']} (PSI={f['psi']})"
                for f in features
                if f.get("status") == "critical"
            ]

            payload = json.dumps({
                "text": (
                    f"🚨 *FRAUD MODEL DRIFT ALERT* 🚨\n"
                    f"Environment: `{settings.environment}`\n"
                    f"Max PSI: `{psi:.4f}` (threshold: {self.psi_threshold_critical})\n"
                    f"Critical features: {', '.join(critical_features)}\n"
                    f"Action: Model retraining required immediately.\n"
                    f"Timestamp: {datetime.now(timezone.utc).isoformat()}"
                )
            }).encode()

            req = urllib.request.Request(
                webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
            logger.info(f"Drift alert sent to webhook (PSI={psi:.4f})")

        except Exception as e:
            logger.error(f"Failed to send drift alert: {e}")

    # ──────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────

    def _save_status(self, report: ModelDriftResponse) -> None:
        """Cache drift status to JSON for dashboard polling."""
        self.STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(self.STATUS_PATH, "w") as f:
            json.dump(
                {
                    "status": report.status,
                    "psi_score": report.psi_score,
                    "drift_detected": report.drift_detected,
                    "monitored_features": report.monitored_features,
                    "recommendation": report.recommendation,
                    "timestamp": report.timestamp.isoformat(),
                },
                f,
                indent=2,
            )

    def _no_reference_response(self) -> ModelDriftResponse:
        return ModelDriftResponse(
            status="unavailable",
            psi_score=0.0,
            drift_detected=False,
            monitored_features=[],
            recommendation="Reference distribution not set. Run set_reference() after training.",
        )


# ──────────────────────────────────────────────────────────────
# Scheduled monitoring job (run via cron / Cloud Scheduler)
# ──────────────────────────────────────────────────────────────

def run_monitoring_job(
    checkpoint_dir: str = "checkpoints",
    n_recent_samples: int = 5000,
) -> ModelDriftResponse:
    """
    Standalone monitoring job — run hourly via Cloud Scheduler.

    In production:
      1. Load recent predictions from PostgreSQL
      2. Apply preprocessor to get feature vectors
      3. Compute PSI vs training reference
      4. Alert if drift detected
      5. Log to MLflow

    Usage:
        python -c "from src.monitoring.drift_detector import run_monitoring_job; run_monitoring_job()"
    """
    import mlflow

    logger.info("Starting drift monitoring job...")

    detector = DriftDetector()

    if detector._reference_data is None:
        logger.warning("No reference data found. Set reference after training.")
        return detector._no_reference_response()

    # Simulate current production data (replace with DB query)
    rng = np.random.default_rng(int(datetime.now().timestamp()) % 10_000)
    n_feats = max(1, len(detector._reference_data) - 1)

    X_current = rng.normal(0, 1, (n_recent_samples, n_feats)).astype(np.float32)
    scores_current = rng.beta(1, 200, n_recent_samples).astype(np.float32)

    # Add slight drift to simulate production shift
    X_current[:, 0] += rng.normal(0.05, 0.02)  # Small amount distribution shift

    feature_names = [k for k in detector._reference_data if k != "fraud_score"]
    report = detector.detect(X_current, scores_current, feature_names)

    # Log to MLflow
    try:
        mlflow.set_experiment("fraud-detection-monitoring")
        with mlflow.start_run(run_name="drift-monitoring"):
            mlflow.log_metrics({
                "psi_max": report.psi_score,
                "drift_detected": int(report.drift_detected),
            })
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")

    logger.info(
        f"Drift monitoring complete — "
        f"status={report.status} | PSI={report.psi_score:.4f} | "
        f"drift={report.drift_detected}"
    )
    return report