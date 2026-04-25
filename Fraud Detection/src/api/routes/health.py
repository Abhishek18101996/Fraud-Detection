"""
src/api/routes/health.py
─────────────────────────────────────────────────────────────
Health check, readiness, and liveness endpoints.

Compatible with:
  - GCP Cloud Run health probes
  - Kubernetes liveness/readiness probes
  - Power BI gateway connectivity checks
  - Prometheus scrape target verification
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict

import torch
from fastapi import APIRouter
from loguru import logger

from src.api.schemas import HealthResponse
from src.config import get_settings

settings = get_settings()
router = APIRouter()

# Track server startup time
_SERVER_START_TIME = time.time()


def _get_app_state():
    """Deferred import to avoid circular imports."""
    from src.api.main import app_state
    return app_state


# ──────────────────────────────────────────────────────────────
# Health endpoints
# ──────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns overall service health status. Used by GCP Cloud Run.",
    tags=["Health & Monitoring"],
)
async def health_check() -> HealthResponse:
    """
    Basic health check — returns 200 if service is running.
    GCP Cloud Run uses this as the liveness probe.
    """
    app_state = _get_app_state()
    uptime = time.time() - _SERVER_START_TIME
    model_loaded = app_state.ensemble is not None and app_state.ensemble.xgb_model is not None

    status = "healthy" if model_loaded else "degraded"

    return HealthResponse(
        status=status,
        version=settings.app_version,
        model_loaded=model_loaded,
        gnn_enabled=(
            app_state.ensemble is not None
            and app_state.ensemble.use_gnn
            and app_state.ensemble.gnn is not None
        ),
        uptime_seconds=round(uptime, 2),
        environment=settings.environment,
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/health/ready",
    summary="Readiness probe",
    description="Returns 200 only when model is fully loaded and ready to serve.",
    tags=["Health & Monitoring"],
)
async def readiness_check() -> Dict:
    """
    Readiness probe — returns 503 if model not loaded.
    Kubernetes / Cloud Run will not route traffic until this passes.
    """
    app_state = _get_app_state()
    model_ready = (
        app_state.ensemble is not None
        and app_state.ensemble.xgb_model is not None
        and app_state.preprocessor is not None
    )

    if not model_ready:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "message": "Model not yet loaded. Retry in a few seconds.",
                "model_loaded": False,
            },
        )

    return {
        "status": "ready",
        "model_version": app_state.model_version,
        "gnn_enabled": app_state.ensemble.use_gnn,
        "shap_ready": app_state.shap_explainer is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/health/live",
    summary="Liveness probe",
    description="Simple ping — returns 200 if process is alive.",
    tags=["Health & Monitoring"],
)
async def liveness_check() -> Dict:
    """
    Liveness probe — always returns 200 if process is running.
    If this fails, Cloud Run restarts the container.
    """
    return {
        "status": "alive",
        "uptime_seconds": round(time.time() - _SERVER_START_TIME, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/health/model",
    summary="Detailed model diagnostics",
    description="Returns model metadata, version info, and component status.",
    tags=["Health & Monitoring"],
)
async def model_diagnostics() -> Dict:
    """Detailed model health for engineering/ops dashboards."""
    app_state = _get_app_state()

    diagnostics = {
        "model_version": app_state.model_version or "not_loaded",
        "environment": settings.environment,
        "components": {
            "preprocessor": {
                "loaded": app_state.preprocessor is not None,
                "fitted": (
                    app_state.preprocessor is not None
                    and app_state.preprocessor._is_fitted
                ),
                "n_features": (
                    len(app_state.preprocessor.get_feature_names())
                    if app_state.preprocessor and app_state.preprocessor._is_fitted
                    else 0
                ),
            },
            "xgboost": {
                "loaded": (
                    app_state.ensemble is not None
                    and app_state.ensemble.xgb_model is not None
                ),
                "threshold": (
                    app_state.ensemble.xgb_model.threshold_
                    if app_state.ensemble and app_state.ensemble.xgb_model
                    else None
                ),
            },
            "gnn": {
                "loaded": (
                    app_state.ensemble is not None
                    and app_state.ensemble.gnn is not None
                ),
                "enabled": (
                    app_state.ensemble is not None
                    and app_state.ensemble.use_gnn
                ),
                "device": (
                    str(app_state.ensemble.device)
                    if app_state.ensemble
                    else "N/A"
                ),
                "cuda_available": torch.cuda.is_available(),
            },
            "shap_explainer": {
                "loaded": app_state.shap_explainer is not None,
                "n_background_samples": (
                    len(app_state.shap_explainer._background_data)
                    if app_state.shap_explainer and app_state.shap_explainer._background_data is not None
                    else 0
                ),
            },
        },
        "runtime": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
            "uptime_seconds": round(time.time() - _SERVER_START_TIME, 2),
            "pid": os.getpid(),
        },
        "compliance": {
            "jurisdiction": settings.compliance.jurisdiction,
            "cbuae_reporting": settings.compliance.cbuae_reporting_enabled,
            "vara_explainability": settings.compliance.vara_explainability_required,
            "audit_retention_days": settings.compliance.audit_log_retention_days,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return diagnostics


@router.get(
    "/health/drift",
    summary="Model drift status",
    description="Returns current drift detection status from the monitoring module.",
    tags=["Health & Monitoring"],
)
async def drift_status() -> Dict:
    """
    Returns current model drift status.
    Integrates with the DriftDetector monitoring module.
    """
    try:
        from src.monitoring.drift_detector import DriftDetector
        detector = DriftDetector()
        status = detector.get_current_status()
        return status
    except Exception as e:
        logger.warning(f"Drift detector unavailable: {e}")
        return {
            "status": "unavailable",
            "message": "Drift detector not initialised. Run monitoring pipeline first.",
            "psi_score": None,
            "drift_detected": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }