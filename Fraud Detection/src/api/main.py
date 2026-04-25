"""
src/api/main.py
─────────────────────────────────────────────────────────────
FastAPI application entry point.

Features:
  - Async model loading at startup
  - Prometheus metrics middleware
  - CORS for Power BI DirectQuery
  - API key authentication
  - Structured logging with request tracing
  - GCP Cloud Run health check compatibility
"""

from __future__ import annotations

import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.api.routes.dashboard import router as dashboard_router
from src.api.routes.health import router as health_router
from src.api.routes.predict import router as predict_router
from src.config import get_settings

settings = get_settings()

# ──────────────────────────────────────────────────────────────
# Prometheus metrics
# ──────────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "fraud_api_request_latency_seconds",
    "API request latency",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
FRAUD_PREDICTIONS = Counter(
    "fraud_predictions_total",
    "Total fraud predictions",
    ["risk_tier", "is_fraud"],
)
MODEL_LOAD_STATUS = Gauge("fraud_model_loaded", "Whether the ML model is loaded")
ACTIVE_REQUESTS = Gauge("fraud_api_active_requests", "Number of active requests")


# ──────────────────────────────────────────────────────────────
# Application state (singleton model cache)
# ──────────────────────────────────────────────────────────────

class AppState:
    """Shared application state — model instances cached here."""
    ensemble = None
    preprocessor = None
    shap_explainer = None
    startup_time: float = 0.0
    model_version: str = "unknown"

app_state = AppState()


# ──────────────────────────────────────────────────────────────
# Lifespan: load models at startup
# ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models during application startup."""
    logger.info("🚀 Starting Fraud Detection API...")
    t0 = time.time()
    app_state.startup_time = t0

    checkpoint_dir = Path("checkpoints")

    try:
        # Load preprocessor
        preprocessor_path = checkpoint_dir / "preprocessor.pkl"
        if preprocessor_path.exists():
            with open(preprocessor_path, "rb") as f:
                app_state.preprocessor = pickle.load(f)
            logger.info("✓ Preprocessor loaded")
        else:
            logger.warning("Preprocessor not found — using untrained preprocessor")
            from src.data.preprocessor import FraudPreprocessor
            app_state.preprocessor = FraudPreprocessor(apply_smote=False)

        # Load ensemble model
        from src.models.ensemble import FraudEnsemble
        if checkpoint_dir.exists() and (checkpoint_dir / "xgboost_ensemble.pkl").exists():
            app_state.ensemble = FraudEnsemble.load(
                checkpoint_dir,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            app_state.model_version = settings.model.version
            MODEL_LOAD_STATUS.set(1)
            logger.info(f"✓ Ensemble model loaded (version={app_state.model_version})")
        else:
            logger.warning("No trained model found. Running in demo mode with random scores.")
            MODEL_LOAD_STATUS.set(0)

        # Load SHAP explainer
        shap_path = checkpoint_dir / "shap_explainer.pkl"
        if shap_path.exists():
            with open(shap_path, "rb") as f:
                app_state.shap_explainer = pickle.load(f)
            logger.info("✓ SHAP explainer loaded")

        elapsed = time.time() - t0
        logger.info(f"✅ API startup complete in {elapsed:.2f}s")

    except Exception as e:
        logger.error(f"Startup error: {e}")
        MODEL_LOAD_STATUS.set(0)

    yield  # App runs here

    # Shutdown cleanup
    logger.info("Shutting down Fraud Detection API...")
    app_state.ensemble = None
    app_state.preprocessor = None
    app_state.shap_explainer = None


# ──────────────────────────────────────────────────────────────
# FastAPI application
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="🔍 Fraud Detection API",
    description="""
## Real-Time Banking Fraud Detection
**UAE VARA / CBUAE AML Compliant**

Powered by:
- **GraphSAGE** (PyTorch Geometric) — transaction graph analysis
- **XGBoost** ensemble — tabular + GNN feature fusion
- **SHAP** — per-prediction explainability (VARA requirement)

### Authentication
Include `X-API-Key: <your-key>` in all requests.

### Rate Limits
- Standard: 100 requests/minute
- Batch: 10 requests/minute (up to 1000 transactions each)
    """,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────
# Middleware
# ──────────────────────────────────────────────────────────────

# CORS — allow Power BI DirectQuery
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Compression for large batch responses
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Prometheus metrics collection middleware."""
    endpoint = request.url.path
    ACTIVE_REQUESTS.inc()
    t0 = time.time()

    try:
        response = await call_next(request)
        latency = time.time() - t0

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

        # Add latency header
        response.headers["X-Response-Time-Ms"] = f"{latency * 1000:.1f}"
        return response
    finally:
        ACTIVE_REQUESTS.dec()


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Add request tracing ID to all responses."""
    import uuid
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ──────────────────────────────────────────────────────────────
# Routers
# ──────────────────────────────────────────────────────────────

app.include_router(
    predict_router,
    prefix="/api/v1",
    tags=["Fraud Detection"],
)
app.include_router(
    health_router,
    prefix="/api/v1",
    tags=["Health & Monitoring"],
)
app.include_router(
    dashboard_router,
    prefix="/api/v1/dashboard",
    tags=["Power BI Dashboard"],
)


# ──────────────────────────────────────────────────────────────
# Special endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """Prometheus scrape endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Fraud Detection API",
        "version": settings.app_version,
        "status": "operational",
        "docs": "/docs",
        "compliance": "UAE VARA / CBUAE AML 2024",
    }


# ──────────────────────────────────────────────────────────────
# Exception handlers
# ──────────────────────────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "type": "validation_error"},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error. Please contact support.",
            "type": "internal_error",
        },
    )


# ──────────────────────────────────────────────────────────────
# Dev server entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        workers=1 if settings.debug else 4,
        log_level=settings.log_level.lower(),
    )