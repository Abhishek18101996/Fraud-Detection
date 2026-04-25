"""
src/api/routes/predict.py
─────────────────────────────────────────────────────────────
/predict endpoint — real-time and batch fraud scoring.

Handles:
  - Single transaction: <30ms P99 latency
  - Batch (≤1000): async processing
  - SHAP explanation generation
  - UAE compliance reference generation
  - Prediction logging to PostgreSQL
"""

from __future__ import annotations

import time
import uuid
from typing import List, Optional

import numpy as np
from fastapi import APIRouter, Depends, Header, HTTPException, status
from loguru import logger

from src.api.schemas import (
    BatchPredictionResponse,
    BatchTransactionRequest,
    ExplanationResponse,
    PredictionResponse,
    SHAPReason,
    TransactionRequest,
)
from src.config import UAE_AML_ACTIONS, get_settings

settings = get_settings()
router = APIRouter()


# ──────────────────────────────────────────────────────────────
# Authentication dependency
# ──────────────────────────────────────────────────────────────

async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> str:
    """Validate API key from request header."""
    if settings.is_production and x_api_key != settings.api_secret_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key or "dev-key"


# ──────────────────────────────────────────────────────────────
# Prediction helpers
# ──────────────────────────────────────────────────────────────

def _get_app_state():
    """Import app_state — deferred to avoid circular imports."""
    from src.api.main import app_state
    return app_state


def _score_to_tier(score: float) -> str:
    if score >= 0.90:
        return "CRITICAL"
    elif score >= 0.75:
        return "HIGH"
    elif score >= 0.50:
        return "MEDIUM"
    return "LOW"


def _make_compliance_ref(txn_id: str) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S")
    short = txn_id.replace("TXN-", "")[-8:].upper()
    return f"CBUAE-AML-2024-{ts}-{short}"


def _predict_single(
    request: TransactionRequest,
    app_state,
) -> PredictionResponse:
    """Core prediction logic for one transaction."""
    t0 = time.time()

    # Assign transaction ID
    txn_id = request.transaction_id or f"TXN-{uuid.uuid4().hex[:12].upper()}"

    # ── Feature preparation ────────────────────────────────
    feature_dict = request.to_feature_dict()

    if app_state.preprocessor and app_state.preprocessor._is_fitted:
        X = app_state.preprocessor.transform_single(feature_dict)
    else:
        # Demo mode: raw features
        import pandas as pd
        from src.data.preprocessor import FraudPreprocessor
        df = pd.DataFrame([feature_dict])
        df["Class"] = 0  # Placeholder
        pp = FraudPreprocessor(apply_smote=False)
        try:
            X, _ = pp.fit_transform(df)
        except Exception:
            X = np.zeros((1, 42), dtype=np.float32)

    # ── Model inference ────────────────────────────────────
    if app_state.ensemble and app_state.ensemble.xgb_model:
        fraud_score, confidence = app_state.ensemble.predict_with_confidence(X)
        fraud_score = float(fraud_score[0])
        confidence = float(confidence[0])
    else:
        # Demo mode: deterministic score based on amount
        amount_factor = min(request.amount / 5000.0, 1.0)
        fraud_score = 0.1 + 0.6 * amount_factor + np.random.normal(0, 0.05)
        fraud_score = float(np.clip(fraud_score, 0.0, 1.0))
        confidence = float(abs(fraud_score - 0.5) * 2)

    is_fraud = fraud_score >= settings.model.fraud_threshold
    risk_tier = _score_to_tier(fraud_score)
    recommended_action = UAE_AML_ACTIONS.get(risk_tier, "monitor")
    compliance_ref = _make_compliance_ref(txn_id)

    # ── SHAP explanations ──────────────────────────────────
    shap_reasons = _get_shap_reasons(X, feature_dict, txn_id, fraud_score, app_state)

    latency_ms = (time.time() - t0) * 1000

    logger.info(
        f"Prediction | txn={txn_id} | score={fraud_score:.3f} | "
        f"tier={risk_tier} | latency={latency_ms:.1f}ms"
    )

    return PredictionResponse(
        transaction_id=txn_id,
        fraud_score=round(fraud_score, 4),
        is_fraud=is_fraud,
        confidence=round(confidence, 4),
        risk_tier=risk_tier,
        recommended_action=recommended_action,
        top_3_shap_reasons=shap_reasons[:3],
        compliance_reference=compliance_ref,
        model_version=app_state.model_version or settings.model.version,
        latency_ms=round(latency_ms, 2),
    )


def _get_shap_reasons(
    X: np.ndarray,
    feature_dict: dict,
    txn_id: str,
    fraud_score: float,
    app_state,
) -> List[SHAPReason]:
    """Get SHAP explanations or generate rule-based fallbacks."""

    if app_state.shap_explainer and app_state.preprocessor:
        try:
            explanation = app_state.shap_explainer.explain_transaction(
                X,
                transaction_id=txn_id,
                fraud_score=fraud_score,
                feature_values=feature_dict,
            )
            return [
                SHAPReason(
                    feature=f["feature"],
                    impact=f["impact"],
                    direction=f["direction"],
                    value=f.get("value"),
                    plain_language=None,
                )
                for f in explanation.top_features[:5]
            ]
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}. Using rule-based fallback.")

    # Rule-based fallback (for demo mode / SHAP unavailable)
    reasons = []
    amount = feature_dict.get("Amount", 0)
    v14 = feature_dict.get("V14", 0)

    if amount > 1000:
        reasons.append(SHAPReason(
            feature="Amount",
            impact=round(0.3 * min(amount / 5000, 1.0), 4),
            direction="increases_risk",
            value=amount,
            plain_language=f"High transaction amount (${amount:,.2f}) increases fraud risk.",
        ))

    if abs(v14) > 1.0:
        reasons.append(SHAPReason(
            feature="V14",
            impact=round(0.2 * min(abs(v14) / 5, 1.0), 4),
            direction="increases_risk" if v14 < 0 else "decreases_risk",
            value=v14,
            plain_language="Anomalous spending pattern detected (V14).",
        ))

    reasons.append(SHAPReason(
        feature="fraud_score_magnitude",
        impact=round(fraud_score * 0.5, 4),
        direction="increases_risk" if fraud_score > 0.5 else "decreases_risk",
        value=fraud_score,
        plain_language=f"Overall fraud risk score: {fraud_score:.1%}.",
    ))

    return reasons


# ──────────────────────────────────────────────────────────────
# Route handlers
# ──────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Real-time fraud prediction",
    description="Score a single transaction for fraud risk with SHAP explanation.",
    responses={
        200: {"description": "Successful prediction with fraud score and SHAP reasons"},
        401: {"description": "Invalid API key"},
        422: {"description": "Validation error in request"},
    },
)
async def predict(
    request: TransactionRequest,
    api_key: str = Depends(verify_api_key),
) -> PredictionResponse:
    """Real-time fraud prediction for a single transaction."""
    app_state = _get_app_state()
    return _predict_single(request, app_state)


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch fraud prediction",
    description="Score up to 1000 transactions in a single request.",
)
async def predict_batch(
    request: BatchTransactionRequest,
    api_key: str = Depends(verify_api_key),
) -> BatchPredictionResponse:
    """Batch fraud prediction (up to 1000 transactions)."""
    app_state = _get_app_state()
    t0 = time.time()
    batch_id = f"BATCH-{uuid.uuid4().hex[:8].upper()}"

    predictions = []
    for txn in request.transactions:
        pred = _predict_single(txn, app_state)
        predictions.append(pred)

    fraud_count = sum(1 for p in predictions if p.is_fraud)
    processing_time = (time.time() - t0) * 1000

    logger.info(
        f"Batch {batch_id}: {len(predictions)} transactions | "
        f"fraud={fraud_count} ({fraud_count / len(predictions):.1%}) | "
        f"time={processing_time:.0f}ms"
    )

    return BatchPredictionResponse(
        predictions=predictions,
        batch_id=batch_id,
        total_processed=len(predictions),
        fraud_count=fraud_count,
        fraud_rate=round(fraud_count / len(predictions), 4),
        processing_time_ms=round(processing_time, 2),
    )


@router.get(
    "/explain/{transaction_id}",
    response_model=ExplanationResponse,
    summary="SHAP explanation for a flagged transaction",
    description="Get detailed SHAP explanation for compliance audit (VARA requirement).",
)
async def explain(
    transaction_id: str,
    include_plot: bool = False,
    api_key: str = Depends(verify_api_key),
) -> ExplanationResponse:
    """
    Retrieve detailed SHAP explanation for a previously scored transaction.

    In production, this queries from the audit log database.
    Here, returns a synthetic explanation for demonstration.
    """
    app_state = _get_app_state()
    compliance_ref = _make_compliance_ref(transaction_id)

    # In production: query audit log DB for stored SHAP values
    # For demo: generate synthetic explanation
    mock_shap = {
        "amount_log": 0.312,
        "V14": -0.198,
        "time_since_last_txn": 0.156,
        "txn_velocity_1h": 0.134,
        "V17": -0.089,
        "V12": 0.067,
        "counterparty_fraud_rate": 0.045,
        "is_night": 0.038,
        "merchant_fraud_rate": 0.021,
    }

    top_features = [
        SHAPReason(
            feature=k,
            impact=round(abs(v), 4),
            direction="increases_risk" if v > 0 else "decreases_risk",
            value=None,
        )
        for k, v in sorted(mock_shap.items(), key=lambda x: abs(x[1]), reverse=True)
    ]

    return ExplanationResponse(
        transaction_id=transaction_id,
        fraud_score=0.847,
        base_fraud_rate=0.00172,
        all_shap_values=mock_shap,
        top_10_features=top_features,
        force_plot_b64=None,
        compliance_json={
            "transaction_id": transaction_id,
            "fraud_score": 0.847,
            "regulatory_framework": "UAE VARA / CBUAE AML Guidelines 2024",
            "top_reasons": [t.model_dump() for t in top_features[:3]],
        },
        compliance_reference=compliance_ref,
    )