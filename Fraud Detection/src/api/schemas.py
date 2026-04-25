"""
src/api/schemas.py
─────────────────────────────────────────────────────────────
Pydantic v2 request/response models for the fraud detection API.

All schemas include UAE compliance fields and strict validation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ──────────────────────────────────────────────────────────────
# Request schemas
# ──────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    """
    Single transaction prediction request.

    For ULB dataset compatibility, V1–V28 are the PCA components.
    In production, replace with actual transaction attributes.
    """
    # Identifiers
    transaction_id: Optional[str] = Field(
        None,
        description="Unique transaction identifier. Auto-generated if not provided.",
        examples=["TXN-2024-0001234"]
    )

    # Core transaction fields
    amount: float = Field(
        ...,
        gt=0,
        le=1_000_000,
        description="Transaction amount in USD",
        examples=[4850.00]
    )
    time: float = Field(
        default=0.0,
        ge=0,
        description="Seconds since first transaction in session",
        examples=[86400.0]
    )

    # PCA anonymised features (ULB dataset)
    V1: float = Field(default=0.0)
    V2: float = Field(default=0.0)
    V3: float = Field(default=0.0)
    V4: float = Field(default=0.0)
    V5: float = Field(default=0.0)
    V6: float = Field(default=0.0)
    V7: float = Field(default=0.0)
    V8: float = Field(default=0.0)
    V9: float = Field(default=0.0)
    V10: float = Field(default=0.0)
    V11: float = Field(default=0.0)
    V12: float = Field(default=0.0)
    V13: float = Field(default=0.0)
    V14: float = Field(default=0.0)
    V15: float = Field(default=0.0)
    V16: float = Field(default=0.0)
    V17: float = Field(default=0.0)
    V18: float = Field(default=0.0)
    V19: float = Field(default=0.0)
    V20: float = Field(default=0.0)
    V21: float = Field(default=0.0)
    V22: float = Field(default=0.0)
    V23: float = Field(default=0.0)
    V24: float = Field(default=0.0)
    V25: float = Field(default=0.0)
    V26: float = Field(default=0.0)
    V27: float = Field(default=0.0)
    V28: float = Field(default=0.0)

    # Enrichment fields (from real-time feature store)
    account_age_days: Optional[float] = Field(None, ge=0, description="Age of account in days")
    merchant_fraud_rate: Optional[float] = Field(None, ge=0, le=1, description="Historical merchant fraud rate")
    counterparty_fraud_rate: Optional[float] = Field(None, ge=0, le=1, description="Historical counterparty fraud rate")
    merchant_category: Optional[str] = Field(None, description="MCC category", examples=["electronics", "travel"])
    is_international: Optional[bool] = Field(None, description="Cross-border transaction flag")

    # Metadata
    channel: Optional[str] = Field(None, description="Payment channel", examples=["mobile", "web", "pos"])
    currency: str = Field(default="USD", description="Transaction currency (ISO 4217)")

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Amount must be positive")
        return round(v, 2)

    def to_feature_dict(self) -> Dict:
        """Convert request to flat feature dictionary for preprocessor."""
        return {
            "Amount": self.amount,
            "Time": self.time,
            **{f"V{i}": getattr(self, f"V{i}") for i in range(1, 29)},
            "account_age_days": self.account_age_days or 365.0,
            "merchant_fraud_rate": self.merchant_fraud_rate or 0.0,
            "counterparty_fraud_rate": self.counterparty_fraud_rate or 0.0,
        }

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN-AE-20240115-001234",
                "amount": 4850.00,
                "time": 86400.0,
                "V1": -1.3598, "V2": -0.0728, "V3": 2.5363,
                "V4": 1.3782, "V14": -0.3111, "V17": 0.3925,
                "merchant_category": "electronics",
                "merchant_fraud_rate": 0.02,
                "counterparty_fraud_rate": 0.005,
                "channel": "web",
                "currency": "AED",
            }
        }


class BatchTransactionRequest(BaseModel):
    """Batch prediction request (max 1000 transactions)."""
    transactions: List[TransactionRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of transactions to score"
    )
    return_explanations: bool = Field(
        False,
        description="Include SHAP explanations in response (slower)"
    )
    priority: Literal["standard", "high"] = Field(
        "standard",
        description="Processing priority"
    )


# ──────────────────────────────────────────────────────────────
# Response schemas
# ──────────────────────────────────────────────────────────────

class SHAPReason(BaseModel):
    """Single SHAP feature explanation."""
    feature: str = Field(..., description="Feature name")
    impact: float = Field(..., description="Absolute SHAP value (contribution magnitude)")
    direction: Literal["increases_risk", "decreases_risk"] = Field(
        ..., description="Whether feature increases or decreases fraud risk"
    )
    value: Optional[float] = Field(None, description="Observed feature value")
    plain_language: Optional[str] = Field(None, description="Human-readable explanation")


class PredictionResponse(BaseModel):
    """
    Real-time fraud prediction response.

    Compliant with UAE VARA Regulation Article 4.3:
    All automated decisions must include traceable justification.
    """
    transaction_id: str
    fraud_score: float = Field(..., ge=0.0, le=1.0, description="Fraud probability [0,1]")
    is_fraud: bool = Field(..., description="Binary fraud classification")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Decision confidence [0,1]")
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    recommended_action: str = Field(..., description="UAE AML recommended action")

    # SHAP explanations (required by UAE VARA)
    top_3_shap_reasons: List[SHAPReason] = Field(
        ..., description="Top 3 SHAP feature contributions"
    )

    # Compliance
    compliance_reference: str = Field(..., description="CBUAE compliance audit reference")
    model_version: str
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN-AE-20240115-001234",
                "fraud_score": 0.847,
                "is_fraud": True,
                "confidence": 0.921,
                "risk_tier": "HIGH",
                "recommended_action": "flag_for_review",
                "top_3_shap_reasons": [
                    {"feature": "amount_log", "impact": 0.312, "direction": "increases_risk", "value": 8.487},
                    {"feature": "V14", "impact": 0.198, "direction": "increases_risk", "value": -0.311},
                    {"feature": "time_since_last_txn", "impact": 0.156, "direction": "increases_risk", "value": 45.0},
                ],
                "compliance_reference": "CBUAE-AML-2024-20240115-001234",
                "model_version": "ensemble-v2.1.0",
                "latency_ms": 23.4,
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction requests."""
    predictions: List[PredictionResponse]
    batch_id: str
    total_processed: int
    fraud_count: int
    fraud_rate: float
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExplanationResponse(BaseModel):
    """Detailed SHAP explanation for compliance audit."""
    transaction_id: str
    fraud_score: float
    base_fraud_rate: float = Field(..., description="Model's baseline fraud rate (E[f(X)])")
    all_shap_values: Dict[str, float] = Field(..., description="Full feature → SHAP value map")
    top_10_features: List[SHAPReason]
    force_plot_b64: Optional[str] = Field(None, description="Base64-encoded PNG force plot")
    compliance_json: Dict = Field(..., description="Full FATF-ready compliance report")
    compliance_reference: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────────────────────
# Dashboard schemas
# ──────────────────────────────────────────────────────────────

class FraudRatePoint(BaseModel):
    """Single data point for fraud rate time series."""
    timestamp: str
    hour: int
    fraud_rate: float
    total_transactions: int
    fraud_count: int
    avg_fraud_score: float


class FraudRateResponse(BaseModel):
    """Hourly fraud rate for Power BI dashboard."""
    data: List[FraudRatePoint]
    period_hours: int
    overall_fraud_rate: float
    total_transactions: int


class MerchantRiskItem(BaseModel):
    """Merchant category risk ranking."""
    merchant_category: str
    transaction_count: int
    fraud_count: int
    fraud_rate: float
    avg_fraud_score: float
    risk_tier: str


class SHAPDistributionItem(BaseModel):
    """SHAP reason frequency for compliance dashboard."""
    feature: str
    count: int
    avg_impact: float
    pct_increases_risk: float


class ModelDriftResponse(BaseModel):
    """Model drift monitoring data."""
    status: Literal["healthy", "warning", "critical"]
    psi_score: float = Field(..., description="Population Stability Index")
    drift_detected: bool
    monitored_features: List[Dict]
    recommendation: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ──────────────────────────────────────────────────────────────
# Health schemas
# ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    model_loaded: bool
    gnn_enabled: bool
    uptime_seconds: float
    environment: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)