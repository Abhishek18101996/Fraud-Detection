"""
src/config.py
─────────────────────────────────────────────────────────────
Central configuration management using Pydantic Settings.
All environment variables are validated and typed here.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────
# Settings Models
# ──────────────────────────────────────────────────────────────

class DatabaseSettings(BaseSettings):
    url: str = Field("postgresql://fraud_user:fraud_pass@localhost:5432/fraud_db",
                     alias="DATABASE_URL")
    pool_size: int = Field(10, alias="DB_POOL_SIZE")
    max_overflow: int = Field(20, alias="DB_MAX_OVERFLOW")
    echo: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class RedisSettings(BaseSettings):
    url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")
    cache_ttl: int = Field(3600, alias="CACHE_TTL_SECONDS")
    rate_limit_requests: int = Field(100, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(60, alias="RATE_LIMIT_WINDOW_SECONDS")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class MLflowSettings(BaseSettings):
    tracking_uri: str = Field("http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    experiment_name: str = Field("fraud-detection-production",
                                 alias="MLFLOW_EXPERIMENT_NAME")
    artifact_root: str = Field("./mlflow_artifacts", alias="MLFLOW_ARTIFACT_ROOT")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class ModelSettings(BaseSettings):
    version: str = Field("ensemble-v2.1.0", alias="MODEL_VERSION")
    gnn_checkpoint_path: str = Field(
        str(BASE_DIR / "checkpoints" / "graphsage_best.pt"),
        alias="GNN_CHECKPOINT_PATH"
    )
    xgboost_model_path: str = Field(
        str(BASE_DIR / "checkpoints" / "xgboost_ensemble.pkl"),
        alias="XGBOOST_MODEL_PATH"
    )
    fraud_threshold: float = Field(0.5, alias="FRAUD_SCORE_THRESHOLD")
    high_risk_threshold: float = Field(0.75, alias="HIGH_RISK_THRESHOLD")
    critical_risk_threshold: float = Field(0.90, alias="CRITICAL_RISK_THRESHOLD")

    # GNN architecture
    gnn_hidden_channels: int = 128
    gnn_num_layers: int = 3
    gnn_dropout: float = 0.3
    gnn_embedding_dim: int = 128

    # XGBoost hyperparameters (tuned via Optuna)
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_scale_pos_weight: float = 578.0  # Imbalance ratio for ULB dataset

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class ComplianceSettings(BaseSettings):
    jurisdiction: str = Field("UAE", alias="COMPLIANCE_JURISDICTION")
    cbuae_reporting_enabled: bool = Field(True, alias="CBUAE_REPORTING_ENABLED")
    vara_explainability_required: bool = Field(True, alias="VARA_EXPLAINABILITY_REQUIRED")
    audit_log_retention_days: int = Field(2555, alias="AUDIT_LOG_RETENTION_DAYS")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class MonitoringSettings(BaseSettings):
    prometheus_port: int = Field(9090, alias="PROMETHEUS_PORT")
    drift_psi_threshold: float = Field(0.2, alias="DRIFT_PSI_THRESHOLD")
    monitoring_window_hours: int = Field(24, alias="MONITORING_WINDOW_HOURS")
    drift_alert_webhook: Optional[str] = Field(None, alias="DRIFT_ALERT_WEBHOOK")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class Settings(BaseSettings):
    # Application
    app_name: str = Field("fraud-detection-api", alias="APP_NAME")
    app_version: str = Field("2.1.0", alias="APP_VERSION")
    environment: str = Field("development", alias="ENVIRONMENT")
    debug: bool = Field(False, alias="DEBUG")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Security
    api_secret_key: str = Field("change-me-in-production", alias="API_SECRET_KEY")
    allowed_origins: str = Field("http://localhost:3000", alias="ALLOWED_ORIGINS")

    # GCP
    gcp_project_id: str = Field("your-project", alias="GCP_PROJECT_ID")
    gcp_region: str = Field("me-central1", alias="GCP_REGION")
    gcp_bucket_name: str = Field("fraud-detection-models", alias="GCP_BUCKET_NAME")

    # Nested settings
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    compliance: ComplianceSettings = Field(default_factory=ComplianceSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)

    @property
    def allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False
    )


# ──────────────────────────────────────────────────────────────
# Feature Definitions
# ──────────────────────────────────────────────────────────────

TABULAR_FEATURES = [
    # PCA components from ULB dataset (anonymised)
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    # Engineered features
    "Amount",
    "Time",
    "amount_log",
    "amount_zscore",
    "hour_of_day",
    "is_weekend",
    "time_since_last_txn",
    "txn_velocity_1h",
    "txn_velocity_24h",
    "account_age_days",
    "merchant_fraud_rate",
    "counterparty_fraud_rate",
]

GNN_EMBEDDING_FEATURES = [f"gnn_emb_{i}" for i in range(128)]

ENSEMBLE_FEATURES = TABULAR_FEATURES + GNN_EMBEDDING_FEATURES

LABEL_COL = "Class"
TRANSACTION_ID_COL = "transaction_id"


# ──────────────────────────────────────────────────────────────
# Risk Tier Mapping
# ──────────────────────────────────────────────────────────────

RISK_TIERS = {
    "LOW": (0.0, 0.5),
    "MEDIUM": (0.5, 0.75),
    "HIGH": (0.75, 0.90),
    "CRITICAL": (0.90, 1.0),
}

UAE_AML_ACTIONS = {
    "LOW": "monitor",
    "MEDIUM": "enhanced_due_diligence",
    "HIGH": "flag_for_review",
    "CRITICAL": "block_and_report_cbuae",
}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance — call this everywhere."""
    return Settings()