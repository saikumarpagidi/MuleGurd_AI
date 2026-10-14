"""
services/ai-engine/app/schemas.py
==================================
Request/response contracts for the AI Engine scoring API.

Kept separate from main.py so the wire contract can be versioned and reused
by other services (scoring-orchestrator's Java client, tests, docs) without
importing FastAPI internals.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PaymentFormat(str, Enum):
    """Payment channel identifiers matching the IBM AML dataset."""
    ACH = "ACH"
    WIRE = "Wire"
    CASH = "Cash"
    CHEQUE = "Cheque"
    CREDIT_CARD = "Credit Card"
    DEBIT_CARD = "Debit Card"
    BITCOIN = "Bitcoin"
    REINVESTMENT = "Reinvestment"
    OTHER = "Other"


class ShapDirection(str, Enum):
    RAISES_RISK = "RAISES_RISK"
    LOWERS_RISK = "LOWERS_RISK"


class AeAnomalyDirection(str, Enum):
    """Severity of a feature's contribution to the Autoencoder anomaly score.

    Thresholds are calibrated against the training-set feature error distribution:
      - NO_ANOMALY_CONTRIBUTION  : z-score ≤ 0.05  (normal, within expected noise)
      - SMALL_ANOMALY_CONTRIBUTION: 0.05 < z-score < 1.0 (mildly unusual)
      - RAISES_ANOMALY           : z-score ≥ 1.0  (strongly anomalous)
    """
    NO_ANOMALY_CONTRIBUTION = "NO_ANOMALY_CONTRIBUTION"
    SMALL_ANOMALY_CONTRIBUTION = "SMALL_ANOMALY_CONTRIBUTION"
    RAISES_ANOMALY = "RAISES_ANOMALY"


class DecisionPreliminary(str, Enum):
    """Preliminary decision from the AI Engine alone (LightGBM + AE only).
    The authoritative BLOCK/REVIEW/APPROVE decision is issued by the
    scoring-orchestrator after fusing TGN scores."""
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class FeatureSource(str, Enum):
    REDIS = "redis"
    ON_THE_FLY = "on_the_fly"


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class TransactionInput(BaseModel):
    """Raw transaction event as published on muleguard.transactions.

    Only `transaction_id`, `from_account`, `to_account`, `amount`, and
    `timestamp` are strictly required — everything else is optional and
    falls back to Redis feature lookup or a safe default so the endpoint
    degrades gracefully instead of hard-failing on partial payloads.
    """

    transaction_id: str = Field(..., min_length=1, max_length=128)
    from_account: str = Field(..., min_length=1, max_length=128)
    to_account: str = Field(..., min_length=1, max_length=128)
    amount: float = Field(..., gt=0)
    timestamp: datetime
    payment_format: PaymentFormat = PaymentFormat.OTHER
    currency: str = Field(default="USD", max_length=32)

    @field_validator("amount")
    @classmethod
    def amount_must_be_finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):  # NaN / inf guard
            raise ValueError("amount must be a finite positive number")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transaction_id": "TXN-20260824-000123",
                "from_account": "ACC-88213",
                "to_account": "ACC-44107",
                "amount": 9500.00,
                "timestamp": "2026-08-24T09:15:00Z",
                "payment_format": "ACH",
                "currency": "USD",
            }
        }
    )


# ---------------------------------------------------------------------------
# Response — SHAP feature attribution (LightGBM)
# ---------------------------------------------------------------------------

class ShapFeatureContribution(BaseModel):
    feature: str
    value: float = Field(..., description="The observed value of the feature for this transaction")
    shap_contribution: float = Field(..., description="Signed SHAP value (logit-space push toward/away from fraud)")
    direction: ShapDirection


# ---------------------------------------------------------------------------
# Response — Autoencoder feature anomaly explanation
# ---------------------------------------------------------------------------

class AeFeatureContribution(BaseModel):
    """Per-feature reconstruction anomaly from the Autoencoder.

    Unlike SHAP, there is no negative direction — the Autoencoder only
    measures HOW ANOMALOUS a feature is, not whether it helps or hurts.
    The `reconstruction_z_score` is the feature-level squared error,
    normalized by the training-set mean/std for that feature.
    """
    feature: str
    value: float = Field(..., description="The observed (pre-standardization) value of this feature")
    reconstruction_z_score: float = Field(
        ...,
        ge=0,
        description=(
            "Feature-level reconstruction error Z-score. "
            "0 = perfectly reconstructed (normal). "
            "≥1.0 = strongly anomalous compared to training distribution."
        ),
    )
    direction: AeAnomalyDirection


# ---------------------------------------------------------------------------
# Response — full scoring result
# ---------------------------------------------------------------------------

class ScoreResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    transaction_id: str
    lgbm_score: float = Field(..., ge=0, le=1)
    ae_score: float = Field(..., ge=0, le=1, description="Normalized autoencoder anomaly score (0=normal, 1=max anomaly)")
    ae_reconstruction_error: float = Field(..., ge=0, description="Raw mean squared reconstruction error from the Autoencoder")
    shap_top_5: list[ShapFeatureContribution] = Field(
        ...,
        description="Top-K LightGBM SHAP feature attributions, sorted by absolute contribution magnitude",
    )
    ae_top_5: list[AeFeatureContribution] = Field(
        default_factory=list,
        description=(
            "Top-K Autoencoder anomaly features, sorted by reconstruction Z-score. "
            "Empty list when ae_score=0 (cold-start or normal transaction)."
        ),
    )
    decision_preliminary: DecisionPreliminary
    feature_source: FeatureSource
    model_versions: dict[str, str]
    inference_latency_ms: float


# ---------------------------------------------------------------------------
# Health response
# ---------------------------------------------------------------------------

class HealthStatus(BaseModel):
    status: str
    lgbm_model_loaded: bool
    autoencoder_model_loaded: bool
    shap_explainer_ready: bool
    redis_connected: bool
    uptime_seconds: float
