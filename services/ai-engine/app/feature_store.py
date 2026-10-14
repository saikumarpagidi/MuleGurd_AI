"""
services/ai-engine/app/feature_store.py
=========================================
Feature retrieval with a Redis (Feast online store) primary path and an
on-the-fly computation fallback, per the Sprint 4C spec:

    "Load features from Redis (Feast) OR compute on-the-fly from input"

Design principles
------------------
* Never let a Redis outage take the scoring endpoint down. If Redis is
  unreachable, times out, or has a partial/missing key, we fall back to
  computing the same LightGBM features directly from the transaction
  payload. Degraded accuracy (some rolling-window features can't be fully
  reconstructed from a single event) is preferable to a 500 on every request.
* The response always reports which path was used (`feature_source`) so the
  orchestrator and monitoring can track fallback-rate as an SLO signal.
* Redis connection uses short, explicit timeouts (config.py) so a hanging
  Redis does not itself become the latency problem it was meant to prevent.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Optional

try:
    import redis
    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False

from .config import Settings
from .schemas import TransactionInput

log = logging.getLogger("ai-engine.feature_store")


class FeatureStore:
    """Retrieves model features from Redis with on-the-fly fallback."""

    def __init__(self, settings: Settings, feature_order: list[str], fill_missing_with_zero: bool = True):
        self._settings = settings
        self._feature_order = feature_order
        self._fill_missing_with_zero = fill_missing_with_zero
        self._client: Optional[object] = None
        if _HAS_REDIS:
            self._connect()

    def _connect(self) -> None:
        try:
            self._client = redis.Redis(
                host=self._settings.redis_host,
                port=self._settings.redis_port,
                db=self._settings.redis_db,
                password=self._settings.redis_password,
                socket_connect_timeout=self._settings.redis_connect_timeout_seconds,
                socket_timeout=self._settings.redis_socket_timeout_seconds,
                decode_responses=True,
            )
            self._client.ping()
            log.info("Connected to Redis feature store at %s:%s", self._settings.redis_host, self._settings.redis_port)
        except Exception as exc:
            log.warning("Redis unavailable at startup (%s). Will use on-the-fly features until it recovers.", exc)
            self._client = None

    def is_connected(self) -> bool:
        if not _HAS_REDIS or self._client is None:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            return False

    def _redis_key(self, account_id: str) -> str:
        return f"muleguard:features:account:{account_id}"

    def fetch(self, txn: TransactionInput) -> tuple[dict[str, float], str]:
        """Return (features_dict, source) where source is 'redis' or 'on_the_fly'.
        """
        features: dict[str, float] = {}
        source = "on_the_fly"

        # Try Redis primary path
        if self._client is not None and self._settings.feature_ttl_fallback:
            try:
                raw = self._client.hgetall(self._redis_key(txn.from_account))
                if raw:
                    features.update({k: float(v) for k, v in raw.items() if k in self._feature_order})
                    source = "redis"
            except Exception as exc:
                log.warning("Redis read failed for %s, falling back to on-the-fly: %s", txn.from_account, exc)

        # BUG FIX (Bugs 1+2+3): Event-level features MUST always be computed
        # from the live transaction payload and MUST overwrite any stale value
        # that Redis may have stored (e.g. day_of_month from a historical row).
        #
        # Previously used features.setdefault() here, which silently kept the
        # Redis value when a key already existed — causing the model to evaluate
        # a $10 transaction as if it were $5,331, and a Sep-1 transaction as if
        # it happened on the 10th of the month.
        #
        # Using features.update() here ensures event features always win.
        computed = compute_on_the_fly_features(txn, self._feature_order)
        features.update(computed)

        # Ensure every model-required feature has SOME value, unless disabled (for AE)
        if self._fill_missing_with_zero:
            for name in self._feature_order:
                features.setdefault(name, 0.0)

        return features, source


def compute_on_the_fly_features(txn: TransactionInput, feature_order: list[str]) -> dict[str, float]:
    """Best-effort reconstruction of features directly from a single
    transaction event, with no account history available.

    This intentionally cannot reproduce true rolling/aggregate features
    (e.g. sender_receiver_pair_txn_count) — those require Redis/Flink state.
    It fills what IS derivable from a lone event and lets everything else
    fall back to the FeatureStore's 0.0 default, which is a conservative
    (non-alarming) value for count/ratio-style features.

    BUG FIX: Added day_of_month and minute_of_hour extraction from the live
    timestamp. These were previously missing, causing the inference API to
    use stale Redis values for these fields on every transaction scored.
    """
    ts: datetime = txn.timestamp if txn.timestamp.tzinfo else txn.timestamp.replace(tzinfo=timezone.utc)
    hour = ts.hour
    weekday = ts.weekday()
    day = ts.day        # BUG FIX: Extract day-of-month from live timestamp
    minute = ts.minute  # BUG FIX: Extract minute from live timestamp

    derivable = {
        # Amount-derived
        "log_amount": math.log10(max(txn.amount, 1e-9)),
        "log_amount_paid": math.log10(max(txn.amount, 1e-9)),
        "log_amount_received": math.log10(max(txn.amount, 1e-9)),
        "amount_paid": txn.amount,
        "amount_received": txn.amount,
        "is_large_amount": 1.0 if txn.amount > 50_000 else 0.0,
        "round_amount_flag": 1.0 if txn.amount == int(txn.amount) and txn.amount >= 1000 else 0.0,
        "structuring_threshold_flag": 1.0 if 8000 <= txn.amount < 10000 else 0.0,
        # Payment format
        "is_ach_payment": 1.0 if txn.payment_format.value == "ACH" else 0.0,
        "is_cash_payment": 1.0 if txn.payment_format.value == "Cash" else 0.0,
        "is_bitcoin_transaction": 1.0 if txn.payment_format.value == "Bitcoin" else 0.0,
        "is_cross_currency": 0.0,  # unknown without receiver-currency context
        "payment_format_encoded": float({"ACH": 0, "Bitcoin": 1, "Cash": 2, "Cheque": 3,
                                          "Credit Card": 4, "Debit Card": 6, "Reinvestment": 5, 
                                          "Wire": 6, "Other": 6}.get(txn.payment_format.value, 6)),
        "payment_format_id": float({"ACH": 0, "Bitcoin": 1, "Cash": 2, "Cheque": 3,
                                          "Credit Card": 4, "Debit Card": 6, "Reinvestment": 5, 
                                          "Wire": 6, "Other": 6}.get(txn.payment_format.value, 6)),
        # Temporal — always computed from the live timestamp, never from Redis
        "hour_of_day": float(hour),
        "day_of_week": float(weekday),
        "day_of_month": float(day),        # BUG FIX: was missing, defaulted to stale Redis value
        "minute_of_hour": float(minute),   # BUG FIX: was missing, defaulted to stale Redis value
        "is_night": 1.0 if (hour < 6 or hour >= 22) else 0.0,
        "is_night_transaction": 1.0 if (hour < 6 or hour >= 22) else 0.0,
        "is_weekend": 1.0 if weekday >= 5 else 0.0,
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "weekday_sin": math.sin(2 * math.pi * weekday / 7),
        "weekday_cos": math.cos(2 * math.pi * weekday / 7),
        # Self-transfer detection
        "is_self_transfer": 1.0 if txn.from_account == txn.to_account else 0.0,
    }
    return {name: value for name, value in derivable.items() if name in feature_order}
