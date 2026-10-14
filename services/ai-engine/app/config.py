"""
services/ai-engine/app/config.py
==================================
Centralized, environment-driven configuration for the AI Engine service.

Nothing here is hardcoded to a path or host that would break in a different
deployment (local, docker-compose, k8s) — everything is overridable via env
vars with sane local defaults, per 12-factor practice.

Env var prefix: AI_ENGINE_  (e.g. AI_ENGINE_LGBM_MODEL_PATH)
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AI_ENGINE_", extra="ignore")

    # ── Model artifact paths ─────────────────────────────────────────────
    # Defaults point to actual project-root-relative artifacts.
    # In Docker, override via env vars to /app/ml/models/...
    lgbm_model_path: str = "ml/models/LightGBM/lgbm_best_final.pkl"
    autoencoder_model_path: str = "ml/models/Autoencoder/autoencoder_final.pt"
    autoencoder_scaler_path: str = "ml/models/Autoencoder/scaler_stats.json"
    autoencoder_threshold_path: str = "ml/models/Autoencoder/threshold_config.json"
    autoencoder_feature_norm_path: str = "ml/models/Autoencoder/feature_error_norm.json"

    # ── Feature store (Redis / Feast online store) ───────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None
    redis_connect_timeout_seconds: float = 0.5
    redis_socket_timeout_seconds: float = 0.2
    feature_ttl_fallback: bool = True  # if Redis is down/miss, compute on-the-fly

    # ── Decision thresholds ──────────────────────────────────────────────
    # Must match Sprint 6's fusion decision engine.
    review_threshold: float = 0.45
    block_threshold: float = 0.75

    # ── SHAP ─────────────────────────────────────────────────────────────
    shap_top_k: int = 10

    # ── Service ──────────────────────────────────────────────────────────
    max_inference_latency_ms: float = 200.0  # SLA target from the plan
    log_level: str = "INFO"


settings = Settings()
