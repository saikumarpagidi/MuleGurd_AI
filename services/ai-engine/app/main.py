"""
services/ai-engine/app/main.py
================================
MuleGuard.AI — Sprint 4, Step 4C

Upgrades the ai-engine service from a health-endpoint skeleton to a real
scoring endpoint:

    POST /score
      1. Load features from Redis (Feast) OR compute on-the-fly from input
      2. Run LightGBM inference -> lgbm_score
      3. Run Autoencoder inference -> ae_reconstruction_error -> ae_score
      4. Compute SHAP values for this transaction
      5. Return {lgbm_score, ae_score, shap_top_5, decision_preliminary}

Models are loaded ONCE at process startup (FastAPI lifespan), not per
request, so the <200ms latency target is realistic. /health reports whether
loading actually succeeded, so an orchestrator/k8s readiness probe can keep
a broken pod out of rotation instead of routing traffic to a 500 machine.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .config import settings
from .feature_store import FeatureStore
from .inference import ModelRegistry, score_transaction
from .schemas import HealthStatus, ScoreResponse, TransactionInput

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("ai-engine.main")

_START_TIME = time.time()


class AppState:
    registry: ModelRegistry | None = None
    lgbm_feature_store: FeatureStore | None = None
    ae_feature_store: FeatureStore | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting AI Engine — loading model artifacts ...")
    registry = ModelRegistry(settings)
    try:
        registry.load()
    except (FileNotFoundError, RuntimeError) as exc:
        # Do NOT crash the whole process on a bad model path — this would
        # take the pod into a crash-loop instead of a clean "unready" state
        # that a k8s readiness probe (via /health) can act on sanely.
        log.error("Model loading failed: %s. Service will start but report unready.", exc)
    state.registry = registry
    state.lgbm_feature_store = FeatureStore(settings, registry.lgbm_feature_order or [])
    state.ae_feature_store = FeatureStore(
        settings, 
        registry.ae_feature_order or [], 
        fill_missing_with_zero=False
    )
    yield
    log.info("Shutting down AI Engine")


app = FastAPI(
    title="MuleGuard.AI — AI Engine",
    description="LightGBM + Autoencoder scoring with SHAP explainability",
    version="2.0.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal scoring error. See ai-engine logs for details."},
    )


@app.get("/health", response_model=HealthStatus, tags=["ops"])
async def health() -> HealthStatus:
    registry = state.registry
    redis_ok = bool(state.lgbm_feature_store and state.lgbm_feature_store.is_connected())
    return HealthStatus(
        status="ok" if (registry and registry.is_ready) else "degraded",
        lgbm_model_loaded=bool(registry and registry.lgbm_model is not None),
        autoencoder_model_loaded=bool(registry and registry.ae_model is not None),
        shap_explainer_ready=bool(registry and registry.shap_explainer is not None),
        redis_connected=redis_ok,
        uptime_seconds=round(time.time() - _START_TIME, 1),
    )


@app.get("/metrics", tags=["ops"])
async def metrics() -> dict:
    """Minimal placeholder for Prometheus. Sprint 2 already wired the
    Grafana/Prometheus scrape config for this service; a real deployment
    should replace this with prometheus_client's generate_latest()."""
    registry = state.registry
    ready = 1 if (registry and registry.is_ready) else 0
    return {
        "ai_engine_ready": ready,
        "note": "Wire prometheus_client generate_latest() here for real metrics.",
    }


@app.post("/score", response_model=ScoreResponse, tags=["scoring"])
async def score(transaction: TransactionInput) -> ScoreResponse:
    registry = state.registry
    if registry is None or not registry.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models are not loaded. Check /health.",
        )

    lgbm_features, lgbm_source = state.lgbm_feature_store.fetch(transaction)
    ae_features, ae_source = state.ae_feature_store.fetch(transaction)
    # Report the "weaker" of the two sources so the caller isn't told
    # "redis" when one of the two models actually fell back on-the-fly.
    feature_source = "on_the_fly" if "on_the_fly" in (lgbm_source, ae_source) else lgbm_source

    try:
        result = score_transaction(
            registry=registry,
            settings=settings,
            transaction_id=transaction.transaction_id,
            lgbm_features=lgbm_features,
            ae_features=ae_features,
            feature_source=feature_source,
        )
    except Exception as exc:
        log.exception("Scoring failed for transaction_id=%s", transaction.transaction_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scoring failed: {exc}",
        ) from exc

    return result


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"service": "muleguard-ai-engine", "docs": "/docs", "health": "/health"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8002, reload=False)
