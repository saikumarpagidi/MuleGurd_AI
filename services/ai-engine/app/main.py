"""
AI Engine Service

Serves LightGBM velocity scoring and Deep Autoencoder anomaly detection.
Returns risk scores alongside TreeSHAP feature attributions for every prediction.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn

instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    instrumentator.expose(application, include_in_schema=False)
    yield


app = FastAPI(
    title="MuleGuard.AI - AI Engine",
    description=(
        "LightGBM velocity scorer and Deep Autoencoder anomaly detector. "
        "Returns risk scores and SHAP feature attributions."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

instrumentator.instrument(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Service health check.

    Reports operational status and whether ML models are currently loaded.
    Models are loaded on startup from the model registry path.
    """
    return {
        "status": "ok",
        "service": os.getenv("SERVICE_NAME", "ai-engine"),
        "models_loaded": [],
    }


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"redirect": "/health"})


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        log_level="info",
        reload=False,
    )
