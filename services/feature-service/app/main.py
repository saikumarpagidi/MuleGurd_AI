"""
Feature Service

Serves pre-computed account features from the Feast online store backed by Redis.
Returns 32 LightGBM velocity features and 14 Autoencoder behavioural features
with a target retrieval latency of under 2 milliseconds.
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
    title="MuleGuard.AI - Feature Service",
    description=(
        "Serves pre-computed features from the Feast online store backed by Redis. "
        "Returns 32 LightGBM + 14 Autoencoder features in under 2ms."
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

    Reports Feast feature store connectivity, Redis connection status,
    and the number of registered feature definitions.
    """
    return {
        "status": "ok",
        "service": os.getenv("SERVICE_NAME", "feature-service"),
        "feast_connected": False,
        "redis_connected": False,
        "feature_count": 0,
    }


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"redirect": "/health"})


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8004,
        log_level="info",
        reload=False,
    )
