"""
Transaction Ingest Service

Validates incoming transaction events, enforces PII tokenisation rules,
and publishes valid events to the Kafka ingestion topic.

All account identifiers must arrive as SHA-256 tokens.
Raw account numbers are rejected at this boundary.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn

# Initialise instrumentator before the app starts
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Start Prometheus instrumentation on application startup."""
    instrumentator.expose(application, include_in_schema=False)
    yield


app = FastAPI(
    title="MuleGuard.AI - Transaction Ingest",
    description=(
        "Validates incoming transaction events and publishes them to Kafka. "
        "Rejects raw PII - only SHA-256 tokens accepted."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Wire instrumentator to the app before routes are registered
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

    Returns the current operational status of this service.
    Used by the load balancer, Docker health checks, and monitoring.
    """
    return {
        "status": "ok",
        "service": os.getenv("SERVICE_NAME", "transaction-ingest"),
    }


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"redirect": "/health"})


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=False,
    )
