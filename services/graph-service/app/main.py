"""
Graph Service

Serves Temporal Graph Network (TGN) inference and Neo4j subgraph queries.
Detects ring membership, hub accounts, and bridge nodes using GDS algorithms:
PageRank, Louvain community detection, and Strongly Connected Components (SCC).
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
    title="MuleGuard.AI - Graph Service",
    description=(
        "Temporal Graph Network (TGN) inference and Neo4j subgraph queries. "
        "Detects ring membership, hub accounts, and bridge nodes via GDS algorithms."
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

    Reports graph database connectivity and TGN model load status.
    """
    return {
        "status": "ok",
        "service": os.getenv("SERVICE_NAME", "graph-service"),
        "neo4j_connected": False,
        "tgn_model_loaded": False,
    }


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"redirect": "/health"})


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        log_level="info",
        reload=False,
    )
