"""
services/graph-service/app/main.py
=====================================
MuleGuard.AI — Sprint 5 (v2 — Real Implementation)

Replaces the mocked stub that returned hardcoded tgn_score=0.85.

Service responsibilities (Sprint 5 deliverable):
  1. Load TGN model at startup (or gracefully enter heuristic mode)
  2. Initialize Neo4j async driver connection pool
  3. Serve POST /graph-score in < 25ms P95
  4. Serve GET /subgraph/{account_id} for investigator dashboard
  5. Expose Prometheus /metrics and Spring-compatible /health

Startup latency: TGN model load takes ~15s for the 1.49 GB checkpoint.
The Dockerfile start-period is set to 60s to accommodate this.
"""

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from .core.config import settings
from .core.neo4j_client import Neo4jClient
from .core.tgn_scorer import TGNScorer
from .api.endpoints.graph_score import router as graph_score_router
from .api.endpoints.subgraph import router as subgraph_router

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("graph-service.main")

_START_TIME = time.time()

# ── Prometheus instrumentation ───────────────────────────────────────────────
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/health"],
)


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Application lifecycle manager.

    Startup:
      1. Initialize Neo4j async driver (connection pool)
      2. Load TGN model checkpoint synchronously (blocking — acceptable at startup)
      3. Expose Prometheus metrics

    Shutdown:
      1. Drain Neo4j connection pool
    """
    log.info("Graph Service starting — loading TGN model and connecting to Neo4j ...")

    # ── Neo4j async driver ───────────────────────────────────────────────────
    neo4j_client = Neo4jClient(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
        max_pool_size=settings.neo4j_max_connection_pool_size,
        query_timeout_s=settings.neo4j_query_timeout_s,
    )
    await neo4j_client.connect()
    application.state.neo4j_client = neo4j_client

    # ── TGN model ────────────────────────────────────────────────────────────
    tgn_scorer = TGNScorer(
        model_path=settings.tgn_model_path,
        num_nodes_expected=settings.tgn_num_nodes_expected,
        memory_dim=settings.tgn_memory_dim,
        edge_dim=settings.tgn_edge_dim,
        time_dim=settings.tgn_time_dim,
    )
    # Model load is synchronous/CPU-bound — runs in the event loop here but
    # is acceptable at startup (FastAPI isn't serving requests during lifespan
    # setup). For faster startup, wrap with asyncio.to_thread() if needed.
    tgn_scorer.load()
    application.state.tgn_scorer = tgn_scorer

    # ── Prometheus ──────────────────────────────────────────────────────────
    instrumentator.expose(application, include_in_schema=False)

    log.info(
        "Graph Service ready | TGN=%s | Neo4j=%s | uptime=%.1fs",
        "loaded" if tgn_scorer.is_loaded else "heuristic_mode",
        "connected" if neo4j_client.is_connected else "unavailable",
        time.time() - _START_TIME,
    )

    yield  # ── serving ───────────────────────────────────────────────────────

    log.info("Graph Service shutting down ...")
    await neo4j_client.close()


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="MuleGuard.AI — Graph Service",
    description=(
        "Temporal Graph Network (TGN) inference and Neo4j subgraph queries. "
        "Detects ring membership, hub accounts, and bridge nodes via GDS algorithms "
        "(PageRank, Louvain, SCC). Sprint 5 deliverable."
    ),
    version="2.0.0",
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

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(graph_score_router)
app.include_router(subgraph_router)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health_check():
    """
    Liveness + readiness probe for Docker Compose and Kubernetes.

    Reports:
      - TGN model load status (loaded vs heuristic_mode)
      - Neo4j connection status
      - Whether the service is fully ready (all dependencies up)

    The orchestrator's Resilience4j circuit breaker monitors this endpoint
    through the /actuator/health path mapped to this service's health check.
    """
    tgn_scorer = getattr(app.state, "tgn_scorer", None)
    neo4j = getattr(app.state, "neo4j_client", None)

    tgn_loaded = tgn_scorer.is_loaded if tgn_scorer else False
    neo4j_connected = neo4j.is_connected if neo4j else False
    ready = neo4j_connected or tgn_loaded  # either path can serve a response

    return {
        "status": "ok" if ready else "degraded",
        "service": os.getenv("SERVICE_NAME", "graph-service"),
        "tgn_model_loaded": tgn_loaded,
        "tgn_mode": "tgn_inference" if tgn_loaded else "heuristic_gds_only",
        "neo4j_connected": neo4j_connected,
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "version": "2.0.0",
    }


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"service": "muleguard-graph-service", "docs": "/docs", "health": "/health"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8003, log_level="info", reload=False)
