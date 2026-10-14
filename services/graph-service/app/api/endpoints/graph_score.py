"""
services/graph-service/app/api/endpoints/graph_score.py
=========================================================
MuleGuard.AI — Sprint 5

POST /graph-score — primary scoring endpoint consumed by the Sprint 6
scoring-orchestrator's GraphServiceClient.

Pipeline per request (must complete within 25ms P95):
  1. [~1ms]  Fetch GDS signals from Neo4j (pagerank, community_id, in_scc_ring)
  2. [~20ms] TGN inference via tgn_scorer (or heuristic if model not loaded)
  3. [~1ms]  Ring detection via ring_detector
  4. [<1ms]  Build and return response

Latency guard: if the total wall-clock exceeds settings.latency_budget_ms,
we skip the Neo4j 2-hop subgraph fetch (the most expensive optional step)
and return with GDS-only enrichment. The orchestrator treats the result as
valid either way — degraded_mode=True just informs the fusion engine to
down-weight the TGN contribution slightly.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional, Annotated

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(tags=["Scoring"])


# ──────────────────────────────────────────────────────────────────────── #
# Request / Response schemas (wire-contract for the orchestrator client)
# ──────────────────────────────────────────────────────────────────────── #


class GraphScoreRequest(BaseModel):
    from_account: str = Field(..., description="Sender account ID (SHA-256 token)")
    to_account: str = Field(..., description="Receiver account ID (SHA-256 token)")
    amount: float = Field(..., gt=0, description="Transaction amount in original currency")
    timestamp: datetime = Field(..., description="Transaction event time (ISO 8601)")
    payment_format: str = Field(default="OTHER", description="Payment channel")
    currency: str = Field(default="USD", max_length=50)

    model_config = {
        "json_schema_extra": {
            "example": {
                "from_account": "a4b3c2d1e0f9...",
                "to_account": "f9e8d7c6b5a4...",
                "amount": 9500.00,
                "timestamp": "2026-08-29T09:15:00Z",
                "payment_format": "Wire",
                "currency": "USD",
            }
        }
    }


class GraphScoreResponse(BaseModel):
    tgn_score: Optional[float] = Field(None, ge=0, le=1, description="TGN risk probability")
    is_mule_probability: Optional[float] = Field(None, ge=0, le=1, description="Probability the sender is a mule account")
    ring_role: str = Field(..., description="HUB | BRIDGE | MULE | NONE")
    ring_probability: float = Field(..., ge=0, le=1)
    in_scc_ring: bool = Field(..., description="Whether src account is in an SCC ring")
    community_id: int = Field(-1, description="Louvain community ID (-1 if unknown)")
    pagerank: float = Field(0.0, description="GDS PageRank score")
    processing_time_ms: float
    degraded_mode: bool = Field(False, description="True if heuristic fallback was used")
    model_version: str = Field("unknown")


# ──────────────────────────────────────────────────────────────────────── #
# Endpoint
# ──────────────────────────────────────────────────────────────────────── #


@router.post("/graph-score", response_model=GraphScoreResponse)
async def graph_score(request_body: GraphScoreRequest, request: Request) -> GraphScoreResponse:
    """
    Score a transaction using the Temporal Graph Network and Neo4j GDS signals.

    Called by the Sprint 6 scoring-orchestrator in parallel with the AI Engine
    (/score) via Mono.zip(). Both calls execute concurrently — total wall-clock
    for the orchestrator is max(TGN_latency, LGBM_latency), not the sum.

    Response fields used by the orchestrator:
      - tgn_score → weighted 0.50x in the final risk fusion
      - ring_role + in_scc_ring → written to scoring_decisions.ring_flag/ring_role
      - degraded_mode → orchestrator uses 0.0 weight for TGN if True, renormalizes
    """
    t0 = time.perf_counter()

    # App state injected by lifespan (see main.py)
    app = request.app
    tgn_scorer = getattr(app.state, "tgn_scorer", None)
    neo4j = getattr(app.state, "neo4j_client", None)

    # Step 1: Fetch GDS signals (< 1ms if Neo4j healthy and account is indexed)
    from app.core.gds_runner import fetch_gds_signals
    from app.core.ring_detector import detect_ring_role

    gds_signals: dict = {}
    if neo4j is not None:
        gds_signals = await fetch_gds_signals(neo4j, request_body.from_account)

    # Step 2: TGN inference (≤ 20ms target)
    timestamp_s = request_body.timestamp.timestamp()

    if tgn_scorer is not None:
        result = await tgn_scorer.score(
            from_account=request_body.from_account,
            to_account=request_body.to_account,
            amount=request_body.amount,
            payment_format=request_body.payment_format,
            timestamp_s=timestamp_s,
            neo4j_src_props=gds_signals,
        )
        tgn_score = result.tgn_score
        tgn_mule_prob = result.is_mule_probability
        ring_role_tgn = result.ring_role
        ring_probability = result.ring_probability
        degraded = result.degraded_mode
        model_version = result.model_version
    else:
        # No scorer available at all — full degraded mode
        tgn_score = 0.5  # conservative neutral
        tgn_mule_prob = None
        ring_role_tgn = None
        ring_probability = 0.5
        degraded = True
        model_version = "unavailable"

    # Step 3: Ring detection (deterministic, < 0.1ms)
    ring_role, in_scc_ring = detect_ring_role(gds_signals, ring_role_tgn, tgn_mule_prob)

    processing_ms = (time.perf_counter() - t0) * 1000

    return GraphScoreResponse(
        tgn_score=tgn_score,
        is_mule_probability=tgn_mule_prob,
        ring_role=ring_role.value,
        ring_probability=ring_probability,
        in_scc_ring=in_scc_ring,
        community_id=int(gds_signals.get("community_id", -1)),
        pagerank=float(gds_signals.get("pagerank", 0.0)),
        processing_time_ms=round(processing_ms, 3),
        degraded_mode=degraded,
        model_version=model_version,
    )
