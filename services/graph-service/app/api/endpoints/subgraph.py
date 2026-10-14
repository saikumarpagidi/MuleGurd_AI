"""
services/graph-service/app/api/endpoints/subgraph.py
======================================================
MuleGuard.AI — Sprint 5

GET /subgraph/{account_id} — Returns the 2-hop transaction subgraph for an
account, used by the Sprint 8 Investigator Dashboard's TGN Graph Explorer
(D3.js visualization).

This is NOT called in the hot scoring path (latency is less critical here)
— it's an analytical endpoint for investigators.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(tags=["Graph"])


class SubgraphEdge(BaseModel):
    src: str
    dst: str
    amount: float
    timestamp_s: float
    payment_format: str
    is_laundering: bool


class SubgraphResponse(BaseModel):
    account_id: str
    account_properties: dict
    edges: list[SubgraphEdge]
    edge_count: int
    neo4j_connected: bool


@router.get("/subgraph/{account_id}", response_model=SubgraphResponse)
async def get_subgraph(
    account_id: str,
    request: Request,
    limit: int = Query(default=50, ge=1, le=200, description="Max edges to return"),
) -> SubgraphResponse:
    """
    Returns the 2-hop transaction neighbourhood for an account.

    Used by the Investigator Dashboard's TGN Graph Explorer for manual
    investigation of flagged accounts. Not part of the real-time scoring path.
    """
    neo4j = getattr(request.app.state, "neo4j_client", None)

    if neo4j is None or not neo4j.is_connected:
        raise HTTPException(
            status_code=503,
            detail="Neo4j is not connected. Subgraph queries are unavailable.",
        )

    from app.core.gds_runner import fetch_gds_signals

    account_props = await fetch_gds_signals(neo4j, account_id)
    raw_edges = await neo4j.get_2hop_subgraph(account_id, limit=limit)

    edges = [SubgraphEdge(**e) for e in raw_edges]

    return SubgraphResponse(
        account_id=account_id,
        account_properties=account_props,
        edges=edges,
        edge_count=len(edges),
        neo4j_connected=True,
    )
