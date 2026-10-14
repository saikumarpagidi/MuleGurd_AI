"""
services/graph-service/app/core/gds_runner.py
===============================================
MuleGuard.AI — Sprint 5

Lightweight wrapper around Neo4j GDS algorithm calls that are needed at
serving time (NOT at seeding time — seeding runs via scripts/seed_neo4j.py).

At serving time, we only need READ access to the GDS-computed properties
already written to nodes during Step 4A:
  - Account.pagerank         (GDS PageRank, write mode)
  - Account.community_id     (GDS Louvain, write mode)
  - Account.in_scc_ring      (GDS SCC, written in batch)

This module wraps the Neo4j client to expose a clean "fetch GDS signals
for this account" interface with error handling and result normalization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .neo4j_client import Neo4jClient

log = logging.getLogger("graph-service.gds_runner")


async def fetch_gds_signals(
    neo4j: "Neo4jClient",
    account_id: str,
) -> dict:
    """
    Fetch the pre-computed GDS signals for a single account from Neo4j.

    Returns a normalized dict:
        {
          "pagerank": float,        # 0.0 if account not in graph
          "community_id": int,      # -1 if not set
          "in_scc_ring": bool,      # False if not set
          "found_in_graph": bool,   # True only if the account node exists in Neo4j
        }

    This is the READ-ONLY serving-time counterpart to the write-mode GDS
    algorithm runs in seed_neo4j.py. It does NOT trigger any GDS computation
    — it only reads the pre-computed property values from the Account node.

    Returning a normalized dict (never None) simplifies callers: they can
    always safely access result["pagerank"] without defensive None checks.
    """
    props = await neo4j.get_account_properties(account_id)

    if not props:
        # Account not found in graph (new account, or graph not seeded yet)
        log.debug("Account %s not found in Neo4j graph — returning zero GDS signals.", account_id)
        return {
            "pagerank": 0.0,
            "community_id": -1,
            "in_scc_ring": False,
            "found_in_graph": False,
        }

    return {
        "pagerank": float(props.get("pagerank", 0.0)),
        "community_id": int(props.get("community_id", -1)),
        "in_scc_ring": bool(props.get("in_scc_ring", False)),
        "found_in_graph": True,
    }
