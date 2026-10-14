"""
services/graph-service/app/core/ring_detector.py
==================================================
MuleGuard.AI — Sprint 5

Ring detection logic: determines whether a source account is part of a
Strongly Connected Component (SCC) ring based on the Neo4j GDS SCC property
computed during Step 4A seeding.

This module is intentionally kept separate from tgn_scorer.py so that ring
detection can be used even when the TGN model is not available (e.g., during
Sprint 6 integration testing before TGN training completes).
"""

from __future__ import annotations

import logging
from enum import Enum

log = logging.getLogger("graph-service.ring_detector")


class RingRole(str, Enum):
    """Canonical ring role labels matching the scoring_decisions schema."""
    HUB = "HUB"
    BRIDGE = "BRIDGE"
    MULE = "MULE"
    NONE = "NONE"


def detect_ring_role(
    account_props: dict,
    tgn_ring_role: str | None = None,
    tgn_mule_prob: float | None = None,
) -> tuple[RingRole, bool]:
    """
    Determine the ring role and SCC membership for an account.

    Args:
        account_props: Neo4j GDS properties dict from neo4j_client.get_account_properties()
                       Expected keys: {pagerank, community_id, in_scc_ring}
        tgn_ring_role: Optional role prediction from the TGN model's role_head.
        tgn_mule_prob: Optional mule probability from the TGN model's is_mule_head.
                       Used as a confidence gate: TGN role is only trusted if
                       mule prob > 0.5 or TGN predicts HUB.

    Returns:
        (ring_role: RingRole, in_scc_ring: bool)

    Decision logic (precedence order):
        1. If TGN role is available AND confident (tgn_mule_prob > 0.5 or role==HUB) → use TGN prediction
        2. Else if in_scc_ring=True AND pagerank >= top-10% → HUB
        3. Else if in_scc_ring=True → MULE
        4. Else if account is in a ring-containing community but not in ring itself → BRIDGE
        5. Else → NONE
    """
    in_scc_ring = bool(account_props.get("in_scc_ring", False))
    pagerank = float(account_props.get("pagerank", 0.0))

    # Tier 1: TGN prediction
    if tgn_ring_role is not None and tgn_ring_role in RingRole.__members__:
        # Confidence gate: only trust TGN if it's confident it's a mule/hub
        confident = (
            (tgn_mule_prob is not None and tgn_mule_prob >= 0.5) or
            tgn_ring_role == "HUB"
        )
        if confident:
            return RingRole(tgn_ring_role), in_scc_ring

    # Tier 2: GDS heuristic
    if in_scc_ring:
        # PageRank > 0.001 is roughly the top-10% threshold observed in seeded data
        if pagerank >= 0.001:
            return RingRole.HUB, True
        return RingRole.MULE, True

    # Community signal: accounts in large communities (community_id >= 0) that
    # are not in a ring but sit between ring members are likely bridges.
    # Without full cross-community edge analysis (which requires a more expensive
    # Cypher traversal than fits in 25ms), we use community_id as a proxy.
    community_id = int(account_props.get("community_id", -1))
    if community_id >= 0 and pagerank >= 0.0005:  # above median pagerank
        return RingRole.BRIDGE, False

    return RingRole.NONE, False
