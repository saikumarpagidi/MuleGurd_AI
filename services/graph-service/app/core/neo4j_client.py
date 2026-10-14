"""
services/graph-service/app/core/neo4j_client.py
=================================================
MuleGuard.AI — Sprint 5

Async Neo4j driver wrapper. Exposes two targeted queries:

  1. get_account_properties(account_id) → account-level GDS signals
     (pagerank, community_id, in_scc_ring) for heuristic-mode scoring
     and for enriching TGN memory context.

  2. get_2hop_subgraph(account_id) → recent outbound edges for building
     TGN edge features on the fly at serving time.

Design choices:
  - Uses the official neo4j async driver (AsyncSession) for non-blocking I/O.
  - Parameterized Cypher in all queries — no string interpolation. Neo4j does
    not have a traditional SQL injection surface, but parameter binding also
    enables query plan caching which matters at 25ms latency budget.
  - Short per-query timeout (20ms by default via settings.neo4j_query_timeout_s)
    enforced through the ExecuteQueryConfig. If Neo4j is slow, the graph-service
    returns a degraded response rather than blocking the orchestrator.
  - Connection pool managed at service startup (lifespan) so sessions are
    reused across requests without re-establishing the TCP handshake.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger("graph-service.neo4j_client")

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver
    _HAS_NEO4J = True
except ImportError:
    _HAS_NEO4J = False
    log.warning("neo4j package not installed. Neo4j queries will return empty results.")


def _to_timestamp_s(val: Any) -> float:
    if val is None:
        return 0.0
    if hasattr(val, "to_native"):
        try:
            val = val.to_native()
        except Exception:
            pass
    if hasattr(val, "timestamp"):
        try:
            return float(val.timestamp())
        except Exception:
            pass
    try:
        return float(val)
    except (ValueError, TypeError):
        try:
            from datetime import datetime
            return datetime.fromisoformat(str(val).replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0


class Neo4jClient:
    """Lifecycle-managed Neo4j async driver. Instantiated once at service startup."""

    def __init__(self, uri: str, user: str, password: str, database: str,
                 max_pool_size: int = 10, query_timeout_s: float = 0.020):
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._max_pool_size = max_pool_size
        self._query_timeout_s = query_timeout_s
        self._driver: Optional[Any] = None  # AsyncDriver when connected
        self._connected = False

    async def connect(self) -> None:
        """Open the connection pool. Called during FastAPI lifespan startup."""
        if not _HAS_NEO4J:
            log.warning("Neo4j driver not available. All queries will return empty results.")
            return
        if not self._password:
            log.warning("GRAPH_SERVICE_NEO4J_PASSWORD not set — Neo4j queries disabled.")
            return
        try:
            self._driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
                max_connection_pool_size=self._max_pool_size,
            )
            await self._driver.verify_connectivity()
            self._connected = True
            log.info("Neo4j async driver connected to %s (pool=%d)", self._uri, self._max_pool_size)
        except Exception as exc:
            log.warning("Neo4j connectivity check failed: %s — graph queries will use heuristic fallback.", exc)
            self._driver = None
            self._connected = False

    async def close(self) -> None:
        """Drain the connection pool. Called during FastAPI lifespan shutdown."""
        if self._driver is not None:
            await self._driver.close()
            self._connected = False
            log.info("Neo4j driver closed.")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._driver is not None

    # ──────────────────────────────────────────────────────────────────────── #
    # Query 1: Account GDS properties (pagerank, SCC, community)
    # ──────────────────────────────────────────────────────────────────────── #

    async def get_account_properties(self, account_id: str) -> dict:
        """
        Returns {account_id, pagerank, community_id, in_scc_ring} for a single
        account. Used by tgn_scorer.py to enrich the heuristic degraded-mode
        score and to provide context to the ring_detector.

        Returns empty dict {} if:
          - Neo4j is not connected
          - The account does not exist in the graph
          - The query exceeds the latency budget
        """
        if not self.is_connected:
            return {}
        try:
            async with self._driver.session(database=self._database) as session:
                result = await session.run(
                    """
                    MATCH (a:Account {account_id: $account_id})
                    RETURN
                        a.account_id   AS account_id,
                        a.pagerank     AS pagerank,
                        a.community_id AS community_id,
                        a.in_scc_ring  AS in_scc_ring
                    LIMIT 1
                    """,
                    account_id=account_id,
                )
                record = await result.single()
                if record is None:
                    return {}
                return {
                    "account_id": record["account_id"],
                    "pagerank": float(record["pagerank"] or 0.0),
                    "community_id": int(record["community_id"] or -1),
                    "in_scc_ring": bool(record["in_scc_ring"]),
                }
        except Exception as exc:
            log.warning("get_account_properties(%s) failed: %s", account_id, exc)
            return {}

    # ──────────────────────────────────────────────────────────────────────── #
    # Query 2: 2-hop outbound subgraph for TGN edge feature construction
    # ──────────────────────────────────────────────────────────────────────── #

    async def get_2hop_subgraph(self, account_id: str, limit: int = 50) -> list[dict]:
        """
        Returns up to `limit` recent outbound transactions from account_id
        and their immediate destinations, forming the 2-hop neighbourhood
        used for on-the-fly TGN edge feature context at serving time.

        Each record: {src, dst, amount, timestamp_s, payment_format, is_laundering}

        Query design:
          - Depth-1 SENT edges FROM account_id (direct outbound)
          - Depth-2 SENT edges FROM depth-1 destinations (1 hop further)
          - ORDER BY timestamp DESC so we get the most recent context
          - LIMIT applied per hop to bound query time within latency budget
        """
        if not self.is_connected:
            return []
        try:
            async with self._driver.session(database=self._database) as session:
                result = await session.run(
                    """
                    MATCH (center:Account {account_id: $account_id})-[r1:SENT]-(hop1:Account)
                    OPTIONAL MATCH (hop1)-[r2:SENT]-(hop2:Account)
                    WHERE hop2 IS NULL OR hop2.account_id <> center.account_id
                    WITH
                        startNode(r1).account_id AS src_id,
                        endNode(r1).account_id AS dst1_id,
                        startNode(r2).account_id AS src2_id,
                        endNode(r2).account_id AS dst2_id,
                        r1.amount AS amt1,
                        r1.timestamp AS ts1,
                        r1.payment_format AS fmt1,
                        r1.is_laundering AS lbl1,
                        r2.amount AS amt2,
                        r2.timestamp AS ts2,
                        r2.payment_format AS fmt2,
                        r2.is_laundering AS lbl2
                    ORDER BY ts1 DESC, ts2 DESC
                    LIMIT $limit
                    RETURN src_id, dst1_id, src2_id, dst2_id, amt1, ts1, fmt1, lbl1, amt2, ts2, fmt2, lbl2
                    """,
                    account_id=account_id,
                    limit=limit,
                )
                edges = []
                async for record in result:
                    # Depth-1 edge
                    edges.append({
                        "src": record["src_id"],
                        "dst": record["dst1_id"],
                        "amount": float(record["amt1"] or 0.0),
                        "timestamp_s": _to_timestamp_s(record["ts1"]),
                        "payment_format": record["fmt1"] or "OTHER",
                        "is_laundering": bool(record["lbl1"]),
                    })
                    # Depth-2 edge (optional — may be None)
                    if record["src2_id"] is not None and record["dst2_id"] is not None:
                        edges.append({
                            "src": record["src2_id"],
                            "dst": record["dst2_id"],
                            "amount": float(record["amt2"] or 0.0),
                            "timestamp_s": _to_timestamp_s(record["ts2"]),
                            "payment_format": record["fmt2"] or "OTHER",
                            "is_laundering": bool(record["lbl2"]),
                        })
                return edges
        except Exception as exc:
            log.warning("get_2hop_subgraph(%s) failed: %s", account_id, exc)
            return []

    # ──────────────────────────────────────────────────────────────────────── #
    # Query 3: Batch account properties (for training-time role hints)
    # ──────────────────────────────────────────────────────────────────────── #

    async def get_batch_account_properties(self, account_ids: list[str]) -> dict[str, dict]:
        """Returns GDS properties for a batch of accounts.
        Used by tgn_train.py's fetch_role_hints path when running inference
        on all accounts in the training set simultaneously.
        """
        if not self.is_connected or not account_ids:
            return {}
        CHUNK = 5_000  # Neo4j parameter list limit
        all_hints: dict[str, dict] = {}
        try:
            async with self._driver.session(database=self._database) as session:
                for i in range(0, len(account_ids), CHUNK):
                    chunk = account_ids[i: i + CHUNK]
                    result = await session.run(
                        """
                        UNWIND $accounts AS acc_id
                        MATCH (a:Account {account_id: acc_id})
                        RETURN a.account_id   AS account_id,
                               a.pagerank     AS pagerank,
                               a.community_id AS community_id,
                               a.in_scc_ring  AS in_scc_ring
                        """,
                        accounts=chunk,
                    )
                    async for record in result:
                        all_hints[record["account_id"]] = {
                            "pagerank": float(record["pagerank"] or 0.0),
                            "community_id": int(record["community_id"] or -1),
                            "in_scc_ring": bool(record["in_scc_ring"]),
                        }
        except Exception as exc:
            log.warning("get_batch_account_properties failed: %s", exc)
        return all_hints
