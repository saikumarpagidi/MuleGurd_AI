#!/usr/bin/env python3
"""
scripts/seed_neo4j.py
======================
MuleGuard.AI — Sprint 4, Step 4A

Bulk-loads the IBM AML HI-Medium transaction graph (transactions_clean.parquet)
into Neo4j as (:Account)-[:SENT]->(:Account), then runs three GDS algorithms
and writes their outputs back onto Account nodes:

    PageRank                -> Account.pagerank        (float)
    Louvain                 -> Account.community_id     (int)
    Strongly Connected Comp -> Account.in_scc_ring       (bool)
                                Account.scc_id            (int)

Design notes
------------
* Idempotent: a uniqueness constraint on Account.account_id makes node MERGE
  safe to re-run. Edge load uses MERGE keyed on (from, to, timestamp, amount)
  to avoid duplicate SENT relationships on re-run.
* Scale: built for ~2.5M nodes / ~32M edges. Node and edge loads are batched
  and sent via UNWIND so a single Cypher statement handles a whole batch
  instead of one round-trip per row. Batch size is tunable via --batch-size.
* GDS: uses the Graph Data Science native graph projection (`gds.graph.project`)
  rather than a Cypher projection, since it is dramatically faster at this
  scale. Algorithms are run in "write" mode so results land directly on the
  stored nodes and the in-memory projection is dropped afterward.
* Safety: never fabricates data. If the parquet file or required columns are
  missing, the script fails fast with a clear error rather than silently
  producing a partial graph.

Usage
-----
    python scripts/seed_neo4j.py \
        --parquet data/processed/transactions_clean.parquet \
        --uri bolt://localhost:7687 \
        --user neo4j \
        --password "$NEO4J_PASSWORD" \
        --batch-size 50000

Environment variables (used as fallback for any --uri/--user/--password not
passed explicitly): NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD.

Exit codes: 0 success, 1 configuration/input error, 2 database error.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import Neo4jError, ServiceUnavailable

load_dotenv()

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("seed_neo4j")

REQUIRED_COLUMNS = {
    "from_account",
    "to_account",
    "amount",
    "timestamp",
    "is_laundering",
}

GDS_PROJECTION_NAME = "muleguard_graph"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SeedConfig:
    parquet_path: Path
    uri: str
    user: str
    password: str
    database: str
    batch_size: int
    skip_gds: bool
    dry_run: bool
    clear: bool
    sample: int | None


def parse_args(argv: list[str] | None = None) -> SeedConfig:
    parser = argparse.ArgumentParser(
        description="Seed Neo4j with the IBM AML transaction graph and run GDS algorithms."
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("ml/data/processed/transactions_clean.parquet"),
        help="Path to transactions_clean.parquet",
    )
    parser.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", ""))
    parser.add_argument("--database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Rows per UNWIND batch for node/edge loads (default: 50000)",
    )
    parser.add_argument(
        "--skip-gds",
        action="store_true",
        help="Load the graph only; skip PageRank/Louvain/SCC (useful for incremental re-runs)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Wipe the entire database before seeding (MATCH (n) DETACH DELETE n).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate the parquet file, print planned volumes, connect to DB, but write nothing.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N rows from the parquet file before loading.",
    )
    args = parser.parse_args(argv)

    if not args.password and not args.dry_run:
        parser.error(
            "Neo4j password not set. Pass --password or set NEO4J_PASSWORD."
        )

    return SeedConfig(
        parquet_path=args.parquet,
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
        batch_size=args.batch_size,
        skip_gds=args.skip_gds,
        dry_run=args.dry_run,
        clear=args.clear,
        sample=args.sample,
    )


# --------------------------------------------------------------------------- #
# Data loading / validation
# --------------------------------------------------------------------------- #


def load_and_validate_transactions(parquet_path: Path) -> pd.DataFrame:
    """Load transactions_clean.parquet and validate the columns this script needs.

    Fails fast and loudly rather than proceeding with a partial/garbage graph.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"parquet file not found: {parquet_path}")

    log.info("Reading %s ...", parquet_path)
    t0 = time.time()
    df = pd.read_parquet(parquet_path, columns=None)
    # BUG FIX (#9): the SENT edge represents what the sender paid, not what
    # the receiver got. amount_paid is the semantically correct column.
    # The previous version used amount_received, which differs due to FX.
    if "amount_paid" in df.columns:
        df = df.rename(columns={"amount_paid": "amount"})
    elif "amount_received" in df.columns:
        df = df.rename(columns={"amount_received": "amount"})
    log.info("Loaded %s rows in %.1fs", f"{len(df):,}", time.time() - t0)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"transactions_clean.parquet is missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )

    # Normalize dtypes defensively — upstream parquet may store timestamp as
    # string/int depending on the export step.
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")

    df["amount"] = df["amount"].astype("float64")
    df["is_laundering"] = df["is_laundering"].astype("bool")
    df["from_account"] = df["from_account"].astype("string")
    df["to_account"] = df["to_account"].astype("string")

    n_null = df[["from_account", "to_account", "amount", "timestamp"]].isnull().any(axis=1).sum()
    if n_null:
        log.warning(
            "%s rows have nulls in key columns and will be dropped before load", f"{n_null:,}"
        )
        df = df.dropna(subset=["from_account", "to_account", "amount", "timestamp"])

    n_accounts = pd.unique(pd.concat([df["from_account"], df["to_account"]])).shape[0]
    log.info(
        "Validated: %s transactions, %s unique accounts, %s flagged as laundering",
        f"{len(df):,}",
        f"{n_accounts:,}",
        f"{int(df['is_laundering'].sum()):,}",
    )
    return df


def iter_batches(df: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    n = len(df)
    for start in range(0, n, batch_size):
        yield df.iloc[start : start + batch_size]


# --------------------------------------------------------------------------- #
# Neo4j load
# --------------------------------------------------------------------------- #

CONSTRAINT_CYPHER = """
CREATE CONSTRAINT account_id_unique IF NOT EXISTS
FOR (a:Account) REQUIRE a.account_id IS UNIQUE
"""

# BUG FIX (#6): node MERGE and edge MERGE combined into ONE statement instead
# of two separate round-trips per batch. At 2.5M-node / 32M-edge scale this
# halves network round-trips, and makes each batch atomic — previously a
# crash between the node-write and edge-write calls could leave a batch's
# accounts created with no edges for that batch (recoverable via idempotent
# MERGE on retry, but wasteful and momentarily inconsistent).
NODE_AND_EDGE_BATCH_CYPHER = """
UNWIND $rows AS row
MERGE (a:Account {account_id: row.from_account})
MERGE (b:Account {account_id: row.to_account})
MERGE (a)-[r:SENT {
    from_account: row.from_account,
    to_account: row.to_account,
    timestamp: datetime(row.timestamp),
    amount: row.amount
}]->(b)
ON CREATE SET r.is_laundering = row.is_laundering
"""


def rows_for_batch(batch: pd.DataFrame) -> list[dict]:
    return [
        {
            "from_account": r.from_account,
            "to_account": r.to_account,
            "amount": float(r.amount),
            "timestamp": r.timestamp.isoformat(),
            "is_laundering": bool(r.is_laundering),
        }
        for r in batch.itertuples(index=False)
    ]


def ensure_constraint(driver: Driver, database: str) -> None:
    log.info("Ensuring uniqueness constraint on Account.account_id ...")
    with driver.session(database=database) as session:
        session.run(CONSTRAINT_CYPHER)


def load_graph(driver: Driver, database: str, df: pd.DataFrame, batch_size: int) -> None:
    total_batches = (len(df) + batch_size - 1) // batch_size
    log.info(
        "Loading graph: %s edges in %s batches of %s",
        f"{len(df):,}",
        total_batches,
        f"{batch_size:,}",
    )

    with driver.session(database=database) as session:
        for i, batch in enumerate(iter_batches(df, batch_size), start=1):
            rows = rows_for_batch(batch)
            try:
                session.execute_write(lambda tx: tx.run(NODE_AND_EDGE_BATCH_CYPHER, rows=rows).consume())
            except Neo4jError as exc:
                log.error("Batch %s/%s failed: %s", i, total_batches, exc)
                raise
            if i % 10 == 0 or i == total_batches:
                log.info("  batch %s/%s done", i, total_batches)

    log.info("Graph load complete.")


def verify_load(driver: Driver, database: str, expected_edges: int) -> None:
    with driver.session(database=database) as session:
        node_count = session.run("MATCH (a:Account) RETURN count(a) AS c").single()["c"]
        edge_count = session.run("MATCH ()-[r:SENT]->() RETURN count(r) AS c").single()["c"]
    log.info("Post-load counts: %s Account nodes, %s SENT edges", f"{node_count:,}", f"{edge_count:,}")
    if edge_count < expected_edges:
        log.warning(
            "Loaded edge count (%s) is lower than input row count (%s). "
            "This is expected if MERGE deduplicated exact repeat events; "
            "investigate if the gap is large.",
            f"{edge_count:,}",
            f"{expected_edges:,}",
        )


# --------------------------------------------------------------------------- #
# GDS: PageRank, Louvain, SCC
# --------------------------------------------------------------------------- #


def drop_projection_if_exists(driver: Driver, database: str, name: str) -> None:
    with driver.session(database=database) as session:
        exists = session.run(
            "CALL gds.graph.exists($name) YIELD exists RETURN exists", name=name
        ).single()["exists"]
        if exists:
            log.info("Dropping stale GDS projection '%s'", name)
            session.run("CALL gds.graph.drop($name)", name=name)


def project_graph(driver: Driver, database: str, name: str) -> None:
    log.info("Projecting native GDS graph '%s' ...", name)
    with driver.session(database=database) as session:
        result = session.run(
            """
            CALL gds.graph.project(
                $name,
                'Account',
                {
                    SENT: {
                        orientation: 'NATURAL'
                    }
                }
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """,
            name=name,
        ).single()
        log.info(
            "Projection ready: %s nodes, %s relationships",
            f"{result['nodeCount']:,}",
            f"{result['relationshipCount']:,}",
        )


def run_pagerank(driver: Driver, database: str, name: str) -> None:
    log.info("Running GDS PageRank -> Account.pagerank ...")
    with driver.session(database=database) as session:
        stats = session.run(
            """
            CALL gds.pageRank.write($name, {
                writeProperty: 'pagerank',
                maxIterations: 20,
                dampingFactor: 0.85
            })
            YIELD nodePropertiesWritten, ranIterations, didConverge
            RETURN nodePropertiesWritten, ranIterations, didConverge
            """,
            name=name,
        ).single()
        log.info(
            "PageRank done: wrote %s properties in %s iterations (converged=%s)",
            stats["nodePropertiesWritten"],
            stats["ranIterations"],
            stats["didConverge"],
        )


def run_louvain(driver: Driver, database: str, name: str) -> None:
    log.info("Running GDS Louvain -> Account.community_id ...")
    with driver.session(database=database) as session:
        stats = session.run(
            """
            CALL gds.louvain.write($name, {
                writeProperty: 'community_id'
            })
            YIELD communityCount, modularity, nodePropertiesWritten
            RETURN communityCount, modularity, nodePropertiesWritten
            """,
            name=name,
        ).single()
        log.info(
            "Louvain done: %s communities, modularity=%.4f, wrote %s properties",
            stats["communityCount"],
            stats["modularity"],
            stats["nodePropertiesWritten"],
        )
        if stats["communityCount"] < 100:
            log.warning(
                "Louvain produced only %s communities; the forward plan's "
                "'Done when' criterion expects 100+. Verify graph connectivity.",
                stats["communityCount"],
            )


def run_scc(driver: Driver, database: str, name: str) -> None:
    """Strongly Connected Components -> scc_id + in_scc_ring boolean flag.

    gds.wcc is weak connectivity; SCC (directed circular flow) is what the
    plan calls for, so we use gds.scc explicitly. A node is flagged
    in_scc_ring=true only if its component has size > 1 (a lone node is
    trivially its own SCC and is not a laundering ring).
    """
    log.info("Running GDS Strongly Connected Components -> Account.scc_id / in_scc_ring ...")
    with driver.session(database=database) as session:
        stats = session.run(
            """
            CALL gds.scc.write($name, {
                writeProperty: 'scc_id'
            })
            YIELD componentCount, nodePropertiesWritten
            RETURN componentCount, nodePropertiesWritten
            """,
            name=name,
        ).single()
        log.info(
            "SCC done: %s components, wrote %s properties",
            stats["componentCount"],
            stats["nodePropertiesWritten"],
        )

        # Flag ring membership: component size > 1 implies a real circular
        # money-flow loop rather than a trivial singleton "component".
        # We fetch the account_ids first to avoid OOM from collecting Node objects.
        res = session.run(
            """
            MATCH (a:Account)
            WITH a.scc_id AS scc_id, collect(a.account_id) AS members
            WHERE size(members) > 1
            UNWIND members AS acc_id
            RETURN acc_id
            """
        )
        ring_account_ids = [record["acc_id"] for record in res]
        accounts_in_rings = len(ring_account_ids)

        # Initialize all to false in batched transactions
        session.run(
            """
            MATCH (n:Account)
            CALL {
                WITH n
                SET n.in_scc_ring = false
            } IN TRANSACTIONS OF 50000 ROWS
            """
        )

        # Batch update the ring members to true using the indexed account_id
        if ring_account_ids:
            batch_sz = 10000
            for i in range(0, accounts_in_rings, batch_sz):
                batch = ring_account_ids[i:i + batch_sz]
                session.run(
                    """
                    UNWIND $batch_ids AS acc_id
                    MATCH (n:Account {account_id: acc_id})
                    SET n.in_scc_ring = true
                    """,
                    batch_ids=batch
                )

        log.info(
            "Ring flagging done: %s accounts across multi-node SCCs flagged in_scc_ring=true",
            f"{accounts_in_rings:,}",
        )


def run_gds_pipeline(driver: Driver, database: str) -> None:
    drop_projection_if_exists(driver, database, GDS_PROJECTION_NAME)
    project_graph(driver, database, GDS_PROJECTION_NAME)
    try:
        run_pagerank(driver, database, GDS_PROJECTION_NAME)
        run_louvain(driver, database, GDS_PROJECTION_NAME)
        run_scc(driver, database, GDS_PROJECTION_NAME)
    finally:
        with driver.session(database=database) as session:
            session.run("CALL gds.graph.drop($name, false)", name=GDS_PROJECTION_NAME)
        log.info("Dropped in-memory GDS projection '%s'", GDS_PROJECTION_NAME)


# --------------------------------------------------------------------------- #
# Acceptance checks (mirrors the plan's "Done when" criteria)
# --------------------------------------------------------------------------- #


def run_acceptance_checks(driver: Driver, database: str, skip_gds: bool) -> bool:
    """BUG FIX (#5/#8): previously this ran the same checks regardless of
    --skip-gds, so a deliberate edge-only incremental reload (exactly what
    --skip-gds exists for) would always report FAIL and exit 2, because
    community_id/pagerank/in_scc_ring are never populated on that path. GDS-
    dependent checks are now skipped when GDS itself was skipped.
    """
    log.info("Running acceptance checks against 'Done when' criteria ...")
    ok = True
    with driver.session(database=database) as session:
        node_count = session.run("MATCH (a:Account) RETURN count(a) AS c").single()["c"]
        edge_count = session.run("MATCH ()-[r:SENT]->() RETURN count(r) AS c").single()["c"]

        checks = [("Account nodes + SENT relationships exist", node_count > 0 and edge_count > 0)]

        if skip_gds:
            log.info("--skip-gds was set: skipping GDS-dependent acceptance checks (pagerank/community_id/in_scc_ring).")
        else:
            ring_count = session.run(
                "MATCH (a:Account) WHERE a.in_scc_ring = true RETURN count(a) AS c"
            ).single()["c"]
            community_count = session.run(
                "MATCH (a:Account) WHERE a.community_id IS NOT NULL "
                "RETURN count(DISTINCT a.community_id) AS c"
            ).single()["c"]
            pagerank_written = session.run(
                "MATCH (a:Account) WHERE a.pagerank IS NOT NULL RETURN count(a) AS c"
            ).single()["c"]
            checks.extend([
                ("in_scc_ring=true nodes exist (circular flows confirmed)", ring_count > 0),
                ("community_id spans 100+ Louvain communities", community_count >= 100),
                ("pagerank written to all nodes", pagerank_written == node_count),
            ])

    for description, passed in checks:
        status = "PASS" if passed else "FAIL"
        (log.info if passed else log.error)("[%s] %s", status, description)
        ok = ok and passed

    return ok


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)

    try:
        df = load_and_validate_transactions(cfg.parquet_path)
        if cfg.sample:
            log.info("Sampling %s rows (--sample flag active)", f"{cfg.sample:,}")
            df = df.sample(n=min(cfg.sample, len(df)), random_state=42).copy()
    except (FileNotFoundError, ValueError) as exc:
        log.error("Input validation failed: %s", exc)
        return 1

    if cfg.dry_run:
        n_accounts = pd.unique(pd.concat([df["from_account"], df["to_account"]])).shape[0]
        n_batches = (len(df) + cfg.batch_size - 1) // cfg.batch_size
        log.info(
            "[DRY RUN] Would load %s edges / ~%s accounts in %s batches. No writes performed.",
            f"{len(df):,}",
            f"{n_accounts:,}",
            n_batches,
        )
        return 0

    try:
        driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
        driver.verify_connectivity()
    except ServiceUnavailable as exc:
        log.error("Could not connect to Neo4j at %s: %s", cfg.uri, exc)
        return 2

    try:
        if cfg.clear and not cfg.dry_run:
            log.warning("CLEAR flag passed: wiping entire Neo4j database...")
            with driver.session(database=cfg.database) as session:
                log.info("Deleting relationships...")
                while True:
                    c = session.run("MATCH ()-[r]->() WITH r LIMIT 100000 DELETE r RETURN count(r) AS c").single()["c"]
                    if c == 0: break
                
                # BUG FIX (#7): DETACH DELETE instead of plain DELETE for
                # defense-in-depth. Plain DELETE throws if any relationship
                # is still attached (e.g. a GDS-created or out-of-band edge
                # type this script doesn't know about).
                log.info("Deleting nodes...")
                while True:
                    c = session.run("MATCH (n) WITH n LIMIT 100000 DETACH DELETE n RETURN count(n) AS c").single()["c"]
                    if c == 0: break
            log.info("Database wiped.")

        ensure_constraint(driver, cfg.database)
        load_graph(driver, cfg.database, df, cfg.batch_size)
        verify_load(driver, cfg.database, expected_edges=len(df))

        if not cfg.skip_gds:
            run_gds_pipeline(driver, cfg.database)

        passed = run_acceptance_checks(driver, cfg.database, cfg.skip_gds)
        if not passed:
            log.error("One or more acceptance checks failed. See log above.")
            return 2

        log.info("Sprint 4A complete: graph seeded and GDS properties written successfully.")
        return 0

    except Neo4jError as exc:
        log.error("Neo4j operation failed: %s", exc)
        return 2
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
