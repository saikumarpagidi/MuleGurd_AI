#!/usr/bin/env python3
"""
ml/training/static_gnn_baseline.py
====================================
MuleGuard.AI — Sprint 4, Step 4B

Trains a 2-layer GraphSAGE + GAT static baseline on the Account graph seeded
into Neo4j by scripts/seed_neo4j.py (Step 4A), and records its AUC as the
number that the Temporal GNN (Sprint 5) must beat by >= 1 percentage point.

This is intentionally a WEAKER baseline than the eventual TGN:
  * Static node features only (no per-transaction edge features, no time).
  * An 80/20 RANDOM split rather than a temporal split, so it does not test
    the model's ability to generalize forward in time.
That gap is the point: it is the floor the temporal model has to clear.

Node features pulled from Neo4j (must exist — i.e. Step 4A already ran):
    pagerank, community_id, in_scc_ring          <- from GDS (Step 4A)
    lifetime_txn_count, lifetime_total_sent_log,
    night_activity_ratio, fan_out_24h            <- computed here from SENT edges

Target: Account.is_laundering, aggregated from transaction-level labels
        (an account is a positive if it appears as the source of at least
        one transaction flagged is_laundering=true).

Architecture: GraphSAGEConv -> GATConv (2 layers), binary classification head.

Usage
-----
    python ml/training/static_gnn_baseline.py \
        --uri bolt://localhost:7687 --user neo4j --password "$NEO4J_PASSWORD" \
        --epochs 100 --hidden-dim 64 --out ml/models/baseline_gnn_auc.json

    # Lean/fast run (token/time constrained, still a defensible number):
    python ml/training/static_gnn_baseline.py --epochs 30 --hidden-dim 32 ...

Outputs
-------
    ml/models/static_gnn_baseline.pt   — trained model weights
    ml/models/baseline_gnn_auc.json    — {"static_gnn_auc": ..., "n_nodes": ..., ...}

Exit codes: 0 success, 1 config/data error, 2 database error.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import timedelta
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv

load_dotenv()

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("static_gnn_baseline")

FEATURE_NAMES = [
    "pagerank",
    "community_id",
    "in_scc_ring",
    "lifetime_txn_count",
    "lifetime_total_sent_log",
    "night_activity_ratio",
    "fan_out_24h",
]

AUC_TARGET = 0.85  # go/no-go gate from the forward plan


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainConfig:
    uri: str
    user: str
    password: str
    database: str
    epochs: int
    hidden_dim: int
    lr: float
    weight_decay: float
    dropout: float
    val_fraction: float
    seed: int
    patience: int
    model_out: Path
    metrics_out: Path
    device: str
    synthetic_smoke_test: bool
    mule_loss_weight: float
    role_loss_weight: float
    mule_threshold: float
    resume: bool


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    p = argparse.ArgumentParser(description="Train the Sprint 4B static GNN baseline.")
    p.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    p.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    p.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", ""))
    p.add_argument("--database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--val-fraction", type=float, default=0.2, help="80/20 split per the plan")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=15, help="early stopping patience (epochs)")
    p.add_argument("--out", type=Path, default=Path("ml/models/baseline_gnn_multitask_auc.json"))
    p.add_argument("--model-out", type=Path, default=Path("ml/models/static_gnn_multitask.pt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mule-loss-weight", type=float, default=0.2)
    p.add_argument("--role-loss-weight", type=float, default=0.1)
    p.add_argument("--mule-threshold", type=float, default=0.3)
    p.add_argument("--resume", action="store_true", help="Resume training from model_out checkpoint if it exists.")
    p.add_argument(
        "--synthetic-smoke-test",
        action="store_true",
        help="Bypass Neo4j and train on a small synthetic graph. For CI / local "
        "smoke testing of the pipeline only — never use its AUC as the reported gate metric.",
    )
    args = p.parse_args(argv)

    if not args.password and not args.synthetic_smoke_test:
        p.error("Neo4j password not set. Pass --password or set NEO4J_PASSWORD.")

    return TrainConfig(
        uri=args.uri,
        user=args.user,
        password=args.password,
        database=args.database,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        val_fraction=args.val_fraction,
        seed=args.seed,
        patience=args.patience,
        model_out=args.model_out,
        metrics_out=args.out,
        device=args.device,
        synthetic_smoke_test=args.synthetic_smoke_test,
        mule_loss_weight=args.mule_loss_weight,
        role_loss_weight=args.role_loss_weight,
        mule_threshold=args.mule_threshold,
        resume=args.resume,
    )


# --------------------------------------------------------------------------- #
# Neo4j -> PyG Data
# --------------------------------------------------------------------------- #

# Computes the four behavioral features (lifetime_txn_count,
# lifetime_total_sent_log, night_activity_ratio, fan_out_24h) directly in
# Cypher from the SENT edges, and reads the three GDS features written by
# Step 4A off the node. Also derives the account-level label.
FETCH_GRAPH_CYPHER = """
MATCH (a:Account)
OPTIONAL MATCH (a)-[out:SENT]->()
WITH a,
     count(out) AS lifetime_txn_count,
     coalesce(sum(out.amount), 0.0) AS lifetime_total_sent,
     sum(CASE WHEN out.timestamp.hour < 6 OR out.timestamp.hour >= 22 THEN 1 ELSE 0 END) AS night_txns,
     sum(CASE WHEN out.timestamp >= $cutoff_24h THEN 1 ELSE 0 END) AS fan_out_24h,
     sum(CASE WHEN out.is_laundering THEN 1 ELSE 0 END) AS laundering_out_count
RETURN a.account_id AS account_id,
       coalesce(a.pagerank, 0.0) AS pagerank,
       coalesce(a.community_id, -1) AS community_id,
       coalesce(a.in_scc_ring, false) AS in_scc_ring,
       lifetime_txn_count,
       lifetime_total_sent,
       CASE WHEN lifetime_txn_count > 0 THEN toFloat(night_txns) / lifetime_txn_count ELSE 0.0 END AS night_activity_ratio,
       fan_out_24h,
       (laundering_out_count > 0) AS is_laundering,
       CASE WHEN lifetime_txn_count > 0 THEN toFloat(laundering_out_count) / lifetime_txn_count ELSE 0.0 END AS fraud_ratio
"""

FETCH_EDGES_CYPHER = """
MATCH (a:Account)-[:SENT]->(b:Account)
RETURN DISTINCT a.account_id AS src, b.account_id AS dst
"""


# BUG FIX (#1): fan_out_24h must be relative to the DATASET's own timeline, not the
# real-world wall clock. The IBM AML data is historical (dated within its own
# ~27-day window); comparing it against datetime.now() means "timestamp >=
# cutoff" is false for every single row, silently zeroing out this feature
# for every account. Fetch the graph's own max SENT timestamp once and use
# THAT as the reference point for "last 24 hours" instead.
FETCH_MAX_TIMESTAMP_CYPHER = """
MATCH ()-[r:SENT]->()
RETURN max(r.timestamp) AS max_ts
"""


def fetch_graph_from_neo4j(cfg: TrainConfig) -> tuple[list[dict], list[tuple[str, str]]]:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable

    log.info("Connecting to Neo4j at %s ...", cfg.uri)
    try:
        driver = GraphDatabase.driver(cfg.uri, auth=(cfg.user, cfg.password))
        driver.verify_connectivity()
    except ServiceUnavailable as exc:
        raise RuntimeError(f"Could not connect to Neo4j at {cfg.uri}: {exc}") from exc

    try:
        with driver.session(database=cfg.database) as session:
            # BUG FIX (#1): compute cutoff from the graph's own max timestamp
            max_ts_record = session.run(FETCH_MAX_TIMESTAMP_CYPHER).single()
            max_ts = max_ts_record["max_ts"] if max_ts_record else None
            if max_ts is None:
                raise RuntimeError(
                    "Could not determine the graph's max SENT.timestamp. "
                    "Has scripts/seed_neo4j.py (Step 4A) been run against this database?"
                )
            # neo4j.time.DateTime -> python datetime for arithmetic.
            max_ts_py = max_ts.to_native()
            cutoff_24h = max_ts_py - timedelta(days=1)
            log.info(
                "fan_out_24h reference window: dataset max timestamp = %s, "
                "cutoff = %s (dataset-relative, NOT wall-clock now())",
                max_ts_py, cutoff_24h,
            )

            log.info("Fetching node features + labels (this aggregates over SENT edges) ...")
            node_rows = [dict(r) for r in session.run(FETCH_GRAPH_CYPHER, cutoff_24h=cutoff_24h)]
            if not node_rows:
                raise RuntimeError(
                    "No Account nodes returned. Has scripts/seed_neo4j.py (Step 4A) "
                    "been run against this database?"
                )
            log.info("Fetched %s account nodes", f"{len(node_rows):,}")

            log.info("Fetching SENT edge list ...")
            edge_rows = [(r["src"], r["dst"]) for r in session.run(FETCH_EDGES_CYPHER)]
            log.info("Fetched %s distinct directed edges", f"{len(edge_rows):,}")
    finally:
        driver.close()

    return node_rows, edge_rows


def build_synthetic_graph(n_accounts: int = 4000, seed: int = 42) -> tuple[list[dict], list[tuple[str, str]]]:
    """Small synthetic Account graph with the same schema Neo4j would return.

    Used only for --synthetic-smoke-test to validate the training pipeline
    end-to-end without a live database. Not a substitute for the real gate metric.
    """
    rng = np.random.default_rng(seed)
    ids = [f"ACC{i:06d}" for i in range(n_accounts)]

    # Plant a "ring" subpopulation with different feature statistics and a
    # higher laundering rate, so the model has real (if easy) signal to learn.
    is_ring = rng.random(n_accounts) < 0.05

    node_rows = []
    for i, acc_id in enumerate(ids):
        ring = is_ring[i]
        node_rows.append(
            {
                "account_id": acc_id,
                "pagerank": float(rng.exponential(0.3 if ring else 0.05)),
                "community_id": int(rng.integers(0, 150)),
                "in_scc_ring": bool(ring),
                "lifetime_txn_count": int(rng.poisson(40 if ring else 8)),
                "lifetime_total_sent": float(rng.exponential(50_000 if ring else 2_000)),
                "night_activity_ratio": float(np.clip(rng.beta(3, 2) if ring else rng.beta(1, 5), 0, 1)),
                "fan_out_24h": int(rng.poisson(6 if ring else 1)),
                "is_laundering": bool(ring and rng.random() < 0.6),
                "fraud_ratio": float(rng.uniform(0.3, 1.0) if ring else rng.uniform(0.0, 0.1)),
            }
        )

    edge_rows = []
    for i in range(n_accounts):
        n_out = rng.poisson(3)
        targets = rng.choice(n_accounts, size=min(n_out, n_accounts - 1), replace=False)
        for t in targets:
            if t != i:
                edge_rows.append((ids[i], ids[t]))

    return node_rows, edge_rows


# BUG FIX (#3): Continuous features get z-scored; community_id is handled
# separately as a categorical embedding rather than being z-scored alongside
# them, since Louvain community IDs are arbitrary bucket labels with no
# numeric/ordinal meaning — "community 5" is not "less than" "community 240"
# in any sense the model should learn as a magnitude.
CONTINUOUS_FEATURE_NAMES = [
    "pagerank",
    "in_scc_ring",
    "lifetime_txn_count",
    "lifetime_total_sent_log",
    "night_activity_ratio",
    "fan_out_24h",
]


def build_pyg_data(node_rows: list[dict], edge_rows: list[tuple[str, str]], cfg: TrainConfig) -> tuple[Data, list[str]]:
    """Assemble node feature matrix, edge_index, community_id, and labels into
    a PyG Data object. Computes multi-task labels (is_mule and role_id).
    """
    account_ids = [r["account_id"] for r in node_rows]
    id_to_idx = {aid: i for i, aid in enumerate(account_ids)}

    log_amounts = np.log1p(np.array([r["lifetime_total_sent"] for r in node_rows], dtype=np.float64))

    # BUG FIX (#3): community_id removed from continuous features
    continuous_matrix = np.stack(
        [
            np.array([r["pagerank"] for r in node_rows], dtype=np.float64),
            np.array([float(r["in_scc_ring"]) for r in node_rows], dtype=np.float64),
            np.array([r["lifetime_txn_count"] for r in node_rows], dtype=np.float64),
            log_amounts,
            np.array([r["night_activity_ratio"] for r in node_rows], dtype=np.float64),
            np.array([r["fan_out_24h"] for r in node_rows], dtype=np.float64),
        ],
        axis=1,
    )

    # community_id: -1 means "no community assigned" (shouldn't happen after
    # a real Louvain run, but defend against it). Shift to a 0-based index so
    # -1 doesn't collide with a real community id 0, and reserve index 0 for
    # "missing" in the embedding table.
    raw_community = np.array([r["community_id"] for r in node_rows], dtype=np.int64)
    community_idx = np.where(raw_community < 0, 0, raw_community + 1)
    num_communities = int(community_idx.max()) + 1

    is_laundering = np.array([float(r["is_laundering"]) for r in node_rows], dtype=np.float32)
    fraud_ratios = np.array([float(r["fraud_ratio"]) for r in node_rows], dtype=np.float32)
    is_mule = (fraud_ratios >= cfg.mule_threshold).astype(np.float32)

    # Replicate v4 role heuristics exactly
    role_ids = np.zeros(len(node_rows), dtype=np.int64) # NONE = 0
    for i, r in enumerate(node_rows):
        pr = r["pagerank"]
        scc = r["in_scc_ring"]
        if pr > 0.15:
            role_ids[i] = 1 # HUB
        elif scc:
            role_ids[i] = 2 # BRIDGE
        elif is_mule[i]:
            role_ids[i] = 3 # MULE

    src, dst = [], []
    dropped = 0
    for s, d in edge_rows:
        if s in id_to_idx and d in id_to_idx:
            src.append(id_to_idx[s])
            dst.append(id_to_idx[d])
        else:
            dropped += 1
    if dropped:
        log.warning("%s edges referenced accounts not in the node set and were dropped", dropped)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x_continuous = torch.tensor(continuous_matrix, dtype=torch.float32)
    community_id = torch.tensor(community_idx, dtype=torch.long)

    y = torch.tensor(is_laundering, dtype=torch.float32)
    y_mule = torch.tensor(is_mule, dtype=torch.float32)
    y_role = torch.tensor(role_ids, dtype=torch.long)

    data = Data(x=x_continuous, edge_index=edge_index, y=y, 
                community_id=community_id, y_mule=y_mule, y_role=y_role)
    data.num_communities = num_communities  # not a tensor; PyG leaves scalars alone
    log.info(
        "Built PyG graph: %s nodes, %s edges, %s continuous features + 1 categorical "
        "(community_id, %s distinct buckets), positive rate %.3f%%, mule rate %.3f%%",
        data.num_nodes,
        data.num_edges,
        x_continuous.shape[1],
        num_communities,
        100.0 * is_laundering.mean(),
        100.0 * is_mule.mean(),
    )
    return data, account_ids


def make_split(n: int, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """80/20 RANDOM split, intentionally not temporal — see module docstring."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    return train_mask, val_mask


# --------------------------------------------------------------------------- #
# Model: 2-layer GraphSAGE -> GAT
# --------------------------------------------------------------------------- #


class StaticGNNMultiTask(nn.Module):
    """2-layer static GNN extended for multi-task learning (predicting laundering, mule, and role).
    
    This mirrors the architecture of TGN v4's node-level heads, but uses static
    GraphSAGE/GAT aggregations instead of chronological GRU memory.

    This mirrors the plan's "GraphSAGE + GAT (2 layers)" spec: layer 1
    aggregates broad neighborhood signal, layer 2 lets attention focus on
    the most informative neighbors (e.g. a single high-pagerank hub in a ring).

    FIX #3: community_id is embedded via nn.Embedding rather than z-scored
    alongside the continuous features. Louvain community IDs are categorical
    bucket labels with no numeric ordering; an embedding lets the model learn
    whatever similarity structure actually exists between communities (if
    any) instead of being told a false ordinal relationship.
    """

    def __init__(self, continuous_dim: int, num_communities: int, hidden_dim: int, dropout: float,
                 gat_heads: int = 4, community_embed_dim: int = 8):
        super().__init__()
        self.community_embedding = nn.Embedding(num_communities, community_embed_dim)
        in_dim = continuous_dim + community_embed_dim
        self.sage = SAGEConv(in_dim, hidden_dim)
        self.gat = GATConv(hidden_dim, hidden_dim, heads=gat_heads, concat=False, dropout=dropout)
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.is_mule_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.role_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4), # NONE, HUB, BRIDGE, MULE
        )

    def forward(self, x_continuous: torch.Tensor, community_id: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        community_vec = self.community_embedding(community_id)
        x = torch.cat([x_continuous, community_vec], dim=-1)
        h = self.sage(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gat(h, edge_index)
        h = F.elu(h)
        logits = self.head(h).squeeze(-1)
        mule_logits = self.is_mule_head(h).squeeze(-1)
        role_logits = self.role_head(h)
        return logits, mule_logits, role_logits


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


def train(cfg: TrainConfig, data: Data) -> dict:
    from torch_geometric.loader import NeighborLoader

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)

    train_mask, val_mask = make_split(data.num_nodes, cfg.val_fraction, cfg.seed)

    # BUG FIX (#2): standardize using TRAIN-ONLY statistics, then apply that same
    # scale to the validation rows. The previous version standardized using
    # mean/std computed over train+val combined, which leaks the validation
    # set's distribution into the features the model trains on.
    train_mean = data.x[train_mask].mean(dim=0)
    train_std = data.x[train_mask].std(dim=0)
    train_std = torch.where(train_std < 1e-8, torch.ones_like(train_std), train_std)
    data.x = (data.x - train_mean) / train_std
    log.info("Standardized continuous features using TRAIN-split-only mean/std (no validation leakage).")

    n_pos = int(data.y[train_mask].sum().item())
    n_neg = int(train_mask.sum().item()) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    log.info(
        "Train split: %s nodes (%s positive, %s negative) | pos_weight=%.2f",
        int(train_mask.sum()),
        n_pos,
        n_neg,
        pos_weight.item(),
    )

    # Use NeighborLoader for mini-batching on large graphs to prevent OOM
    # Sampling 10 neighbors per hop for 2 layers
    train_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10],
        batch_size=1024,
        input_nodes=train_mask,
        shuffle=True,
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors=[10, 10],
        batch_size=4096,
        input_nodes=val_mask,
        shuffle=False,
    )

    n_mule_pos = int(data.y_mule[train_mask].sum().item())
    n_mule_neg = int(train_mask.sum().item()) - n_mule_pos
    mule_pos_weight = torch.tensor([n_mule_neg / max(n_mule_pos, 1)], device=device)
    
    train_roles = data.y_role[train_mask]
    role_counts = torch.bincount(train_roles, minlength=4)
    role_weights = 1.0 / (role_counts.float() + 1e-6)
    role_weights = (role_weights / role_weights.sum()) * 4.0
    role_weights = role_weights.to(device)

    model = StaticGNNMultiTask(
        continuous_dim=data.x.shape[1],
        num_communities=data.num_communities,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    criterion_laundering = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_mule = nn.BCEWithLogitsLoss(pos_weight=mule_pos_weight)
    criterion_role = nn.CrossEntropyLoss(weight=role_weights)

    best_val_auc = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []
    start_epoch = 1

    if cfg.resume and cfg.model_out.exists():
        log.info("Resuming training from checkpoint: %s", cfg.model_out)
        ckpt = torch.load(cfg.model_out, map_location=device)
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt["epoch"] + 1
            best_val_auc = ckpt["best_val_auc"]
            history = ckpt.get("history", [])
            epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            # It's an old-format weights-only checkpoint
            model.load_state_dict(ckpt)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    log.info("Training for up to %s epochs (patience=%s) on %s using NeighborLoader ...", cfg.epochs, cfg.patience, device)
    t0 = time.time()

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        total_loss = 0
        train_probs, train_true = [], []
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # The NeighborLoader puts the target nodes at the beginning of the batch
            batch_size = batch.batch_size
            logits, mule_logits, role_logits = model(batch.x, batch.community_id, batch.edge_index)
            logits = logits[:batch_size]
            mule_logits = mule_logits[:batch_size]
            role_logits = role_logits[:batch_size]
            
            targets = batch.y[:batch_size]
            mule_targets = batch.y_mule[:batch_size]
            role_targets = batch.y_role[:batch_size]
            
            loss_laundering = criterion_laundering(logits, targets)
            loss_mule = criterion_mule(mule_logits, mule_targets)
            loss_role = criterion_role(role_logits, role_targets)
            
            loss = loss_laundering + cfg.mule_loss_weight * loss_mule + cfg.role_loss_weight * loss_role
            loss.backward()
            optimizer.step()
            
            total_loss += float(loss.item()) * batch_size
            train_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
            train_true.extend(targets.cpu().numpy())

        avg_loss = total_loss / int(train_mask.sum())
        train_auc = _safe_auc(np.array(train_true), np.array(train_probs))

        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                batch_size = batch.batch_size
                logits, mule_logits, role_logits = model(batch.x, batch.community_id, batch.edge_index)
                logits = logits[:batch_size]
                targets = batch.y[:batch_size]
                
                val_probs.extend(torch.sigmoid(logits).cpu().numpy())
                val_true.extend(targets.cpu().numpy())

        val_auc = _safe_auc(np.array(val_true), np.array(val_probs))

        history.append({"epoch": epoch, "loss": avg_loss, "train_auc": train_auc, "val_auc": val_auc})

        if True: # Changed from epoch % 5 == 0 to print every epoch since they take several minutes
            log.info(
                "epoch %4d | loss %.4f | train_auc %.4f | val_auc %.4f",
                epoch,
                avg_loss,
                train_auc,
                val_auc,
            )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            
            cfg.model_out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": best_state,
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_auc": best_val_auc,
                "history": history,
                "epochs_without_improvement": epochs_without_improvement
            }, cfg.model_out)
            log.info("Saved best checkpoint to %s", cfg.model_out)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                log.info("Early stopping at epoch %s (no val_auc improvement for %s epochs)", epoch, cfg.patience)
                break

    elapsed = time.time() - t0
    log.info("Training complete in %.1fs. Best val_auc = %.4f", elapsed, best_val_auc)

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on validation set with best model
    model.eval()
    val_probs, val_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            batch_size = batch.batch_size
            logits, mule_logits, role_logits = model(batch.x, batch.community_id, batch.edge_index)
            logits = logits[:batch_size]
            targets = batch.y[:batch_size]
            val_probs.extend(torch.sigmoid(logits).cpu().numpy())
            val_true.extend(targets.cpu().numpy())
            
    val_true = np.array(val_true)
    val_probs = np.array(val_probs)
    best_val_pr_auc = _safe_pr_auc(val_true, val_probs)

    # Evaluate Mule and Role on validation set
    model.eval()
    val_mule_probs, val_mule_true = [], []
    val_role_preds, val_role_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            batch_size = batch.batch_size
            _, mule_logits, role_logits = model(batch.x, batch.community_id, batch.edge_index)
            val_mule_probs.extend(torch.sigmoid(mule_logits[:batch_size]).cpu().numpy())
            val_mule_true.extend(batch.y_mule[:batch_size].cpu().numpy())
            
            val_role_preds.extend(role_logits[:batch_size].argmax(dim=-1).cpu().numpy())
            val_role_true.extend(batch.y_role[:batch_size].cpu().numpy())

    return {
        "model": model,
        "best_val_auc": best_val_auc,
        "best_val_pr_auc": best_val_pr_auc,
        "val_true": val_true,
        "val_probs": val_probs,
        "val_mule_probs": np.array(val_mule_probs),
        "val_mule_true": np.array(val_mule_true),
        "val_role_preds": np.array(val_role_preds),
        "val_role_true": np.array(val_role_true),
        "history": history,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "elapsed_seconds": elapsed,
    }


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        # Degenerate split (no positives or no negatives) — cannot compute AUC.
        return float("nan")
    return float(roc_auc_score(y_true, y_score))

def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    cfg = parse_args(argv)

    try:
        if cfg.synthetic_smoke_test:
            log.warning(
                "Running in --synthetic-smoke-test mode. This validates the pipeline "
                "only — do NOT report this AUC as the Sprint 4B gate metric."
            )
            node_rows, edge_rows = build_synthetic_graph(seed=cfg.seed)
        else:
            node_rows, edge_rows = fetch_graph_from_neo4j(cfg)
    except RuntimeError as exc:
        log.error("Failed to build graph: %s", exc)
        return 2

    data, account_ids = build_pyg_data(node_rows, edge_rows, cfg)

    if data.y.sum() < 2:
        log.error(
            "Fewer than 2 positive (is_laundering) accounts in the graph — "
            "cannot train or evaluate meaningfully. Check the Step 4A label pipeline."
        )
        return 1

    result = train(cfg, data)
    best_val_auc = result["best_val_auc"]
    best_val_pr_auc = result["best_val_pr_auc"]

    cfg.model_out.parent.mkdir(parents=True, exist_ok=True)
    # Re-save as pure state dict for final usage if we want, or just leave the checkpoint format
    # The checkpoint format is fine since torch.load handles it if inference code checks for keys.
    # We will just leave the latest checkpoint as-is, which is already saved during train().
    
    gate_passed = best_val_auc >= AUC_TARGET
    
    # Calculate Multi-Task metrics
    from sklearn.metrics import average_precision_score, accuracy_score
    mule_roc_auc = _safe_auc(result["val_mule_true"], result["val_mule_probs"])
    mule_pr_auc = _safe_pr_auc(result["val_mule_true"], result["val_mule_probs"])
    role_acc = float(accuracy_score(result["val_role_true"], result["val_role_preds"]))
    
    log.info("═══════════════════════════════════════════════════════")
    log.info("v4 GNN MULTI-TASK RESULTS (Validation split)")
    log.info("═══════════════════════════════════════════════════════")
    log.info("  Role Accuracy (vs Pseudo-Labels): %.4f", role_acc)
    log.info("  Mule ROC-AUC: %.4f", mule_roc_auc)
    log.info("  Mule PR-AUC:  %.4f", mule_pr_auc)
    log.info("═══════════════════════════════════════════════════════")

    metrics = {
        "static_gnn_auc": best_val_auc,
        "static_gnn_pr_auc": best_val_pr_auc,
        "static_gnn_mule_roc_auc": mule_roc_auc,
        "static_gnn_mule_pr_auc": mule_pr_auc,
        "static_gnn_role_accuracy": role_acc,
        "auc_target": AUC_TARGET,
        "gate_passed": gate_passed,
        "n_nodes": data.num_nodes,
        "n_edges": data.num_edges,
        "n_features": int(data.x.shape[1]),
        "feature_names": FEATURE_NAMES,
        "positive_rate": float(data.y.mean().item()),
        "mule_positive_rate": float(data.y_mule.mean().item()),
        "val_fraction": cfg.val_fraction,
        "split_type": "random_80_20",
        "hidden_dim": cfg.hidden_dim,
        "epochs_run": len(result["history"]),
        "epochs_requested": cfg.epochs,
        "training_seconds": result["elapsed_seconds"],
        "synthetic_smoke_test": cfg.synthetic_smoke_test,
        "note": (
            "This is the baseline the Sprint 5 TGN must beat by >= 1 percentage point. "
            "IMPORTANT: static_gnn_auc/static_gnn_pr_auc are computed on the VALIDATION "
            "split (the same 20% used for early-stopping model selection), not an "
            "independent held-out test set \u2014 this baseline's spec is a single 80/20 "
            "random split with no third partition."
        ),
        "fixes_applied": [
            "fan_out_24h now uses a dataset-relative 24h window (max SENT.timestamp - 1 day) "
            "instead of wall-clock datetime.now(), which previously zeroed this feature out entirely.",
            "Feature standardization now fit on train-split-only statistics (no validation leakage).",
            "community_id is now a learned categorical embedding rather than a z-scored continuous value.",
            "amount_paid used for Neo4j edge amounts instead of amount_received.",
        ],
    }

    cfg.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    roc_out = Path("ml/results/static_gnn_roc_curve.png")
    roc_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fpr, tpr, _ = roc_curve(result["val_true"], result["val_probs"])
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (area = {best_val_auc:.4f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Static GNN Baseline ROC')
        plt.legend(loc="lower right")
        plt.savefig(roc_out)
        plt.close()
    except Exception as e:
        log.warning("Could not plot ROC curve: %s", e)

    # BUG FIX (#4): this is the VALIDATION set (the same 20% used for early-stopping
    # model selection), not an independent held-out test set.
    log.info("Static GNN Baseline \u2014 Final Validation Metrics (NOT an independent held-out test set)")
    log.info("  ROC-AUC : %.4f", best_val_auc)
    log.info("  PR-AUC  : %.4f", best_val_pr_auc)
    log.info("Baseline AUC record saved: %s", cfg.metrics_out)
    log.info("ROC curve saved:           %s", roc_out)

    status = "PASS" if gate_passed else "FAIL"
    log.info(
        "[%s] Static GNN AUC = %.4f (target >= %.2f)",
        status,
        best_val_auc,
        AUC_TARGET,
    )

    if cfg.synthetic_smoke_test:
        # Smoke test success is about the pipeline running end-to-end, not the AUC gate.
        return 0

    return 0 if gate_passed else 3


if __name__ == "__main__":
    sys.exit(main())
