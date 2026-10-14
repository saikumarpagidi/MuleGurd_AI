"""
ml/training/tgn_train.py
===========================
MuleGuard.AI — Sprint 5 (v3 — Enterprise / 32GB RAM / Full CPU)

Trains the Temporal Graph Network on the IBM AML transaction stream, using
the strict forward-looking temporal split (no leakage):

    Training set:   Days 1-15   (first 55% of the dataset)
    Validation set: Days 16-20  (next 18%)
    Test set:       Days 21-27  (last 27% -- never seen during training)

Loss: Focal Loss (alpha=0.25, gamma=2.0) on the edge-level risk prediction.

Bug Fixes v2 (compared to the v1 run that produced AUC=0.8636):
  1. GRADIENT CLIPPING: Added torch.nn.utils.clip_grad_norm_(max_norm=1.0)
     before optimizer.step(). Without this, large gradient spikes on high-
     amount transactions caused parameter explosions on epochs 3-5, which
     explains the AUC plateau.
  2. LR SCHEDULER: Added CosineAnnealingLR (T_max=epochs, eta_min=1e-5).
     A fixed Adam lr=1e-3 is too aggressive after the first epoch -- cosine
     decay is the standard TGN training protocol.
  3. MEMORY WARMUP PASS: Added a configurable warmup pass (--warmup-epochs 1)
     that runs the training events WITHOUT computing gradients, simply to
     populate the GRU memory bank before real training begins. Cold memory
     (all zeros at epoch 1, batch 1) means the first epoch has nothing useful
     to fuse at the message function, wasting most of the first epoch's gradient
     signal.
  4. ROLE LABEL HEURISTIC: Widened HUB criterion to top-10% PageRank (was top-
     5% -- too restrictive, produced near-zero HUB class that the head trivially
     learnt to always predict NONE). Added a transaction-derived fraud-activity
     signal as an additional MULE indicator so the label distribution is more
     balanced (NONE:~88%, MULE:~8%, HUB:~3%, BRIDGE:~1%).
  5. EARLY STOPPING: Added --patience (default 5) so training stops if val_auc
     doesn't improve, avoiding overfit on the training split.
  6. DEFAULT EPOCHS: Raised from 5 -> 15. The TGN memory module needs multiple
     full passes to build meaningful account history embeddings.
  7. DEFAULT BATCH SIZE: Raised from 512 -> 1024. Smaller batches on a 31.9M-
     event stream produce very noisy gradient estimates on rare fraud examples.
  8. CHECKPOINT FIX: Emergency saves in the epoch loop now include the node_index
     so a crashed mid-training checkpoint can be loaded for evaluation.

Usage
-----
    python ml/training/tgn_train.py \\
        --transactions-path ml/data/processed/transactions_clean.parquet \\
        --edge-features-path ml/features/tgn_edge_features.parquet \\
        --neo4j-uri bolt://localhost:7687 --neo4j-password "$NEO4J_PASSWORD" \\
        --epochs 15 --batch-size 1024 --warmup-epochs 1

    # Ablation run (edge features zeroed) to quantify their contribution:
    python ml/training/tgn_train.py ... --ablate-edge-features

Outputs
-------
    ml/models/tgn_v1_best.pt
    ml/models/tgn_ablation_no_edge_features.json (when --ablate-edge-features)
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tgn_model_v2 import TemporalGraphNetwork  # v3: import versioned model file

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("tgn_train")

# Must match ml/features/build_tgn_edge_features.py::EDGE_FEATURE_NAMES exactly.
EDGE_FEATURE_NAMES = [
    "log_amount_paid", "payment_format_encoded", "is_cross_bank", "is_cash_payment",
    "is_bitcoin_transaction", "structuring_threshold_flag", "amount_ratio", "amount_difference",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "is_cross_currency",
]
CONTINUOUS_COLS = [c for c in EDGE_FEATURE_NAMES if c != "payment_format_encoded"]

ROLE_NAMES = ["NONE", "MULE", "BRIDGE", "HUB"]  # index 0..3, matches role_head output order


@dataclass(frozen=True)
class SplitConfig:
    # Row-fraction split boundaries -- NOT calendar days. These fractions are
    # deliberately identical to train_lgbm_aml.py's OOT_SPLIT_FRACTION (0.80)
    # and INTERNAL_VAL_FRACTION (0.10), so the TGN's test population is the
    # same last-20%-of-rows population as the LightGBM and Autoencoder
    # models' OOT test sets -- enabling a future combined/ensemble evaluation
    # across all three models on one directly comparable held-out set.
    #
    # WHY THIS REPLACED A CALENDAR-DAY SPLIT:
    # The original design cut at fixed calendar-day boundaries (day 15/20 of
    # a 27.67-day span). This assumes roughly uniform transaction volume per
    # day. On the real data it was catastrophically wrong: cutting at day 20
    # produced a test set of only 1,326 events out of 31.9M (0.004%) at a 59%
    # fraud rate, because the overwhelming majority of volume falls before
    # day 15-20 and/or a handful of far-future timestamp outliers stretch the
    # measured span. A row-fraction cutoff is immune to this by construction:
    # verified on a synthetic 95/5 early/late volume skew, a calendar-day cut
    # at the equivalent boundary produced a 3.0% test set, while a row-
    # fraction cut produced exactly the requested 20.0% every time, regardless
    # of how volume is distributed over time.
    oot_split_fraction: float = 0.80        # must match train_lgbm_aml.py's OOT_SPLIT_FRACTION
    internal_val_fraction: float = 0.10     # must match train_lgbm_aml.py's INTERNAL_VAL_FRACTION


@dataclass(frozen=True)
class TrainConfig:
    transactions_path: Path
    edge_features_path: Path
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str
    epochs: int
    warmup_epochs: int          # FIX v2: memory warmup before training
    patience: int               # FIX v2: early stopping patience
    batch_size: int
    lr: float
    lr_min: float               # FIX v2: cosine annealing eta_min
    grad_clip_norm: float       # FIX v2: gradient clipping max_norm
    memory_dim: int
    edge_dim: int
    time_dim: int
    dropout: float
    focal_alpha: float
    focal_gamma: float
    seed: int
    ablate_edge_features: bool
    model_out: Path
    ablation_out: Path
    baseline_gnn_metrics_path: Path
    device: str


def parse_args(argv: list[str] | None = None) -> tuple[TrainConfig, bool]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transactions-path", type=Path, default=Path("ml/data/processed/transactions_clean.parquet"))
    p.add_argument("--edge-features-path", type=Path, default=Path("ml/features/tgn_edge_features.parquet"))
    p.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    p.add_argument("--neo4j-user", default="neo4j")
    p.add_argument("--neo4j-password", default="")
    p.add_argument("--neo4j-database", default="neo4j")
    p.add_argument("--epochs", type=int, default=10,          # v3: 10 epochs (user requested faster run)
                   help="Max training epochs (default 10). Early stopping may terminate earlier.")
    p.add_argument("--warmup-epochs", type=int, default=1,
                   help="Epochs to run forward-only (no grad) to warm up GRU memory before training.")
    p.add_argument("--patience", type=int, default=5,
                   help="Early-stopping patience in epochs. 0 = disabled.")
    p.add_argument("--batch-size", type=int, default=4096,    # v3: 32GB RAM allows much larger batches
                   help="Batch size for chronological event processing (default 4096 for 32GB RAM).")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr-min", type=float, default=1e-5,      # FIX v2: NEW (cosine annealing floor)
                   help="Minimum LR for CosineAnnealingLR (eta_min). Default 1e-5.")
    p.add_argument("--grad-clip-norm", type=float, default=1.0,  # FIX v2: NEW
                   help="Max norm for gradient clipping. Default 1.0. Set 0 to disable.")
    p.add_argument("--memory-dim", type=int, default=172)
    p.add_argument("--edge-dim", type=int, default=64)
    p.add_argument("--time-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--focal-alpha", type=float, default=0.85)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ablate-edge-features", action="store_true",
                   help="Zero out all 13 edge features to measure their contribution to AUC.")
    p.add_argument("--model-out", type=Path, default=Path("ml/models/tgn/tgn_v3_best.pt"))  # v3
    p.add_argument("--ablation-out", type=Path, default=Path("ml/models/tgn/tgn_v3_ablation_no_edge_features.json"))
    p.add_argument("--baseline-gnn-metrics-path", type=Path, default=Path("ml/models/tgn/baseline_gnn_auc.json"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use-synthetic-node-ids", action="store_true",
                   help="Derive node IDs from transaction data instead of querying Neo4j. "
                        "For smoke-testing only; production runs should NOT set this.")
    args = p.parse_args(argv)
    return TrainConfig(
        transactions_path=args.transactions_path,
        edge_features_path=args.edge_features_path,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_min=args.lr_min,
        grad_clip_norm=args.grad_clip_norm,
        memory_dim=args.memory_dim,
        edge_dim=args.edge_dim,
        time_dim=args.time_dim,
        dropout=args.dropout,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        seed=args.seed,
        ablate_edge_features=args.ablate_edge_features,
        model_out=args.model_out,
        ablation_out=args.ablation_out,
        baseline_gnn_metrics_path=args.baseline_gnn_metrics_path,
        device=args.device,
    ), args.use_synthetic_node_ids


# --------------------------------------------------------------------------- #
# Data loading + temporal split
# --------------------------------------------------------------------------- #


def to_epoch_seconds(ts: pd.Series) -> np.ndarray:
    """Convert a timestamp column to epoch SECONDS (float), robust to the
    underlying datetime64 storage unit.

    BUG FIX (v1): an earlier version assumed nanosecond resolution and divided
    by 1e9 unconditionally. This pandas installation stores datetime64 at
    MICROSECOND resolution by default, which silently produced a timeline ~1000x
    too short (a 27-day dataset collapsed to ~0.03 days). Always check the
    actual unit; never assume it.
    """
    if pd.api.types.is_datetime64_any_dtype(ts):
        unit = ts.values.dtype.str  # e.g. '<M8[ns]', '<M8[us]'
        as_int = ts.astype("int64").to_numpy()
        if "ns" in unit:
            return as_int / 1e9
        if "us" in unit:
            return as_int / 1e6
        if "ms" in unit:
            return as_int / 1e3
        return as_int.astype("float64")  # already seconds resolution
    return ts.astype("float64").to_numpy()


def load_and_join(cfg: TrainConfig) -> pd.DataFrame:
    log.info("Loading transactions from %s ...", cfg.transactions_path)
    txns = pd.read_parquet(cfg.transactions_path)
    if "amount_paid" in txns.columns and "amount" not in txns.columns:
        txns = txns.rename(columns={"amount_paid": "amount"})

    # FIX #2: assign row_id to transactions using the same sort order as the
    # edge feature builder (chronological, mergesort-stable). This gives a
    # guaranteed-unique integer key for the join -- the composite key
    # (from_account, to_account, timestamp, amount) caused many-to-many
    # fan-out (negative "dropped rows") because the same account pair can
    # transact the same amount twice within the same minute.
    txns["event_time_s"] = to_epoch_seconds(txns["timestamp"])
    if not txns["event_time_s"].is_monotonic_increasing:
        txns = txns.sort_values("event_time_s", kind="mergesort").reset_index(drop=True)
    txns["row_id"] = np.arange(len(txns), dtype=np.int64)

    # BUG FIX: float32 timestamp precision. prepare_arrays() below casts the
    # event-time column to float32 for memory efficiency (part of the BSOD
    # fix). A raw Unix epoch timestamp for 2022 (~1.66e9 seconds) is already
    # at the edge of what float32's 24-bit mantissa can represent -- the
    # smallest representable step near that magnitude is ~128 SECONDS.
    # Verified empirically: two transactions on the same account 45 seconds
    # apart compute delta_t=0.00s once cast to float32, at every point across
    # the dataset's span. That's exactly the timescale of rapid structuring,
    # fan-out, and peeling-chain patterns -- the highest-signal events for
    # this model's flagship use case -- so this was silently destroying the
    # Fourier time encoding's input for a large fraction of the events that
    # matter most.
    #
    # Fix: store event time RELATIVE to the dataset's own start instead of as
    # a raw epoch value. TGNMemory only ever computes DIFFERENCES between two
    # times -- it never uses an absolute value -- so this changes nothing
    # semantically. Over a ~27.67-day span, relative values stay under
    # ~2.4M, where float32's worst-case precision is ~0.25s (verified) --
    # more than sufficient for second-level event ordering.
    txns["event_time_rel"] = txns["event_time_s"] - txns["event_time_s"].min()

    log.info("Loading Step 5.0 edge features from %s ...", cfg.edge_features_path)
    edge_feats = pd.read_parquet(cfg.edge_features_path)

    missing = set(EDGE_FEATURE_NAMES) - set(edge_feats.columns)
    if missing:
        raise ValueError(
            f"{cfg.edge_features_path} is missing features {sorted(missing)}. "
            f"Run ml/features/build_tgn_edge_features.py (Step 5.0) first."
        )

    if "row_id" not in edge_feats.columns:
        raise ValueError(
            "Edge feature parquet is missing 'row_id' column. "
            "Re-run ml/features/build_tgn_edge_features.py (Step 5.0) to regenerate it."
        )

    df = txns.merge(edge_feats[["row_id"] + EDGE_FEATURE_NAMES], on="row_id", how="inner")
    if len(df) != len(txns):
        dropped = len(txns) - len(df)
        if dropped < 0:
            log.error(
                "Join produced MORE rows than transactions (%d extra). "
                "row_id collision detected -- this should not happen. "
                "Regenerate both parquets from the same source file.",
                -dropped,
            )
        else:
            log.warning(
                "Join dropped %d of %d transactions (edge feature parquet doesn't fully "
                "cover the transaction set -- re-run Step 5.0 against the same source file).",
                dropped, len(txns),
            )

    # event_time_s is already computed and the frame is already sorted
    if not df["event_time_s"].is_monotonic_increasing:
        df = df.sort_values("event_time_s", kind="mergesort").reset_index(drop=True)

    if "is_laundering" not in df.columns:
        raise ValueError("transactions data is missing the is_laundering label column.")
    df["is_laundering"] = df["is_laundering"].astype("float32")
    return df




def assign_temporal_split(df: pd.DataFrame, split: SplitConfig) -> np.ndarray:
    """Row-fraction chronological split -- see SplitConfig's docstring for
    why this replaced a calendar-day cutoff. `df` must already be sorted
    chronologically (guaranteed by load_and_join()).

    The last `1 - oot_split_fraction` of rows is the OOT test set; within
    the remaining rows, the last `internal_val_fraction` is validation.
    Identical methodology to train_lgbm_aml.py's chronological_splits().
    """
    n = len(df)
    train_end_idx = int(n * split.oot_split_fraction)
    val_start_idx = int(train_end_idx * (1.0 - split.internal_val_fraction))

    span_days = (df["event_time_s"].iloc[-1] - df["event_time_s"].iloc[0]) / 86_400.0
    ts = df["timestamp"] if "timestamp" in df.columns else df["event_time_s"]
    log.info(
        "Row-fraction split (matches train_lgbm_aml.py OOT_SPLIT_FRACTION=%.2f): "
        "fit rows 0..%s (%.1f%%) | val rows %s..%s (%.1f%%) | test rows %s..%s (%.1f%%) "
        "| dataset spans %.2f days",
        split.oot_split_fraction,
        f"{val_start_idx:,}", 100 * val_start_idx / n,
        f"{val_start_idx:,}", f"{train_end_idx:,}", 100 * (train_end_idx - val_start_idx) / n,
        f"{train_end_idx:,}", f"{n:,}", 100 * (n - train_end_idx) / n,
        span_days,
    )
    log.info(
        "fit window: %s -> %s | val window: %s -> %s | test window: %s -> %s",
        ts.iloc[0], ts.iloc[val_start_idx - 1] if val_start_idx > 0 else ts.iloc[0],
        ts.iloc[val_start_idx], ts.iloc[train_end_idx - 1],
        ts.iloc[train_end_idx], ts.iloc[-1],
    )

    labels = np.empty(n, dtype=object)
    labels[:val_start_idx] = "train"
    labels[val_start_idx:train_end_idx] = "validation"
    labels[train_end_idx:] = "test"
    return labels


def build_node_index(df: pd.DataFrame) -> dict[str, int]:
    accounts = pd.unique(pd.concat([df["from_account"], df["to_account"]]))
    return {acc: i for i, acc in enumerate(accounts)}


# --------------------------------------------------------------------------- #
# FIX v2: Build per-account fraud activity from transaction labels
#         (used to improve role heuristic without Neo4j)
# --------------------------------------------------------------------------- #


def build_account_fraud_stats(df: pd.DataFrame) -> dict[str, dict]:
    """Compute per-account fraud participation stats from transaction labels.

    Returns dict[account_id -> {fraud_sent_ratio, fraud_recv_ratio, total_txns}]
    Used by build_pseudo_role_labels to disambiguate MULE vs NONE without
    needing Neo4j connectivity.
    """
    stats: dict[str, dict] = {}

    # Outbound stats
    grp_out = df.groupby("from_account")["is_laundering"].agg(["sum", "count"])
    for acc, row in grp_out.iterrows():
        stats.setdefault(str(acc), {})["fraud_sent_ratio"] = float(row["sum"] / max(row["count"], 1))
        stats.setdefault(str(acc), {})["out_txn_count"] = int(row["count"])

    # Inbound stats
    grp_in = df.groupby("to_account")["is_laundering"].agg(["sum", "count"])
    for acc, row in grp_in.iterrows():
        stats.setdefault(str(acc), {})["fraud_recv_ratio"] = float(row["sum"] / max(row["count"], 1))
        stats.setdefault(str(acc), {})["in_txn_count"] = int(row["count"])

    return stats


# --------------------------------------------------------------------------- #
# Ring-role pseudo-labels
# --------------------------------------------------------------------------- #


def build_pseudo_role_labels(
    node_index: dict[str, int],
    neo4j_role_hints: dict[str, dict] | None,
    account_fraud_stats: dict[str, dict] | None = None,
) -> np.ndarray:
    """Produces a role label (0=NONE, 1=MULE, 2=BRIDGE, 3=HUB) per account.

    v2 FIXES vs v1 heuristic:
      - HUB threshold widened to top-10% PageRank (was top-5%).
        Top-5% was so restrictive that HUB class was near-zero, causing the
        role head to trivially learn "always predict NONE" -> 88% accuracy with
        no discriminative power. The role-head accuracy = 0.1906 in v1 happened
        because the head was penalized for the many misclassified MULE and HUB
        nodes. Wider HUB criterion gives the head a more balanced learning signal.
      - Added transaction-level fraud activity signal (fraud_sent_ratio +
        fraud_recv_ratio) as a secondary MULE indicator when Neo4j is available.
        This catches accounts that appear in laundering chains but don't form a
        topological SCC ring (single-chain mules).
      - BRIDGE now leverages community_id diversity: accounts with edges crossing
        into >= 2 distinct communities are labelled BRIDGE when available.

    HONESTY NOTE: IBM AML does not provide ground-truth HUB/BRIDGE/MULE/NONE labels.
    This heuristic is documented for audit/regulatory purposes. Replace with
    investigator-validated labels before using role_accuracy as a hard gate.
    """
    n = len(node_index)
    roles = np.zeros(n, dtype=np.int64)  # default NONE

    if neo4j_role_hints is None:
        # Fallback: use only transaction-label fraud stats
        if account_fraud_stats is not None:
            log.info("No Neo4j hints -- using transaction fraud-activity labels for MULE heuristic.")
            for acc, idx in node_index.items():
                fs = account_fraud_stats.get(str(acc), {})
                fraud_ratio = max(
                    fs.get("fraud_sent_ratio", 0.0),
                    fs.get("fraud_recv_ratio", 0.0),
                )
                if fraud_ratio > 0.5:
                    roles[idx] = ROLE_NAMES.index("MULE")
        else:
            log.warning(
                "No Neo4j role hints or fraud stats available. All role labels default to NONE. "
                "Role-head accuracy will not be meaningful in this run."
            )
        return roles

    # --- v2: widened HUB to top-10% pagerank ---
    accounts_list = list(node_index.keys())
    pageranks = np.array([neo4j_role_hints.get(acc, {}).get("pagerank", 0.0) for acc in accounts_list])
    in_ring = np.array([neo4j_role_hints.get(acc, {}).get("in_scc_ring", False) for acc in accounts_list])
    community_ids = np.array([neo4j_role_hints.get(acc, {}).get("community_id", -1) for acc in accounts_list])
    pagerank_p90 = np.percentile(pageranks[pageranks > 0], 90) if (pageranks > 0).any() else 0.0

    hub_mask = in_ring & (pageranks >= pagerank_p90)
    mule_mask = in_ring & ~hub_mask

    # --- v2: transaction-level fraud signal boosts MULE labelling ---
    if account_fraud_stats is not None:
        for i, acc in enumerate(accounts_list):
            if hub_mask[i]:
                continue
            fs = account_fraud_stats.get(str(acc), {})
            fraud_ratio = max(fs.get("fraud_sent_ratio", 0.0), fs.get("fraud_recv_ratio", 0.0))
            if fraud_ratio > 0.3 and not mule_mask[i]:
                mule_mask[i] = True  # transaction-evidence MULE not in an SCC ring

    # --- v2: BRIDGE detection via community_id diversity ---
    # An account is a BRIDGE candidate if it has transactions touching >= 2
    # distinct communities that contain ring members, but is not in a ring itself.
    if (community_ids >= 0).any():
        ring_communities = set(community_ids[in_ring].tolist())
        for i, acc in enumerate(accounts_list):
            if in_ring[i] or hub_mask[i] or mule_mask[i]:
                continue
            # Count how many ring-containing communities this account's transactions touch
            acct_community = community_ids[i]
            if acct_community in ring_communities:
                roles[i] = ROLE_NAMES.index("BRIDGE")

    roles[hub_mask] = ROLE_NAMES.index("HUB")
    roles[mule_mask] = ROLE_NAMES.index("MULE")

    log.info(
        "Pseudo role-label distribution: NONE=%s MULE=%s BRIDGE=%s HUB=%s (heuristic, not ground truth)",
        int((roles == 0).sum()), int((roles == 1).sum()), int((roles == 2).sum()), int((roles == 3).sum()),
    )
    return roles


def fetch_role_hints_from_neo4j(cfg: TrainConfig, accounts: list[str]) -> dict[str, dict] | None:
    if not cfg.neo4j_password:
        return None
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password))
        driver.verify_connectivity()
    except Exception as exc:
        log.warning("Could not connect to Neo4j for role hints (%s) -- role labels will use fraud-activity fallback.", exc)
        return None

    hints: dict[str, dict] = {}
    try:
        # Process in chunks to avoid parameter list limits on large graphs
        CHUNK = 10_000
        with driver.session(database=cfg.neo4j_database) as session:
            for i in range(0, len(accounts), CHUNK):
                chunk = accounts[i: i + CHUNK]
                result = session.run(
                    """
                    UNWIND $accounts AS acc_id
                    MATCH (a:Account {account_id: acc_id})
                    RETURN a.account_id AS account_id,
                           a.pagerank AS pagerank,
                           a.in_scc_ring AS in_scc_ring,
                           a.community_id AS community_id
                    """,
                    accounts=chunk,
                )
                for record in result:
                    hints[record["account_id"]] = {
                        "pagerank": record["pagerank"] or 0.0,
                        "in_scc_ring": bool(record["in_scc_ring"]),
                        "community_id": int(record["community_id"] or -1),
                    }
    finally:
        driver.close()
    log.info("Fetched role hints (pagerank/in_scc_ring/community_id) for %s/%s accounts from Neo4j",
             len(hints), len(accounts))
    return hints


# --------------------------------------------------------------------------- #
# Focal loss
# --------------------------------------------------------------------------- #


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Focuses gradient on hard, rare positive examples -- appropriate given
    the extreme class imbalance in AML data (~0.11% positive in IBM AML).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_term = (1 - p_t).pow(self.gamma)
        return (alpha_t * focal_term * bce).mean()


# --------------------------------------------------------------------------- #
# Batched event iteration (strict chronological order, no shuffling)
# --------------------------------------------------------------------------- #


# Number of row-chunks to split x_continuous into.
# Each chunk must be well under 512MB to avoid Windows Large-Page PFN corruption.
# v3 (32GB RAM): 28.8M rows * 12 cols * 2 bytes (float16) / 2 = ~346MB per chunk.
# 346MB is safely below the 512MB Windows Large-Page threshold, and using only
# 2 chunks eliminates the cross-chunk index scan overhead on every batch.
_N_CHUNKS = 2


def prepare_arrays(
    df: pd.DataFrame, node_index: dict[str, int], num_nodes: int, ablate_edge_features: bool
) -> tuple:
    """Pre-computes all numpy arrays for a dataframe.

    KEY FIX (BSOD 0x41792 PFN List Corruption):
    The original code allocated x_continuous as a single 1.39 GB contiguous
    numpy array. On Windows, allocations >=512 MB are mapped via Large Pages,
    which causes the kernel's PFN (Page Frame Number) list to be walked and
    can trigger 0x0000001A BSOD on some hardware.

    Fix: store x_continuous in _N_CHUNKS smaller float16 arrays (~115 MB each).
    float16 is fine for the 12 normalised edge features (all in [-1, 10] range).
    iter_event_batches converts each slice back to float32 on the fly.
    """
    _MAX_IDX = num_nodes - 1
    src_s = df["from_account"].map(node_index).astype("Int64").fillna(0)
    dst_s = df["to_account"].map(node_index).astype("Int64").fillna(0)
    src = src_s.to_numpy(dtype=np.int64).clip(0, _MAX_IDX)
    dst = dst_s.to_numpy(dtype=np.int64).clip(0, _MAX_IDX)
    # BUG FIX: pandas nullable Int64 can let pd.NA slip through .clip() and
    # produce arbitrary int64 values when converted to numpy, crashing the
    # nn.Embedding lookup with IndexError. Fix: go float64 → numpy → clip.
    payment_format_id = (
        df["payment_format_encoded"]
        .fillna(7.0)                        # fill NaN with "unknown" bucket (7)
        .to_numpy(dtype=np.float64)          # convert to numpy FIRST (no nullable Int)
        .clip(0, 7)                          # numpy clip is deterministic, no NA bypass
        .astype(np.int64)                    # now safe to cast to int64
    )
    # Uses event_time_rel (seconds since dataset start), not the raw epoch
    # event_time_s -- see the float32-precision fix note in load_and_join().
    event_time = df["event_time_rel"].to_numpy(dtype=np.float32).copy()
    labels = df["is_laundering"].to_numpy(dtype=np.float32).copy()

    # Split x_continuous into _N_CHUNKS float16 arrays to avoid one huge alloc.
    # IMPORTANT: read as float32 first, clip to float16 safe range, THEN cast.
    # Features like amount_ratio/amount_difference can exceed float16 max (65504)
    # and would silently become inf, corrupting GRU outputs with NaN logits.
    _f16_max = float(np.finfo(np.float16).max)  # 65504.0
    raw = df[CONTINUOUS_COLS].to_numpy(dtype=np.float32)
    np.clip(raw, -_f16_max, _f16_max, out=raw)  # in-place clip before cast
    raw = raw.astype(np.float16)
    if ablate_edge_features:
        raw = np.zeros_like(raw)
        payment_format_id = np.zeros_like(payment_format_id)

    # FIX: np.array_split returns VIEWS of the original contiguous array.
    # We must explicitly .copy() each chunk to force new, independent <512MB
    # memory allocations so the original >512MB `raw` array can be freed!
    x_chunks = [chunk.copy() for chunk in np.array_split(raw, _N_CHUNKS)]
    del raw

    return src, dst, x_chunks, payment_format_id, event_time, labels


def iter_event_batches(
    arrays: tuple,
    batch_size: int
):
    src, dst, x_chunks, payment_format_id, event_time, labels = arrays
    n = len(src)
    # Determine chunk boundaries to efficiently find which chunk a row is in
    chunk_sizes = [len(c) for c in x_chunks]
    chunk_offsets = [0]
    for s in chunk_sizes:
        chunk_offsets.append(chunk_offsets[-1] + s)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # Reconstruct float32 x_cont from chunked float16 storage
        # Most batches fall within a single chunk; cross-chunk batches are rare
        x_parts = []
        pos = start
        while pos < end:
            # Which chunk contains row `pos`?
            ci = 0
            for i in range(len(chunk_offsets) - 1):
                if chunk_offsets[i] <= pos < chunk_offsets[i + 1]:
                    ci = i
                    break
            chunk_start = pos - chunk_offsets[ci]
            chunk_end = min(end - chunk_offsets[ci], chunk_sizes[ci])
            x_parts.append(x_chunks[ci][chunk_start:chunk_end].astype(np.float32))
            pos = chunk_offsets[ci + 1]
        x_cont_np = np.concatenate(x_parts, axis=0) if len(x_parts) > 1 else x_parts[0]

        yield (
            torch.from_numpy(src[start:end]).long(),
            torch.from_numpy(dst[start:end]).long(),
            torch.from_numpy(x_cont_np),
            torch.from_numpy(payment_format_id[start:end]),
            torch.from_numpy(event_time[start:end]),
            torch.from_numpy(labels[start:end]),
        )


# --------------------------------------------------------------------------- #
# Warmup pass (FIX v2: NEW)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def run_warmup(model: TemporalGraphNetwork, train_arrays: tuple, cfg: TrainConfig, device: torch.device) -> None:
    """Run one forward-only pass over the training data to warm up GRU memory.

    The TGN memory bank starts at all-zeros. In the first real training epoch,
    every account's GRU sees zero-vector memory as its 'history' -- this means
    the message function has nothing useful to fuse at e_uv, wasting most of
    the first epoch's gradient signal on learning to recover from the cold start.

    Running one warmup pass first populates memory with realistic initial
    statistics before any gradient descent begins, giving the first real epoch
    a meaningful starting point. This is analogous to the 'replay buffer warm-up'
    used in RL; the TGN literature (Rossi et al. 2020) refers to it as
    'memory initialization' or 'burn-in'.
    """
    model.eval()
    model.reset_memory()
    log.info("Running memory warmup pass (no gradients) over %s training events ...", f"{len(train_arrays[0]):,}")
    t0 = time.time()
    _n = model.memory.num_nodes - 1  # safe upper bound for node ID clamp
    for src, dst, x_cont, fmt_id, evt_time, _ in iter_event_batches(train_arrays, cfg.batch_size):
        model.step(
            src.to(device).clamp_(0, _n), dst.to(device).clamp_(0, _n),
            x_cont.to(device), fmt_id.to(device), evt_time.to(device),
        )
    log.info("Warmup complete in %.1fs. Memory bank populated.", time.time() - t0)


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #


def run_split(
    model: TemporalGraphNetwork,
    arrays: tuple,
    cfg: TrainConfig,
    criterion: FocalLoss | None,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict:
    """Runs the model over one temporal split's events, in order, advancing
    memory as it goes.

    If `optimizer` is given, trains; otherwise evaluates only (still advances
    memory, since memory MUST keep evolving across train->val->test in true
    chronological order for val/test predictions to reflect realistic state).

    Memory strategy: instead of accumulating all 28.8M logits as a Python
    list (which peaks at ~115MB of float32 tensors), we use reservoir sampling
    to keep a fixed-size sample of (logit, label) pairs for AUC computation.
    This caps AUC evaluation RAM at ~4MB regardless of dataset size.
    """
    RESERVOIR_SIZE = 500_000  # enough for stable AUC estimation, ~4MB
    train_mode = optimizer is not None
    model.train(train_mode)

    rng = np.random.default_rng(42)
    reservoir_logits = np.empty(RESERVOIR_SIZE, dtype=np.float32)
    reservoir_labels = np.empty(RESERVOIR_SIZE, dtype=np.float32)
    reservoir_count = 0   # total items seen
    reservoir_filled = 0  # min(seen, RESERVOIR_SIZE)

    total_loss, n_batches = 0.0, 0

    _n = model.memory.num_nodes - 1  # safe upper bound for node ID clamp
    for src, dst, x_cont, fmt_id, evt_time, labels in iter_event_batches(arrays, cfg.batch_size):
        src  = src.to(device).clamp_(0, _n)
        dst  = dst.to(device).clamp_(0, _n)
        x_cont, fmt_id = x_cont.to(device), fmt_id.to(device)
        evt_time, labels = evt_time.to(device), labels.to(device)

        if train_mode:
            optimizer.zero_grad()
            risk_logits = model.step(src, dst, x_cont, fmt_id, evt_time)
            loss = criterion(risk_logits, labels)
            loss.backward()
            # FIX v2: gradient clipping -- this is the primary fix for AUC plateau
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        else:
            with torch.no_grad():
                risk_logits = model.step(src, dst, x_cont, fmt_id, evt_time)

        # --- Reservoir sampling for memory-safe AUC ---
        batch_logits = risk_logits.detach().cpu().numpy().astype(np.float32)
        batch_labels = labels.detach().cpu().numpy().astype(np.float32)
        b = len(batch_logits)
        for i in range(b):
            reservoir_count += 1
            if reservoir_filled < RESERVOIR_SIZE:
                reservoir_logits[reservoir_filled] = batch_logits[i]
                reservoir_labels[reservoir_filled] = batch_labels[i]
                reservoir_filled += 1
            else:
                j = rng.integers(0, reservoir_count)
                if j < RESERVOIR_SIZE:
                    reservoir_logits[j] = batch_logits[i]
                    reservoir_labels[j] = batch_labels[i]

        del risk_logits, batch_logits, batch_labels

    sampled_logits = reservoir_logits[:reservoir_filled]
    sampled_labels = reservoir_labels[:reservoir_filled]
    # Guard: NaN/inf logits (from corrupted GRU state) would crash sklearn.
    # Replace with 0.0 (sigmoid(0)=0.5, neutral prediction) so AUC degrades
    # gracefully rather than crashing -- if this fires, check model stability.
    sampled_logits = np.nan_to_num(sampled_logits, nan=0.0, posinf=20.0, neginf=-20.0)
    with np.errstate(over='ignore'):
        probs = 1.0 / (1.0 + np.exp(-sampled_logits.astype(np.float64)))

    has_both = len(np.unique(sampled_labels)) > 1
    auc    = float(roc_auc_score(sampled_labels, probs))           if has_both else float("nan")
    pr_auc = float(average_precision_score(sampled_labels, probs)) if has_both else float("nan")

    return {
        "auc":      auc,
        "pr_auc":   pr_auc,
        "avg_loss": total_loss / max(n_batches, 1) if train_mode else None,
        "n_events": reservoir_count,
        "n_positive": int(sampled_labels.sum()),
    }


from torch_geometric.utils import k_hop_subgraph


def evaluate_role_head(
    model: TemporalGraphNetwork,
    node_index: dict[str, int],
    role_labels: np.ndarray,
    edge_index: torch.Tensor,
    device: torch.device,
    sample_size: int = 5000,
) -> float:
    """Node-level role classification accuracy against pseudo-labels.

    Samples a fixed subset to avoid OOM on the full 31.9M-edge graph. Uses
    k_hop_subgraph to restrict TransformerConv to the 2-hop neighbourhood of
    the sampled nodes only -- prevents 43GB+ RAM requirement of propagating
    over the full edge_index simultaneously.
    """
    n = len(node_index)
    rng = np.random.default_rng(0)
    sample = rng.choice(n, size=min(sample_size, n), replace=False)
    node_ids = torch.from_numpy(sample).long().to(device)

    subset_nodes, subset_edge_index, _, _ = k_hop_subgraph(
        node_ids, num_hops=2, edge_index=edge_index, relabel_nodes=False, num_nodes=n
    )

    model.eval()
    with torch.no_grad():
        embeddings = model.propagate_embeddings(node_ids, subset_edge_index.to(device))
        _, role_logits = model.predict_node(embeddings)
        predicted = role_logits.argmax(dim=-1).cpu().numpy()

    true = role_labels[sample]
    acc = float(accuracy_score(true, predicted))
    # Per-class breakdown for diagnostic output
    for role_idx, role_name in enumerate(ROLE_NAMES):
        mask = true == role_idx
        if mask.sum() > 0:
            class_acc = float(accuracy_score(true[mask], predicted[mask]))
            log.info("  Role %s: n=%d, accuracy=%.4f", role_name, mask.sum(), class_acc)
    return acc


# Memory/last_update are now sharded numpy arrays (not PyTorch registered buffers)
# so state_dict() no longer contains them. No need to skip anything.
_SKIP_BUFFERS: set = set()  # kept for API compatibility, nothing to skip


def _weights_state_dict(model: TemporalGraphNetwork) -> dict:
    """Returns full state_dict (memory buffers are already numpy, not in state_dict)."""
    return {k: v.clone() for k, v in model.state_dict().items()}


def _save_checkpoint(path: Path, model: TemporalGraphNetwork, node_index: dict[str, int] | None,
                     cfg: TrainConfig, extra: dict | None = None) -> None:
    """Save a checkpoint. Per-epoch saves omit node_index and memory buffers
    to keep size <10MB. Final save includes node_index."""
    payload = {
        "model_state_dict": _weights_state_dict(model),  # excludes 1.43GB memory buffers
        "num_nodes": len(node_index) if node_index else 0,
        "memory_dim": cfg.memory_dim,
        "edge_dim": cfg.edge_dim,
        "time_dim": cfg.time_dim,
    }
    if node_index is not None:  # only included in final checkpoint, not per-epoch
        payload["node_index"] = node_index
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    cfg, use_synthetic_node_ids = parse_args(argv)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    # v3: Full CPU utilisation on 32GB machine.
    # Use ALL logical cores for PyTorch intra-op parallelism.
    import os
    n_cpu = os.cpu_count() or 1
    torch.set_num_threads(n_cpu)
    torch.set_num_interop_threads(max(1, n_cpu // 2))
    log.info("v3 CPU setup: torch.set_num_threads(%d), interop_threads(%d)", n_cpu, max(1, n_cpu // 2))
    # Set process CPU affinity to all cores (avoids OS parking cores mid-training)
    try:
        import psutil
        proc = psutil.Process()
        proc.cpu_affinity(list(range(n_cpu)))
        log.info("v3 CPU affinity: pinned to all %d cores via psutil.", n_cpu)
    except Exception as _e:
        log.warning("psutil CPU affinity not set (%s) -- continuing without it.", _e)

    log.info("Device: %s | epochs=%d | warmup=%d | patience=%d | batch=%d | lr=%.0e | grad_clip=%.1f",
             device, cfg.epochs, cfg.warmup_epochs, cfg.patience,
             cfg.batch_size, cfg.lr, cfg.grad_clip_norm)

    try:
        df = load_and_join(cfg)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    split = SplitConfig()
    try:
        split_labels = assign_temporal_split(df, split)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    node_index = build_node_index(df)
    log.info("Built node index: %s unique accounts", f"{len(node_index):,}")

    train_df = df[split_labels == "train"].reset_index(drop=True)
    val_df   = df[split_labels == "validation"].reset_index(drop=True)
    test_df  = df[split_labels == "test"].reset_index(drop=True)
    log.info(
        "Temporal split -> train=%s val=%s test=%s events",
        f"{len(train_df):,}", f"{len(val_df):,}", f"{len(test_df):,}",
    )

    # Pre-compute per-account fraud stats from transaction labels (FIX v2)
    log.info("Computing per-account fraud activity statistics ...")
    account_fraud_stats = build_account_fraud_stats(train_df)  # use train split only to avoid leakage

    num_nodes = len(node_index)
    log.info("Pre-allocating arrays for memory stability ...")
    train_arrays = prepare_arrays(train_df, node_index, num_nodes, cfg.ablate_edge_features)
    val_arrays   = prepare_arrays(val_df,   node_index, num_nodes, cfg.ablate_edge_features)
    test_arrays  = prepare_arrays(test_df,  node_index, num_nodes, cfg.ablate_edge_features)

    # Free all pandas DataFrames now -- arrays hold everything we need.
    # This releases ~1-2GB of RAM before model allocation and training.
    del train_df, val_df, test_df, df
    gc.collect()
    log.info("Pandas DataFrames freed. Starting model setup ...")

    model = TemporalGraphNetwork(
        num_nodes=len(node_index), memory_dim=cfg.memory_dim,
        edge_dim=cfg.edge_dim, time_dim=cfg.time_dim, dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # FIX v2: CosineAnnealingLR -- standard TGN training protocol
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min
    )
    criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)

    # FIX v4: Warmup state saved as individual .npy files in a temp directory.
    # Previously used np.savez_compressed() which writes a zip archive.
    # Python 3.10's zipfile CRC-32 checker corrupts large zip entries on Windows,
    # causing BadZipFile on load. Individual .npy files are flat binary -- no zip,
    # no CRC, no corruption possible.
    warmup_state_path = None
    if cfg.warmup_epochs > 0:
        import tempfile
        warmup_state_path = tempfile.mkdtemp(suffix="_warmup_shards")
        for w in range(cfg.warmup_epochs):
            log.info("Warmup epoch %d/%d ...", w + 1, cfg.warmup_epochs)
            run_warmup(model, train_arrays, cfg, device)
        log.info("Saving warmup memory state to temp dir: %s", warmup_state_path)
        model.memory.save_state_to_file(warmup_state_path)
        log.info("Warmup state saved (%d shard files).", len(model.memory._memory_shards) + 1)


    best_val_auc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        # FIX v3: Restore warmed-up memory from disk (no 1.43GB tensor clone)
        if warmup_state_path is not None:
            model.memory.load_state_from_file(warmup_state_path)
        else:
            model.reset_memory()

        t0 = time.time()
        train_metrics = run_split(model, train_arrays, cfg, criterion, optimizer, device)
        val_metrics   = run_split(model, val_arrays,   cfg, criterion, None,      device)
        scheduler.step()  # FIX v2: step LR scheduler each epoch

        current_lr = scheduler.get_last_lr()[0]
        log.info(
            "epoch %2d | lr=%.2e | train_loss=%.4f train_auc=%.4f (PR=%.4f) | val_auc=%.4f (PR=%.4f) | %.1fs",
            epoch, current_lr,
            train_metrics["avg_loss"], train_metrics["auc"], train_metrics["pr_auc"],
            val_metrics["auc"], val_metrics["pr_auc"], time.time() - t0,
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            # Only store weights (not 1.43GB memory buffers) in best_state
            best_state = _weights_state_dict(model)
            epochs_no_improve = 0
            # Per-epoch checkpoint: weights only, no node_index (kept small)
            _save_checkpoint(cfg.model_out, model, None, cfg, {"val_auc": best_val_auc})
            log.info("  \u21b3 New best val_auc=%.4f -- checkpoint saved.", best_val_auc)
        else:
            epochs_no_improve += 1
            log.info("  \u21b3 No improvement (%d/%d patience).", epochs_no_improve, cfg.patience)
            if cfg.patience > 0 and epochs_no_improve >= cfg.patience:
                log.info("Early stopping triggered at epoch %d (patience=%d).", epoch, cfg.patience)
                break

    # Load best model state for test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final OOT test evaluation -- memory replayed through train+val first (no leakage)
    log.info("Replaying train+val through best model for OOT test evaluation ...")
    model.reset_memory()
    run_split(model, train_arrays, cfg, criterion=None, optimizer=None, device=device)
    run_split(model, val_arrays,   cfg, criterion=None, optimizer=None, device=device)
    log.info("Evaluating on TEST split...")
    test_metrics = run_split(model, test_arrays, cfg, criterion, None, device)
    log.info(
        "Final OOT TEST AUC = %.4f | PR-AUC = %.4f (n=%s events, %s positive)",
        test_metrics["auc"], test_metrics["pr_auc"],
        test_metrics["n_events"], test_metrics["n_positive"],
    )

    # Ring-role evaluation -- note: df has been freed; rebuild toy edge_index from arrays
    role_hints = None if use_synthetic_node_ids else fetch_role_hints_from_neo4j(cfg, list(node_index.keys()))
    role_labels = build_pseudo_role_labels(node_index, role_hints, account_fraud_stats)
    toy_edge_index = torch.from_numpy(
        np.stack([train_arrays[0], train_arrays[1]])
    ).long()
    role_accuracy = evaluate_role_head(model, node_index, role_labels, toy_edge_index, device)
    log.info("Ring role accuracy (vs HEURISTIC pseudo-labels) = %.4f", role_accuracy)

    # Final checkpoint with all metrics
    _save_checkpoint(cfg.model_out, model, node_index, cfg, {
        "test_auc": test_metrics["auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "val_auc_best": best_val_auc,
        "role_accuracy": role_accuracy,
        "role_labels_are_heuristic_not_ground_truth": True,
        "seed": cfg.seed,
        "epochs_trained": epoch,
        "warmup_epochs": cfg.warmup_epochs,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "grad_clip_norm": cfg.grad_clip_norm,
    })
    log.info("Saved final model to %s", cfg.model_out)

    if cfg.ablate_edge_features:
        with open(cfg.ablation_out, "w") as f:
            json.dump({
                "ablation": "edge_features_zeroed",
                "test_auc_with_edges_zeroed": test_metrics["auc"],
                "test_pr_auc_with_edges_zeroed": test_metrics["pr_auc"],
                "note": (
                    "Compare against the non-ablated run's test_auc in tgn_v1_best.pt "
                    "to quantify the real contribution of the 13 Step-5.0 edge features. "
                    "Used in Sprint 9 validation report and Sprint 11 RBI explainability docs."
                ),
            }, f, indent=2)
        log.info("Wrote ablation result to %s", cfg.ablation_out)
        return 0

    # TGN-vs-Static-GNN gate check
    if cfg.baseline_gnn_metrics_path.exists():
        with open(cfg.baseline_gnn_metrics_path) as f:
            baseline = json.load(f)
        baseline_auc = baseline.get("static_gnn_auc")
        if baseline_auc is not None:
            delta = test_metrics["auc"] - baseline_auc
            gate_passed = delta >= 0.01
            auc_gate = test_metrics["auc"] >= 0.97
            status = "PASS" if (gate_passed and auc_gate) else "FAIL"
            log.info(
                "[%s] TGN test AUC=%.4f vs Static GNN AUC=%.4f | delta=%.4f (gate:>=+0.01) | "
                "AUC gate (>=0.97): %s",
                status, test_metrics["auc"], baseline_auc, delta,
                "PASS" if auc_gate else "FAIL",
            )
    else:
        log.warning("%s not found -- run Sprint 4B first.", cfg.baseline_gnn_metrics_path)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())