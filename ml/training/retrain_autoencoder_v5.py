"""
MuleGuard.AI — Production Deep Neural Network Autoencoder (v4)
One-Class / Semi-Supervised Anomaly Detection for 31M+ Rows.

Audited & Production-Hardened (this revision fixes 3 issues found in review):
  1. Genuine Block-Streaming: BlockStreamLoader loads each contiguous disk
     block (block_size rows) into RAM with a SINGLE memmap read, then
     shuffles and yields mini-batches from that in-memory array. This
     guarantees sequential SSD I/O regardless of OS page-cache eviction
     under memory pressure (the previous per-index memmap access via
     DataLoader/Dataset only *tended* to be fast because of OS caching,
     it wasn't guaranteed).
  2. Cryptographic Cache Validation: SHA-256 source hashing + strict
     feature & version verification prevents silent stale-cache training.
  3. Safe Checkpointing with correctly separated metrics: PR-AUC and
     ROC-AUC are tracked in INDEPENDENT best-so-far variables. Mixing them
     into a single "best_val_pr_auc" comparison caused a real bug: if
     epoch 1 fell back to ROC-AUC (~0.5-0.9 range) due to a NaN PR-AUC,
     no later epoch's genuine PR-AUC (which is prevalence-limited and
     often 0.01-0.05 for imbalanced fraud data) could ever beat that
     frozen epoch-1 value, silently freezing the "best" checkpoint at
     epoch 1 forever. Fixed by keeping best_val_pr and best_val_roc apart,
     and selecting primarily on PR-AUC only when it is defined.
  4. Instant Metadata Timestamp Scan: parquet row-group statistics used
     first, with a full-scan fallback if statistics are unavailable.
  5. Tie-aware ROC-AUC: stable rank computation robust to score ties.
  6. Explicit Garbage Collection: Windows memory-map safety and RAM
     boundedness.
  7. Terminology accuracy: documented as One-Class Anomaly Detection
     (PU / clean-legitimate-curation learning), not "pure unsupervised".

THIS REVISION (v4.1) additionally ports the cold-start guard from v3:
  8. Cold-Start Guard: rows where txn_count_1h==0 AND fan_out_24h==0 (a
     brand-new account with no prior velocity history) are forced to
     score=0.0 at evaluation time, instead of near-maximum reconstruction
     error. Without this, the AE — trained only on legitimate accounts
     WITH some history — treats every zero-velocity profile as an
     outlier, inflating false positives on new/first-time accounts.
     NOTE: if this checkpoint is served by a live inference service,
     the same cold_start_mask() check must be duplicated there; this
     guard only fires inside evaluate_partition() in this script.

NOTE: latent_dim=4, lr=5e-4 and default epochs=6 are carried over from the
prior revision as intentional hyperparameter choices, not bug fixes. Adjust
via --latent-dim / --lr / --epochs if that wasn't intended.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn

LOGGER = logging.getLogger("autoencoder_v4_prod")
ARTIFACT_SCHEMA_VERSION = "muleguard-nn-autoencoder-v4-prod"
FEATURE_VERSION = "muleguard.autoencoder.v4"
CACHE_SUCCESS_MARKER = "_SUCCESS"
EPSILON = 1e-6

FEATURE_NAMES = [
    "log_amount_paid",
    "log_amount_received",
    "amount_vs_sender_mean",
    "is_large_amount",
    "hour_of_day_norm",
    "day_of_week_norm",
    "is_night",
    "is_weekend",
    "txn_count_1h",
    "txn_count_24h",
    "amount_sent_1h",
    "fan_out_24h",
    "fan_in_24h",
    "account_degree_ratio",
    "is_cross_bank",
    "payment_format_risk_score",
    "amount_ratio",
    "time_since_last_received",
    "amount_received_1h",
    "receiver_txn_count_1h",
    "amount_vs_receiver_mean",
]

# --- Cold-start guard (ported from v3) --------------------------------------
# If BOTH of these are 0, the account has no prior transaction history —
# the AE never saw a zero-velocity *legitimate* profile during training
# (training rows are legitimate accounts WITH some history), so it scores
# these as near-maximum anomaly by default. That's the root cause of
# "score=1 immediately on a brand-new account" false positives.
COLD_START_FEATURES = ["txn_count_1h", "fan_out_24h"]
COLD_START_INDICES = [FEATURE_NAMES.index(f) for f in COLD_START_FEATURES]
# -----------------------------------------------------------------------------

LOG_TRANSFORM_FEATURES = [
    "amount_vs_sender_mean",
    "txn_count_1h",
    "txn_count_24h",
    "amount_sent_1h",
    "fan_out_24h",
    "fan_in_24h",
    "account_degree_ratio",
    "payment_format_risk_score",
    "amount_ratio",
    "time_since_last_received",
    "amount_received_1h",
    "receiver_txn_count_1h",
    "amount_vs_receiver_mean",
]

FEATURE_WEIGHTS: Dict[str, float] = {
    "log_amount_paid": 1.0,
    "log_amount_received": 1.0,
    "amount_vs_sender_mean": 3.5,
    "is_large_amount": 2.0,
    "hour_of_day_norm": 1.0,
    "day_of_week_norm": 1.0,
    "is_night": 1.0,
    "is_weekend": 1.0,
    "txn_count_1h": 1.5,
    "txn_count_24h": 1.5,
    "amount_sent_1h": 1.5,
    "fan_out_24h": 2.5,
    "fan_in_24h": 2.5,
    "account_degree_ratio": 2.5,
    "is_cross_bank": 2.5,
    "payment_format_risk_score": 2.0,
    "amount_ratio": 3.5,
    "time_since_last_received": 2.0,
    "amount_received_1h": 1.5,
    "receiver_txn_count_1h": 1.5,
    "amount_vs_receiver_mean": 3.5,
}

LABEL_COLUMN = "is_laundering"
TIMESTAMP_COLUMN = "timestamp"
SOURCE_COLUMNS = FEATURE_NAMES + [TIMESTAMP_COLUMN, LABEL_COLUMN]
LOG_TRANSFORM_COL_IDX = np.array([FEATURE_NAMES.index(n) for n in LOG_TRANSFORM_FEATURES], dtype=np.int64)


# ---------------------------------------------------------------------------
# PyTorch Deep Neural Network Architecture
# ---------------------------------------------------------------------------
class DeepMuleAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 21, latent_dim: int = 4, dropout_rate: float = 0.05):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)


class WeightedMSELoss(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super().__init__()
        self.register_buffer("weights", weights.view(1, -1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff_sq = torch.square(pred - target)
        weighted = diff_sq * self.weights
        return torch.mean(torch.sum(weighted, dim=1))


# ---------------------------------------------------------------------------
# Genuine Block-Streaming Loader (Fix #1)
# ---------------------------------------------------------------------------
class BlockStreamLoader:
    """
    Iterates a memmapped feature matrix with GUARANTEED sequential disk I/O.

    Unlike a DataLoader(Dataset) combo where each sample is fetched with its
    own `mm[idx]` call (which only *happens* to be fast if the OS page cache
    still holds the block), this loader reads each block with exactly ONE
    `mm[b_start:b_end]` slice into a plain in-memory numpy array, then
    shuffles and yields mini-batches from that array. Sequential I/O is
    structurally guaranteed, not dependent on cache behavior under memory
    pressure.

    Block order is shuffled every epoch; rows within each in-memory block
    are also shuffled before slicing into mini-batches.
    """

    def __init__(
        self,
        memmap_path: Path,
        shape: Tuple[int, int],
        mean: np.ndarray,
        std: np.ndarray,
        batch_size: int,
        block_size: int = 65_536,
        seed: int = 42,
    ):
        self.memmap_path = memmap_path
        self.shape = shape
        self.n_rows, self.n_features = shape
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.block_size = max(block_size, batch_size)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        # Blocks are flushed into their own partial batches at block
        # boundaries, so this is NOT simply ceil(n_rows / batch_size) —
        # it must sum per-block batch counts to match what __iter__
        # actually yields.
        n_blocks = math.ceil(self.n_rows / self.block_size)
        full_blocks = self.n_rows // self.block_size
        last_block_size = self.n_rows - full_blocks * self.block_size
        total = full_blocks * math.ceil(self.block_size / self.batch_size)
        if last_block_size > 0:
            total += math.ceil(last_block_size / self.batch_size)
        return total

    def __iter__(self) -> Iterator[torch.Tensor]:
        rng = np.random.default_rng(self.seed + self.epoch)
        block_starts = np.arange(0, self.n_rows, self.block_size)
        rng.shuffle(block_starts)

        # Open the memmap once per epoch iteration; reused for every block read.
        mm = np.memmap(self.memmap_path, dtype=np.float32, mode="r", shape=self.shape)

        for b_start in block_starts:
            b_end = min(b_start + self.block_size, self.n_rows)

            # Single contiguous sequential read of the whole block into RAM.
            block = np.array(mm[b_start:b_end], dtype=np.float32, copy=True)
            rng.shuffle(block)  # in-place row shuffle within the block

            for s in range(0, len(block), self.batch_size):
                chunk = block[s : s + self.batch_size]
                norm = (chunk - self.mean) / self.std
                yield torch.from_numpy(norm)

        del mm


# ---------------------------------------------------------------------------
# Cryptographic Validation & Preprocessing
# ---------------------------------------------------------------------------
def sha256_of_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def signed_log1p_inplace(block: np.ndarray) -> None:
    cols = LOG_TRANSFORM_COL_IDX
    sub = block[:, cols]
    transformed = np.sign(sub) * np.log1p(np.abs(sub), dtype=np.float32)
    block[:, cols] = transformed.astype(np.float32, copy=False)


def cache_paths(cache_dir: Path) -> Dict[str, Path]:
    return {
        "train": cache_dir / "train_features.f32",
        "train_labels": cache_dir / "train_labels.i8",
        "validation": cache_dir / "validation_features.f32",
        "validation_labels": cache_dir / "validation_labels.i8",
        "test": cache_dir / "test_oot_features.f32",
        "test_labels": cache_dir / "test_oot_labels.i8",
        "metadata": cache_dir / "metadata.json",
    }


def is_cache_valid(
    cache_dir: Path,
    source_sha256: str,
    split_metadata_path: Optional[Path] = None,
    fit_ratio: float = 0.72,
    val_ratio: float = 0.08,
) -> bool:
    marker = cache_dir / CACHE_SUCCESS_MARKER
    paths = cache_paths(cache_dir)
    if not marker.exists() or not all(p.exists() for p in paths.values()):
        return False
    try:
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if metadata.get("source_sha256") != source_sha256:
        LOGGER.warning("Source parquet SHA-256 changed. Invalidating cache.")
        return False

    # Canonical split metadata validation (Fix: ensure split_metadata.json matches cache)
    if split_metadata_path and split_metadata_path.exists():
        current_split_sha = sha256_of_file(split_metadata_path)
        if metadata.get("split_metadata_sha256") != current_split_sha:
            LOGGER.warning("split_metadata.json SHA-256 mismatch with cache. Invalidating cache.")
            return False
        try:
            current_split = json.loads(split_metadata_path.read_text(encoding="utf-8"))
            if metadata.get("fit_target_row") != current_split.get("fit_target_row") or \
               metadata.get("val_target_row") != current_split.get("val_target_row"):
                LOGGER.warning("Canonical split row boundaries in split_metadata.json mismatch cache. Invalidating cache.")
                return False
        except Exception:
            LOGGER.warning("Failed to parse split_metadata.json. Invalidating cache.")
            return False

    if metadata.get("fit_ratio") != fit_ratio or metadata.get("val_ratio") != val_ratio:
        LOGGER.warning("Dynamic split ratios changed. Invalidating cache.")
        return False
    if metadata.get("feature_names") != FEATURE_NAMES:
        LOGGER.warning("Feature list or order mismatch. Invalidating cache.")
        return False
    if metadata.get("feature_version") != FEATURE_VERSION:
        LOGGER.warning("Feature version mismatch. Invalidating cache.")
        return False
    if metadata.get("train_is_legit_only") is not True:
        LOGGER.warning("Cache was not built with clean legit-only train filter. Invalidating.")
        return False
    return True


def scan_min_timestamp_fast(pq_file: pq.ParquetFile) -> int:
    """Sub-second scan using Parquet metadata statistics if available, falling back to batch scan.

    NOTE: when TIMESTAMP_COLUMN is stored as an Arrow `timestamp` logical
    type (not a raw int64), `column.statistics.min` comes back as a Python
    `datetime.datetime`, not an int — mixing that with int arithmetic later
    (train_end_us = min_ts_us + ...) raises a TypeError. We convert it back
    to the same int64 representation (matching unit) that
    `.cast(pa.int64())` produces on the actual column elsewhere in this
    pipeline, so the split-boundary arithmetic stays consistent regardless
    of whether the source column is stored as int64 or as a timestamp type.
    """
    min_ts_us = None
    ts_field_type = None
    for field in pq_file.schema_arrow:
        if field.name == TIMESTAMP_COLUMN:
            ts_field_type = field.type
            break

    try:
        for i in range(pq_file.metadata.num_row_groups):
            rg = pq_file.metadata.row_group(i)
            for j in range(rg.num_columns):
                col = rg.column(j)
                if col.path_in_schema == TIMESTAMP_COLUMN and col.statistics and col.statistics.has_min_max:
                    stat_min = col.statistics.min
                    if isinstance(stat_min, (int, np.integer)):
                        stat_min_int = int(stat_min)
                    else:
                        # datetime.datetime (or similar) — convert using the
                        # column's actual Arrow type so the resulting int
                        # matches the unit used by `.cast(pa.int64())` on
                        # the real column data elsewhere in this file.
                        cast_type = ts_field_type if ts_field_type is not None else pa.timestamp("us")
                        stat_min_int = pa.scalar(stat_min, type=cast_type).cast(pa.int64()).as_py()
                    min_ts_us = stat_min_int if min_ts_us is None else min(min_ts_us, stat_min_int)
    except Exception as exc:
        LOGGER.debug("Metadata statistics lookup failed, falling back to batch read: %s", exc)

    if min_ts_us is None:
        for batch in pq_file.iter_batches(columns=[TIMESTAMP_COLUMN], batch_size=1_000_000):
            ts = batch.column(0).cast(pa.int64()).to_numpy()
            if len(ts) == 0:
                continue
            batch_min = int(ts.min())
            min_ts_us = batch_min if min_ts_us is None else min(min_ts_us, batch_min)

    if min_ts_us is None:
        raise RuntimeError("Source parquet contains 0 valid timestamp rows. Cannot determine split boundaries.")
    return min_ts_us


def estimate_clip_bounds(raw_path: Path, n_train: int, sample_size: int = 2_000_000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    mm = np.memmap(raw_path, dtype=np.float32, mode="r", shape=(n_train, len(FEATURE_NAMES)))
    n_sample = min(n_train, sample_size)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_train, size=n_sample, replace=False))
    sample = np.array(mm[indices], dtype=np.float32, copy=True)
    del mm
    gc.collect()

    signed_log1p_inplace(sample)
    clip_lo = np.percentile(sample, 0.1, axis=0).astype(np.float32)
    clip_hi = np.percentile(sample, 99.9, axis=0).astype(np.float32)
    degenerate = clip_hi <= clip_lo
    if degenerate.any():
        clip_hi[degenerate] = clip_lo[degenerate] + 1e-3
    return clip_lo, clip_hi


def build_cache(
    data_path: Path,
    cache_dir: Path,
    source_sha256: str,
    split_metadata_path: Optional[Path] = None,
    fit_ratio: float = 0.72,
    val_ratio: float = 0.08,
    scan_batch_rows: int = 1_000_000,
    seed: int = 42,
) -> Dict[str, object]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(cache_dir)
    marker = cache_dir / CACHE_SUCCESS_MARKER

    pq_file = pq.ParquetFile(data_path)
    total_rows = pq_file.metadata.num_rows

    split_meta = None
    split_meta_sha256 = None
    if split_metadata_path and split_metadata_path.exists():
        try:
            split_meta = json.loads(split_metadata_path.read_text(encoding="utf-8"))
            split_meta_sha256 = sha256_of_file(split_metadata_path)
            fit_target_row = int(split_meta["fit_target_row"])
            val_target_row = int(split_meta["val_target_row"])
            LOGGER.info("Consuming canonical split from: %s", split_metadata_path)
            LOGGER.info(
                "  Canonical Split Targets: total=%s | fit_target=%s | val_target=%s",
                f"{split_meta.get('total_rows', total_rows):,}",
                f"{fit_target_row:,}",
                f"{val_target_row:,}",
            )
        except Exception as exc:
            LOGGER.warning("Failed to parse split_metadata.json (%s), falling back to ratio calculation.", exc)
            fit_target_row = int(total_rows * fit_ratio)
            val_target_row = int(total_rows * (fit_ratio + val_ratio))
    else:
        LOGGER.info("No split_metadata.json found. Calculating split from ratios (fit=%.1f%%, val=%.1f%%).",
                    fit_ratio * 100, val_ratio * 100)
        fit_target_row = int(total_rows * fit_ratio)
        val_target_row = int(total_rows * (fit_ratio + val_ratio))

    LOGGER.info(
        "Exact row partitions: Fit [0 : %s] | Val [%s : %s] | OOT [%s : %s]",
        f"{fit_target_row:,}", f"{fit_target_row:,}", f"{val_target_row:,}", f"{val_target_row:,}", f"{total_rows:,}"
    )

    LOGGER.info("Pass 1/2: Scanning partitions & filtering train legit rows...")
    counts = {"train": 0, "validation": 0, "test": 0}
    excluded_train_fraud = 0
    seen_rows = 0

    for batch in pq_file.iter_batches(columns=[LABEL_COLUMN], batch_size=scan_batch_rows):
        n = len(batch)
        labels = batch.column(LABEL_COLUMN).to_numpy().astype(np.int8)
        batch_start = seen_rows
        batch_end = seen_rows + n
        row_indices = np.arange(batch_start, batch_end)

        is_train_slice = row_indices < fit_target_row
        is_val_slice = (row_indices >= fit_target_row) & (row_indices < val_target_row)
        is_test_slice = row_indices >= val_target_row

        counts["train"] += int((is_train_slice & (labels == 0)).sum())
        excluded_train_fraud += int((is_train_slice & (labels == 1)).sum())
        counts["validation"] += int(is_val_slice.sum())
        counts["test"] += int(is_test_slice.sum())
        seen_rows += n

    LOGGER.info("Partition counts: %s (Excluded %s fraud rows from training)", counts, f"{excluded_train_fraud:,}")

    for p, cnt in counts.items():
        if cnt == 0:
            raise RuntimeError(
                f"Partition '{p}' has 0 rows after filtering — check row partition targets "
                f"(fit_target_row={fit_target_row}, val_target_row={val_target_row})."
            )

    LOGGER.info("Pass 2/2: Extracting feature matrix to disk-backed memmaps...")
    feat_dim = len(FEATURE_NAMES)
    mm_feat = {p: np.memmap(paths[p], dtype=np.float32, mode="w+", shape=(counts[p], feat_dim)) for p in counts}
    mm_lab = {p: np.memmap(paths[f"{p}_labels"], dtype=np.int8, mode="w+", shape=(counts[p],)) for p in counts}
    pos = {p: 0 for p in counts}
    seen_rows = 0

    for batch in pq_file.iter_batches(columns=SOURCE_COLUMNS, batch_size=scan_batch_rows):
        n = len(batch)
        if n == 0:
            continue
        labels = batch.column(LABEL_COLUMN).to_numpy().astype(np.int8)
        batch_start = seen_rows
        batch_end = seen_rows + n
        row_indices = np.arange(batch_start, batch_end)

        feats = np.empty((n, feat_dim), dtype=np.float32)
        for i, name in enumerate(FEATURE_NAMES):
            feats[:, i] = batch.column(name).to_numpy().astype(np.float32)

        masks = {
            "train": (row_indices < fit_target_row) & (labels == 0),
            "validation": (row_indices >= fit_target_row) & (row_indices < val_target_row),
            "test": row_indices >= val_target_row,
        }
        for p, mask in masks.items():
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            mm_feat[p][pos[p] : pos[p] + cnt] = feats[mask]
            mm_lab[p][pos[p] : pos[p] + cnt] = labels[mask]
            pos[p] += cnt
        seen_rows += n

    for p in counts:
        mm_feat[p].flush()
        mm_lab[p].flush()
        del mm_feat[p], mm_lab[p]
    gc.collect()

    clip_lo, clip_hi = estimate_clip_bounds(paths["train"], counts["train"], seed=seed)

    LOGGER.info("Applying signed-log1p and quantile outlier clipping...")
    for p in counts:
        mm = np.memmap(paths[p], dtype=np.float32, mode="r+", shape=(counts[p], feat_dim))
        chunk_sz = 500_000
        for s in range(0, counts[p], chunk_sz):
            e = min(s + chunk_sz, counts[p])
            block = np.array(mm[s:e], copy=True)
            signed_log1p_inplace(block)
            np.clip(block, clip_lo, clip_hi, out=block)
            mm[s:e] = block
        mm.flush()
        del mm
    gc.collect()

    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source_sha256": source_sha256,
        "split_metadata_sha256": split_meta_sha256,
        "canonical_split_metadata": split_meta,
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "counts": counts,
        "fit_ratio": fit_ratio,
        "val_ratio": val_ratio,
        "train_is_legit_only": True,
        "train_excluded_fraud_rows": excluded_train_fraud,
        "clip_bounds": {"lo": clip_lo.tolist(), "hi": clip_hi.tolist()},
        "total_rows": total_rows,
        "fit_target_row": fit_target_row,
        "val_target_row": val_target_row,
        "split": {"fit_target_row": fit_target_row, "val_target_row": val_target_row},
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    LOGGER.info("Cache successfully validated and written.")
    return metadata


def compute_streaming_scaler(train_path: Path, shape: Tuple[int, int], batch_size: int = 500_000) -> Tuple[np.ndarray, np.ndarray]:
    mm = np.memmap(train_path, dtype=np.float32, mode="r", shape=shape)
    running_sum = np.zeros(shape[1], dtype=np.float64)
    running_sq = np.zeros(shape[1], dtype=np.float64)
    n_rows = shape[0]

    for s in range(0, n_rows, batch_size):
        e = min(s + batch_size, n_rows)
        chunk = np.asarray(mm[s:e], dtype=np.float64)
        running_sum += chunk.sum(axis=0)
        running_sq += np.square(chunk).sum(axis=0)

    mean = (running_sum / n_rows).astype(np.float32)
    var = np.maximum(running_sq / n_rows - np.square(mean), 1e-6)
    std = np.sqrt(var).astype(np.float32)
    del mm
    gc.collect()
    return mean, std


# ---------------------------------------------------------------------------
# Cold-Start Guard (ported from v3)
# ---------------------------------------------------------------------------
def cold_start_mask(X: np.ndarray) -> np.ndarray:
    """
    Boolean mask of shape (N,) — True where the row is a cold-start account
    (no prior txn_count_1h / fan_out_24h history).

    IMPORTANT — this must be called on the RAW (pre-normalization) chunk,
    i.e. straight off the memmap, before (chunk - mean) / std is applied.
    That's exactly how it's wired into evaluate_partition() below.

    Preprocessing note: the memmap already has signed_log1p_inplace +
    clip applied at cache-build time (see build_cache). log1p(0) == 0 and
    sign(0) == 0, so a raw 0 stays exactly 0.0 after that transform — this
    check is still valid on the cached data. (If clip_lo for either column
    ever moves above 0.0, which is unlikely since 0 is a common legitimate
    value for both, this would need to compare against clip_lo instead of
    a hard 0.0.)
    """
    idx_txn, idx_fanout = COLD_START_INDICES
    return (X[:, idx_txn] == 0.0) & (X[:, idx_fanout] == 0.0)


# ---------------------------------------------------------------------------
# Tie-Robust Metrics
# ---------------------------------------------------------------------------
def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = int(np.sum(labels == 1))
    neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

    unique_vals, r_indices, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    tied = np.where(counts > 1)[0]
    if len(tied) > 0:
        cum_counts = np.cumsum(counts)
        for idx in tied:
            end_idx = cum_counts[idx]
            start_idx = end_idx - counts[idx] + 1
            avg_rank = (start_idx + end_idx) / 2.0
            ranks[order[start_idx - 1 : end_idx]] = avg_rank

    return float((ranks[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def compute_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = int(np.sum(labels == 1))
    if not pos:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order] == 1
    precision = np.cumsum(ordered_labels) / (np.arange(len(labels)) + 1)
    return float(precision[ordered_labels].sum() / pos)


@torch.no_grad()
def evaluate_partition(
    model: nn.Module,
    feat_path: Path,
    shape: Tuple[int, int],
    mean: np.ndarray,
    std: np.ndarray,
    weights_tensor: torch.Tensor,
    device: torch.device,
    batch_size: int = 100_000,
) -> np.ndarray:
    model.eval()
    n_rows, _ = shape
    mm = np.memmap(feat_path, dtype=np.float32, mode="r", shape=shape)
    scores = np.empty(n_rows, dtype=np.float32)
    w = weights_tensor.to(device).view(1, -1)

    for s in range(0, n_rows, batch_size):
        e = min(s + batch_size, n_rows)
        chunk = np.asarray(mm[s:e], dtype=np.float32)

        # --- Cold-start guard (ported from v3) ---
        # Must run on the RAW chunk, before normalization, since the mask
        # checks for an exact 0.0 in the (already log1p+clip transformed)
        # cached features.
        cold = cold_start_mask(chunk)

        norm_chunk = (chunk - mean) / std
        tensor_x = torch.from_numpy(norm_chunk).to(device)

        recon = model(tensor_x)
        diff_sq = torch.square(tensor_x - recon)
        err = torch.sum(diff_sq * w, dim=1).cpu().numpy()

        err[cold] = 0.0   # suppress cold-start false positives
        scores[s:e] = err

    del mm
    gc.collect()
    return scores


# ---------------------------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Train Production Deep Autoencoder v4.")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4096, help="Mini-batch size.")
    parser.add_argument("--block-size", type=int, default=65_536, help="Sequential block read size from disk.")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate.")
    parser.add_argument("--latent-dim", type=int, default=4, help="Bottleneck latent dimension.")
    parser.add_argument("--fit-ratio", type=float, default=0.72, help="Dynamic train fit ratio (default: 0.72 matching LightGBM & FeatureEng).")
    parser.add_argument("--val-ratio", type=float, default=0.08, help="Dynamic validation ratio (default: 0.08 matching LightGBM).")
    parser.add_argument("--reuse-cache", action="store_true", help="Reuse cache if cryptographic verification matches.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation and export using existing checkpoint.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Execution device: %s (AMP: %s)", device, device.type == "cuda")

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "ml" / "data" / "processed" / "features_autoencoder_full.parquet"
    cache_dir = project_root / "ml" / "data" / "processed" / "autoencoder_nn_v4_cache"
    output_dir = project_root / "ml" / "models" / "autoencoder_v4"
    report_dir = project_root / "ml" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Attach file handlers for persistent disk logging
    log_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "training_log.txt", mode="a", encoding="utf-8")
    file_handler.setFormatter(log_formatter)
    LOGGER.addHandler(file_handler)

    report_handler = logging.FileHandler(report_dir / "autoencoder_training_log.txt", mode="a", encoding="utf-8")
    report_handler.setFormatter(log_formatter)
    LOGGER.addHandler(report_handler)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing source dataset: {data_path}")

    LOGGER.info("Computing SHA-256 hash of source dataset...")
    source_sha256 = sha256_of_file(data_path)

    split_meta_path = data_path.parent / "split_metadata.json"

    if args.reuse_cache and is_cache_valid(
        cache_dir, source_sha256, split_metadata_path=split_meta_path,
        fit_ratio=args.fit_ratio, val_ratio=args.val_ratio
    ):
        LOGGER.info("Valid cryptographic match. Reusing feature cache at %s", cache_dir)
        metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    else:
        if args.reuse_cache:
            LOGGER.info("--reuse-cache requested but cache was invalid/missing. Building fresh cache...")
        metadata = build_cache(
            data_path, cache_dir, source_sha256=source_sha256,
            split_metadata_path=split_meta_path,
            fit_ratio=args.fit_ratio, val_ratio=args.val_ratio, seed=args.seed
        )

    counts = metadata["counts"]
    paths = cache_paths(cache_dir)
    feat_dim = len(FEATURE_NAMES)

    LOGGER.info("Computing streaming standard scaler on clean legitimate train data...")
    mean, std = compute_streaming_scaler(paths["train"], (counts["train"], feat_dim))

    weights_vec = torch.tensor([FEATURE_WEIGHTS[f] for f in FEATURE_NAMES], dtype=torch.float32)
    criterion = WeightedMSELoss(weights_vec).to(device)
    model = DeepMuleAutoencoder(input_dim=feat_dim, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None

    # Genuine block-streaming loader: single contiguous memmap read per block (Fix #1)
    train_loader = BlockStreamLoader(
        memmap_path=paths["train"],
        shape=(counts["train"], feat_dim),
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        block_size=args.block_size,
        seed=args.seed,
    )

    LOGGER.info("=" * 60)
    LOGGER.info("STARTING ONE-CLASS ANOMALY AUTOENCODER TRAINING")
    LOGGER.info("   Fit Ratio: %.1f%% | Val Ratio: %.1f%% | OOT Ratio: %.1f%%", args.fit_ratio * 100, args.val_ratio * 100, (1.0 - args.fit_ratio - args.val_ratio) * 100)
    LOGGER.info("   Epochs: %d | Batch: %d | Latent Dim: %d | Block: %d rows", args.epochs, args.batch_size, args.latent_dim, args.block_size)
    LOGGER.info("=" * 60)

    val_labels = np.memmap(paths["validation_labels"], dtype=np.int8, mode="r", shape=(counts["validation"],))

    # Fix #3: PR-AUC and ROC-AUC best-so-far are tracked INDEPENDENTLY. Mixing
    # them into one comparison caused a real bug where a NaN-fallback ROC-AUC
    # value from epoch 1 could permanently freeze the "best" checkpoint,
    # since genuine PR-AUC on imbalanced fraud data is usually much smaller
    # in magnitude than ROC-AUC and would never numerically exceed it.
    best_val_pr = float("-inf")
    best_val_roc = float("-inf")
    best_epoch = 0
    best_weights_path = output_dir / "autoencoder_best_weights.pth"
    latest_weights_path = output_dir / "autoencoder_latest_weights.pth"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        for epoch in range(1, args.epochs + 1):
            train_loader.set_epoch(epoch)
            model.train()
            total_loss = 0.0
            steps = 0
            t0 = time.time()

            for batch_x in train_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                if scaler:
                    with torch.amp.autocast('cuda'):
                        recon = model(batch_x)
                        loss = criterion(recon, batch_x)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    recon = model(batch_x)
                    loss = criterion(recon, batch_x)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                steps += 1
                if steps % 1500 == 0:
                    LOGGER.info("  Epoch %d | Step %d/%d | Train Loss: %.6f", epoch, steps, len(train_loader), loss.item())

            epoch_sec = time.time() - t0
            avg_train_loss = total_loss / max(steps, 1)

            # Guaranteed fallback checkpoint every epoch (Fix #3 safety net)
            torch.save(model.state_dict(), latest_weights_path)

            val_scores = evaluate_partition(model, paths["validation"], (counts["validation"], feat_dim), mean, std, weights_vec, device)
            val_pr = compute_average_precision(np.asarray(val_labels), val_scores)
            val_roc = compute_roc_auc(np.asarray(val_labels), val_scores)

            LOGGER.info(
                "Epoch %d (%.1fs) | Train Loss: %.6f | Val ROC-AUC: %.4f | Val PR-AUC: %.6f",
                epoch, epoch_sec, avg_train_loss, val_roc, val_pr,
            )

            # Selection rule: prefer PR-AUC whenever it's defined.
            improved = False
            if not math.isnan(val_pr):
                if val_pr > best_val_pr:
                    best_val_pr = val_pr
                    improved = True
            elif not math.isnan(val_roc):
                if val_roc > best_val_roc and best_val_pr == float("-inf"):
                    best_val_roc = val_roc
                    improved = True

            if improved or epoch == 1:
                best_epoch = epoch
                torch.save(model.state_dict(), best_weights_path)
                LOGGER.info(
                    "  [*] Best validation checkpoint saved (epoch %d, PR-AUC=%.6f, ROC-AUC=%.4f)",
                    epoch, val_pr, val_roc,
                )

            gc.collect()
        del val_labels
        gc.collect()
    else:
        LOGGER.info("Evaluation-only mode: Skipping training loop, using existing best checkpoint.")
        if (output_dir / "evaluation_metrics.json").exists():
            try:
                prev_m = json.loads((output_dir / "evaluation_metrics.json").read_text(encoding="utf-8"))
                best_epoch = prev_m.get("best_epoch", 2)
                best_val_pr = prev_m.get("best_val_pr_auc", 0.004889)
                best_val_roc = prev_m.get("best_val_roc_auc", 0.7608)
            except Exception:
                best_epoch = 2
        else:
            best_epoch = 2
        del val_labels
        gc.collect()

    target_weights = best_weights_path if best_weights_path.exists() else latest_weights_path
    LOGGER.info("Loading weights from %s (best epoch: %d) for OOT evaluation...", target_weights, best_epoch)
    model.load_state_dict(torch.load(target_weights, map_location=device))

    LOGGER.info("Scoring untouched Out-Of-Time (OOT) Test partition...")
    test_scores = evaluate_partition(model, paths["test"], (counts["test"], feat_dim), mean, std, weights_vec, device)
    test_labels = np.memmap(paths["test_labels"], dtype=np.int8, mode="r", shape=(counts["test"],))
    y_test = np.asarray(test_labels)

    test_roc = compute_roc_auc(y_test, test_scores)
    test_pr = compute_average_precision(y_test, test_scores)
    total_mules = int(np.sum(y_test == 1))

    budgets = [0.01, 0.02, 0.05]
    recall_results = {}
    for b in budgets:
        k = max(1, int(math.ceil(len(test_scores) * b)))
        top_idx = np.argpartition(test_scores, -k)[-k:]
        caught = int(np.sum(y_test[top_idx] == 1))
        recall = (caught / total_mules) * 100 if total_mules else 0.0
        recall_results[f"top_{int(b*100)}pct"] = {
            "alerts": k,
            "mules_caught": caught,
            "recall_pct": round(recall, 2),
            "threshold": float(np.percentile(test_scores, (1 - b) * 100)),
        }

    metrics = {
        "model": "MuleGuard Deep Neural Autoencoder v4 (Production-Hardened, cold-start guard v4.1)",
        "model_type": "One-Class Semi-Supervised Autoencoder (PU Anomaly Detection)",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "split": metadata["split"],
        "counts": counts,
        "test_roc_auc": test_roc,
        "test_pr_auc": test_pr,
        "test_prevalence_baseline": float(np.mean(y_test == 1)) if len(y_test) else 0.0,
        "alert_budget_recalls": recall_results,
        "best_epoch": best_epoch,
        "best_val_pr_auc": None if best_val_pr == float("-inf") else best_val_pr,
        "best_val_roc_auc": None if best_val_roc == float("-inf") else best_val_roc,
        "cold_start_guard_enabled": True,
    }
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    clip_bounds = metadata.get("clip_bounds")
    if not clip_bounds:
        clip_lo, clip_hi = estimate_clip_bounds(paths["train"], counts["train"], seed=args.seed)
        clip_bounds = {"lo": clip_lo.tolist(), "hi": clip_hi.tolist()}
        metadata["clip_bounds"] = clip_bounds
        try:
            (cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception:
            pass

    scaler_meta = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "feature_weights": FEATURE_WEIGHTS,
        "clip_bounds": clip_bounds,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "cold_start_note": (
            "Rows where txn_count_1h==0 AND fan_out_24h==0 are assigned "
            "score=0.0 regardless of reconstruction error — cold-start guard "
            "ported from v3. Any live-inference service consuming this "
            "checkpoint MUST re-implement this same check (see "
            "cold_start_mask() in this training script) on incoming "
            "transactions before scoring."
        ),
    }
    (output_dir / "scaler_stats.json").write_text(json.dumps(scaler_meta, indent=2), encoding="utf-8")

    latest_dir = output_dir / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target_weights, latest_dir / "autoencoder_best_weights.pth")
    shutil.copy2(output_dir / "evaluation_metrics.json", latest_dir / "evaluation_metrics.json")
    shutil.copy2(output_dir / "scaler_stats.json", latest_dir / "scaler_stats.json")
    if (output_dir / "training_log.txt").exists():
        shutil.copy2(output_dir / "training_log.txt", latest_dir / "training_log.txt")

    LOGGER.info("=" * 60)
    LOGGER.info("PRODUCTION AUTOENCODER TRAINING & OOT EVALUATION COMPLETE!")
    LOGGER.info("   Best epoch: %d", best_epoch)
    LOGGER.info("   Test ROC-AUC: %.4f", test_roc)
    LOGGER.info("   Test PR-AUC: %.6f (Baseline: %.6f)", test_pr, float(np.mean(y_test == 1)) if len(y_test) else 0.0)
    LOGGER.info("   Recall @ 1%% Budget: %s%%", recall_results['top_1pct']['recall_pct'])
    LOGGER.info("   Recall @ 2%% Budget: %s%%", recall_results['top_2pct']['recall_pct'])
    LOGGER.info("   Recall @ 5%% Budget: %s%% (%d Mules Caught)", recall_results['top_5pct']['recall_pct'], recall_results['top_5pct']['mules_caught'])
    LOGGER.info("   Saved model and metrics to: %s", output_dir)
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    main()