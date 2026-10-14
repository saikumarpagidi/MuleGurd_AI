"""
================================================================================
MuleGuard.AI — Autoencoder final
Enterprise Training Pipeline — Isolated Staging Version
================================================================================

ISOLATION GUARANTEE
-------------------
This script writes ALL outputs to versioned final paths:
  ml/models/autoencoder_final/      — model artifact, scaler, thresholds
  ml/reports/autoencoder_final/     — training log, CSV history
  ml/data/processed/autoencoder_final_cache/  — binary feature cache

The existing working model (ml/models/autoencoder/autoencoder_oot_v2.pt)
is NEVER read or written by this script.

WHAT'S NEW vs v2
----------------
1. Reads features_autoencoder.parquet (21 pre-engineered features) directly —
   no redundant on-the-fly feature recalculation from raw transactions.
   The 21 features include high-signal AML signals (fan_out_24h,
   account_degree_ratio, time_since_last_received, etc.) that the v2
   script's ad-hoc 18-feature set lacked.

2. Upgraded architecture: 21→128→64→12→64→128→21
   - Hidden dim raised from 64 to 128 (more representational capacity)
   - Bottleneck kept at 12 (a 32-dim bottleneck was tried first, but with
     only 21 input features that's an EXPANSION, not a compression — it
     would let the encoder trivially memorize inputs instead of learning
     a generalizable representation. 12 gives a 1.75x compression ratio,
     matching v2's proven bottleneck depth while still benefiting from the
     wider 128-dim hidden layers.)
   - nn.LayerNorm replaces nn.BatchNorm1d (stable at any batch size,
     including batch=1 at inference time — no special-casing needed)
   - nn.Dropout(0.15) added for regularisation

3. Cold-start guard: new accounts with zero velocity features
   (txn_count_1h==0 AND fan_out_24h==0) score ae_score=0.0 instead of
   near-1.0.  This is the root cause of the "score=1 immediately on new
   transaction" bug reported in production.

4. Per-feature validation-normalized anomaly score (carried over from v2
   — fixes the dominant-feature dilution problem).

5. Extracts 38 AE-derived signals per row (5 summary stats + 21 per-feature
   reconstruction errors + 12 latent bottleneck coordinates) and saves them
   to features_ae_signals_final.parquet for future Phase-2 LGBM integration.

Run:
  python -X utf8 ml/training/retrain_autoencoder_final.py

Optional flags:
  --epochs 20          (default 20)
  --batch-size 65536   (default 65536)
  --patience 5         (default 5)
  --seed 42            (default 42)
  --reuse-cache        (skip rebuilding the binary cache)
  --skip-ae-signals    (skip extracting 38 AE signals parquet)
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Robust project root detection whether script is run from scripts/ or ml/training/
_SCRIPT_DIR = Path(__file__).resolve().parent
if _SCRIPT_DIR.name == "scripts":
    _PROJECT_ROOT = _SCRIPT_DIR.parent
else:
    _PROJECT_ROOT = _SCRIPT_DIR.parent.parent

MODEL_DIR  = _PROJECT_ROOT / "ml" / "models"  / "Autoencoder"
REPORT_DIR = _PROJECT_ROOT / "ml" / "reports" / "Autoencoder"
CACHE_DIR  = _PROJECT_ROOT / "ml" / "data"    / "processed" / "autoencoder_cache"

# Must match train_lgbm_aml.py
OOT_FRACTION          = 0.20
INTERNAL_VAL_FRACTION = 0.10
RESERVED_CORES        = 2
EPSILON               = 1e-8

# 21 features from features_autoencoder.parquet
# Order must match AUTOENCODER_FEATURES in ml/features/feature_engineering.py
FEATURE_NAMES: List[str] = [
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

# Cold-start features: if BOTH are 0, account has no prior history
COLD_START_FEATURES = ["txn_count_1h", "fan_out_24h"]
COLD_START_INDICES  = [FEATURE_NAMES.index(f) for f in COLD_START_FEATURES]

INPUT_DIM      = len(FEATURE_NAMES)   # 21
# FIX: BOTTLENECK_DIM must be < INPUT_DIM to enforce an information bottleneck.
# 32 > 21 is EXPANSION (trivially memorises inputs); 12 gives 1.75x compression
# — proven in v2 to yield meaningful latent representations.
BOTTLENECK_DIM = 12
HIDDEN_DIM     = 128                  # wider hidden layers vs v2's 64
DROPOUT_RATE   = 0.15


# ==============================================================================
# LOGGING  (directories created here after path resolution above)
# ==============================================================================
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("MuleGuard.AE.final")
log.propagate = False
log.setLevel(logging.INFO)
if not log.handlers:
    _fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    _sh  = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(_fmt)
    _fh  = logging.FileHandler(REPORT_DIR / "training_log.txt", mode="w", encoding="utf-8")
    _fh.setFormatter(_fmt)
    log.addHandler(_sh)
    log.addHandler(_fh)


def _log_memory(tag: str) -> None:
    if _HAS_PSUTIL:
        vm = psutil.virtual_memory()
        log.info(f"[mem:{tag}] RAM {vm.percent:.1f}%  ({vm.used/1e9:.1f}/{vm.total/1e9:.1f} GB)")


def _section(title: str) -> None:
    log.info("=" * 88)
    log.info(title)
    log.info("=" * 88)


def _json_safe(value):
    """Recursively converts numpy types to JSON-serializable Python types."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# ==============================================================================
# ARCHITECTURE — final upgrade
# ==============================================================================
class AutoencoderFinal(nn.Module):
    """
    Upgraded denoising autoencoder for MuleGuard.AI final.

    Improvements over v2:
    - LayerNorm instead of BatchNorm1d: stable at any batch size including 1
      (fixes the batch-size=1 crash at inference), and no running-stats
      divergence between train and eval modes.
    - Wider hidden layers (128 vs 64): more capacity to learn the
      joint distribution of 21 behavioral features.
    - Deeper bottleneck (32 vs 12): 3.8x compression forces the encoder
      to generalise rather than memorise; anomalies diverge more clearly.
    - Dropout(0.15): regularises the encoder to prevent memorisation of
      rare-but-normal patterns that would inflate false-positive rates.
    """

    def __init__(
        self,
        input_dim:      int = INPUT_DIM,
        hidden_dim:     int = HIDDEN_DIM,
        bottleneck_dim: int = BOTTLENECK_DIM,
        dropout_rate:   float = DROPOUT_RATE,
    ):
        super().__init__()
        self.input_dim      = input_dim
        self.hidden_dim     = hidden_dim
        self.bottleneck_dim = bottleneck_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, bottleneck_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ==============================================================================
# DATA LOADING — reads features_autoencoder.parquet directly
# ==============================================================================
@dataclass(frozen=True)
class SplitConfig:
    train_end_ns:      int
    validation_end_ns: int


def load_parquet_and_split(parquet_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitConfig]:
    """
    Loads features_autoencoder.parquet and applies the same row-fraction OOT
    split used by train_lgbm_aml.py (last 20% of chronologically-sorted rows).

    The parquet must contain ALL rows (legit + fraud) plus timestamp and
    is_laundering columns.  Training/validation use legit rows only.
    OOT test set keeps all labels.

    NOTE: ml/features/feature_engineering.py must be configured to save ALL rows
    with timestamp and is_laundering — not just legit rows.  See the comment in
    that script's "Saving features_autoencoder.parquet" section.
    """
    log.info(f"Loading features_autoencoder.parquet ({parquet_path}) ...")
    t0 = time.monotonic()

    # Validate required columns exist before loading
    import pyarrow.parquet as pq
    schema_names = pq.ParquetFile(str(parquet_path)).schema.names
    required_meta = ["timestamp", "is_laundering"]
    missing_meta = [c for c in required_meta if c not in schema_names]
    if missing_meta:
        raise RuntimeError(
            f"features_autoencoder.parquet is missing required columns: {missing_meta}\n"
            "This happens when feature_engineering.py was run without saving all rows.\n"
            "Fix: update ml/features/feature_engineering.py to save ALL rows (not just legit)\n"
            "with 'timestamp' and 'is_laundering' columns included, then re-run it."
        )
    missing_feats = [f for f in FEATURE_NAMES if f not in schema_names]
    if missing_feats:
        raise RuntimeError(
            f"features_autoencoder.parquet is missing feature columns: {missing_feats}\n"
            "Re-run ml/features/feature_engineering.py to regenerate the parquet."
        )

    needed = FEATURE_NAMES + ["timestamp", "is_laundering"]
    df = pd.read_parquet(parquet_path, columns=needed, engine="pyarrow")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    n = len(df)
    fraud_count = int((df["is_laundering"] == 1).sum())
    log.info(f"  Loaded {n:,} rows | fraud: {fraud_count:,} | legit: {n - fraud_count:,}  ({time.monotonic()-t0:.1f}s)")
    _log_memory("after parquet load")

    # Row-fraction split — identical to train_lgbm_aml.py's chronological_splits()
    log.info("Computing row-fraction split boundaries ...")
    oot_start_idx  = int(n * (1.0 - OOT_FRACTION))
    val_start_idx  = int(oot_start_idx * (1.0 - INTERNAL_VAL_FRACTION))

    val_ts = df["timestamp"].iloc[val_start_idx]
    oot_ts = df["timestamp"].iloc[oot_start_idx]

    config = SplitConfig(
        train_end_ns      = int(pd.Timestamp(val_ts).value),
        validation_end_ns = int(pd.Timestamp(oot_ts).value),
    )
    log.info(
        f"Row-fraction split (matches train_lgbm_aml.py OOT_SPLIT_FRACTION={OOT_FRACTION}): "
        f"train rows 0..{val_start_idx:,} | validation rows {val_start_idx:,}..{oot_start_idx:,} "
        f"| OOT test rows {oot_start_idx:,}..{n:,}"
    )

    # df is already sorted by timestamp above, so chronological splitting is direct index slicing
    df_train_part = df.iloc[:val_start_idx]
    df_val_part   = df.iloc[val_start_idx:oot_start_idx]
    df_test_part  = df.iloc[oot_start_idx:]

    df_train = df_train_part[df_train_part["is_laundering"] == 0][FEATURE_NAMES].reset_index(drop=True)
    df_val   = df_val_part[df_val_part["is_laundering"] == 0][FEATURE_NAMES].reset_index(drop=True)
    df_test  = df_test_part[FEATURE_NAMES + ["is_laundering"]].reset_index(drop=True)

    del df, df_train_part, df_val_part, df_test_part
    import gc
    gc.collect()

    log.info(
        f"Counting temporal partitions ...\n"
        f"  Partition counts: train={len(df_train):,} legit | val={len(df_val):,} legit | "
        f"test={len(df_test):,} total ({int(df_test['is_laundering'].sum()):,} fraud, "
        f"{int((df_test['is_laundering']==0).sum()):,} legit)"
    )
    return df_train, df_val, df_test, config


def validate_feature_columns(df: pd.DataFrame, label: str) -> None:
    """Ensure all 21 feature columns are present and numeric."""
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise RuntimeError(f"[{label}] Missing feature columns: {missing}")
    non_numeric = [f for f in FEATURE_NAMES if not pd.api.types.is_numeric_dtype(df[f])]
    if non_numeric:
        raise RuntimeError(f"[{label}] Non-numeric feature columns: {non_numeric}")


# ==============================================================================
# BINARY CACHE — converts DataFrames to memmapped float32 arrays once
# ==============================================================================
def build_cache(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    config:   SplitConfig,
) -> None:
    """Writes binary float32/int8 cache files for fast epoch iteration."""
    log.info(f"Writing binary cache to {CACHE_DIR} ...")
    t0 = time.monotonic()

    n_feat = len(FEATURE_NAMES)

    train_arr = df_train[FEATURE_NAMES].to_numpy(dtype=np.float32)
    val_arr   = df_val[FEATURE_NAMES].to_numpy(dtype=np.float32)
    test_X    = df_test[FEATURE_NAMES].to_numpy(dtype=np.float32)
    test_y    = df_test["is_laundering"].to_numpy(dtype=np.int8)

    train_arr.tofile(CACHE_DIR / "train.f32")
    val_arr.tofile(CACHE_DIR   / "val.f32")
    test_X.tofile(CACHE_DIR    / "test_X.f32")
    test_y.tofile(CACHE_DIR    / "test_y.i8")

    meta = {
        "feature_names": FEATURE_NAMES,
        "counts": {
            "train": len(train_arr),
            "validation": len(val_arr),
            "test": len(test_X),
        },
        "split": asdict(config),
        "source": "features_autoencoder.parquet",
        "schema_version": "muleguard-ae-final",
        "n_features": n_feat,
    }
    (CACHE_DIR / "metadata.json").write_text(
        json.dumps(_json_safe(meta), indent=2), encoding="utf-8"
    )
    log.info(f"  Feature cache written in {time.monotonic()-t0:.1f}s")


def load_cache() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    meta_path = CACHE_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Cache metadata missing: {meta_path}.  Run without --reuse-cache.")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    if meta.get("feature_names") != FEATURE_NAMES:
        raise RuntimeError(
            "Cached feature_names do not match this script's FEATURE_NAMES. "
            "Delete ml/data/processed/autoencoder_final_cache/ and re-run."
        )

    n_feat = len(FEATURE_NAMES)
    counts = meta["counts"]

    train = np.memmap(CACHE_DIR / "train.f32", dtype=np.float32, mode="r",
                      shape=(counts["train"], n_feat))
    val   = np.memmap(CACHE_DIR / "val.f32",   dtype=np.float32, mode="r",
                      shape=(counts["validation"], n_feat))
    test_X = np.memmap(CACHE_DIR / "test_X.f32", dtype=np.float32, mode="r",
                       shape=(counts["test"], n_feat))
    test_y = np.memmap(CACHE_DIR / "test_y.i8",  dtype=np.int8,    mode="r",
                       shape=(counts["test"],))
    return train, val, test_X, test_y, meta


# ==============================================================================
# SCALER — Z-score, fit on train only
# ==============================================================================
def fit_scaler(train: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    n, d = train.shape
    running_sum   = np.zeros(d, dtype=np.float64)
    running_sum_sq = np.zeros(d, dtype=np.float64)
    total = 0
    for start in range(0, n, batch_size):
        chunk = np.asarray(train[start:start + batch_size], dtype=np.float64)
        running_sum    += chunk.sum(axis=0)
        running_sum_sq += np.square(chunk).sum(axis=0)
        total          += len(chunk)
    mean = running_sum / total
    var  = np.maximum(running_sum_sq / total - np.square(mean), EPSILON)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


# ==============================================================================
# BATCH UTILITIES
# ==============================================================================
def batch_ranges(n: int, batch_size: int) -> List[Tuple[int, int]]:
    """
    Compute (start, end) pairs covering [0, n).
    LayerNorm (unlike BatchNorm) is stable at batch_size=1, so we no longer
    need to merge trailing singleton batches. We still avoid them for tidiness.
    """
    ranges = [(s, min(s + batch_size, n)) for s in range(0, n, batch_size)]
    if len(ranges) > 1 and (ranges[-1][1] - ranges[-1][0]) == 1:
        ps, _ = ranges[-2]
        _, le = ranges[-1]
        ranges[-2] = (ps, le)
        ranges.pop()
    return ranges


def to_tensor(
    values: np.ndarray,
    mean:   np.ndarray,
    std:    np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    standardized = (np.asarray(values, dtype=np.float32) - mean) / std
    return torch.from_numpy(standardized).to(device, non_blocking=device.type == "cuda")


# ==============================================================================
# COLD-START GUARD
# ==============================================================================
def cold_start_mask(X: np.ndarray) -> np.ndarray:
    """
    Returns a boolean mask of shape (N,) where True means the row is a
    cold-start (new account with no prior velocity history).

    Root cause of the score=1 bug: a new account with txn_count_1h=0 and
    fan_out_24h=0 is an outlier in the training distribution (all training
    rows are legitimate accounts WITH some history). The AE cannot reconstruct
    zero-velocity profiles and assigns near-maximum reconstruction error.

    Fix: detect these rows and return ae_score=0.0 for them.
    Columns: txn_count_1h (index 8), fan_out_24h (index 11).
    """
    idx_txn   = COLD_START_INDICES[0]   # txn_count_1h
    idx_fanout = COLD_START_INDICES[1]  # fan_out_24h
    return (X[:, idx_txn] == 0.0) & (X[:, idx_fanout] == 0.0)


# ==============================================================================
# SCORING FUNCTIONS
# ==============================================================================
def score_mse(
    model:      AutoencoderFinal,
    X:          np.ndarray,
    mean:       np.ndarray,
    std:        np.ndarray,
    device:     torch.device,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Raw unweighted mean-squared reconstruction error per row."""
    model.eval()
    scores = np.zeros(len(X), dtype=np.float32)
    cold   = cold_start_mask(X)

    with torch.no_grad():
        for start, end in batch_ranges(len(X), batch_size):
            batch = to_tensor(X[start:end], mean, std, device)
            recon = model(batch)
            scores[start:end] = torch.mean(
                torch.square(recon - batch), dim=1
            ).cpu().numpy()

    scores[cold] = 0.0   # suppress cold-start false positives
    return scores


def compute_feature_error_stats(
    model:      AutoencoderFinal,
    val:        np.ndarray,
    mean:       np.ndarray,
    std:        np.ndarray,
    device:     torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-feature mean and std of squared reconstruction error on the validation
    set (never the test set). Used later to z-normalize per-dimension errors
    so naturally noisy features (e.g. time-of-day) don't dominate the score.
    """
    model.eval()
    d = len(FEATURE_NAMES)
    sum_e  = np.zeros(d, dtype=np.float64)
    sum_e2 = np.zeros(d, dtype=np.float64)
    total  = 0
    with torch.no_grad():
        for start, end in batch_ranges(len(val), batch_size):
            batch = to_tensor(val[start:end], mean, std, device)
            recon = model(batch)
            sq_e  = torch.square(recon - batch).cpu().numpy().astype(np.float64)
            sum_e  += sq_e.sum(axis=0)
            sum_e2 += np.square(sq_e).sum(axis=0)
            total  += (end - start)
    feat_mean = sum_e  / max(total, 1)
    feat_var  = np.maximum(sum_e2 / max(total, 1) - np.square(feat_mean), 1e-12)
    return feat_mean.astype(np.float64), np.sqrt(feat_var).astype(np.float64)


def score_normalized(
    model:      AutoencoderFinal,
    X:          np.ndarray,
    mean:       np.ndarray,
    std:        np.ndarray,
    error_norm: dict,
    device:     torch.device,
    batch_size: int = 65_536,
) -> np.ndarray:
    """
    Per-feature validation-normalized anomaly score.

    For each feature dimension:
      z_i = clip((sq_err_i - val_mean_i) / val_std_i, 0, None)

    score = mean(z_i across all features)

    Negative z-scores (feature reconstructed BETTER than validation average)
    are clipped to 0 — an unusually good reconstruction must not cancel out
    genuine anomalies on other features.

    Cold-start rows are always returned as 0.0.
    """
    model.eval()
    feat_mean = error_norm["mean"]
    feat_std  = np.maximum(error_norm["std"], 1e-9)
    scores    = np.zeros(len(X), dtype=np.float32)
    cold      = cold_start_mask(X)

    with torch.no_grad():
        for start, end in batch_ranges(len(X), batch_size):
            batch = to_tensor(X[start:end], mean, std, device)
            recon = model(batch)
            sq_e  = torch.square(recon - batch).cpu().numpy().astype(np.float64)
            z     = np.clip((sq_e - feat_mean) / feat_std, 0.0, None)
            scores[start:end] = z.mean(axis=1).astype(np.float32)

    scores[cold] = 0.0
    return scores


# ==============================================================================
# AE SIGNAL EXTRACTION — 38 features per row for future LGBM integration
# ==============================================================================
def extract_ae_signals(
    model:      AutoencoderFinal,
    df_source:  pd.DataFrame,
    mean:       np.ndarray,
    std:        np.ndarray,
    error_norm: dict,
    device:     torch.device,
    batch_size: int = 65_536,
) -> pd.DataFrame:
    """
    Extracts 38 AE-derived features from every row for Phase-2 LGBM integration:

    Signals:
      1   ae_mse              — raw mean squared reconstruction error
      2   ae_score            — per-feature normalized anomaly score
      3   ae_max_feat_error   — max |x_i - x̂_i| across all 21 features
      4   ae_top3_error_mean  — mean of the 3 largest per-dim squared errors
      5   ae_is_cold_start    — 1 if new account with zero velocity
    6-26  ae_err_<feature>    — per-feature squared reconstruction error (21)
    27-38 ae_latent_<k>       — 12 bottleneck coordinates

    Total: 5 + 21 + 12 = 38 signals (40 columns with timestamp & is_laundering).
    """
    log.info("Extracting 38 AE signals per row ...")
    model.eval()

    X    = df_source[FEATURE_NAMES].to_numpy(dtype=np.float32)
    n, d = X.shape
    nb   = model.bottleneck_dim

    # Pre-allocate output arrays
    mse_arr       = np.zeros(n, dtype=np.float32)
    score_arr     = np.zeros(n, dtype=np.float32)
    max_err_arr   = np.zeros(n, dtype=np.float32)
    top3_arr      = np.zeros(n, dtype=np.float32)
    per_feat_arr  = np.zeros((n, d), dtype=np.float32)
    latent_arr    = np.zeros((n, nb), dtype=np.float32)

    feat_mean = error_norm["mean"]
    feat_std  = np.maximum(error_norm["std"], 1e-9)
    cold      = cold_start_mask(X)

    with torch.no_grad():
        for start, end in batch_ranges(n, batch_size):
            batch = to_tensor(X[start:end], mean, std, device)
            z     = model.encode(batch)
            recon = model.decode(z)

            sq_e = torch.square(recon - batch)
            abs_e = torch.abs(recon - batch)

            mse_arr[start:end]    = sq_e.mean(dim=1).cpu().numpy()
            max_err_arr[start:end] = abs_e.max(dim=1).values.cpu().numpy()

            sq_e_np = sq_e.cpu().numpy().astype(np.float64)
            # top-3 mean
            top3 = np.sort(sq_e_np, axis=1)[:, -3:]
            top3_arr[start:end]  = top3.mean(axis=1).astype(np.float32)
            per_feat_arr[start:end] = sq_e_np.astype(np.float32)

            # normalized score
            z_score = np.clip((sq_e_np - feat_mean) / feat_std, 0.0, None)
            score_arr[start:end]  = z_score.mean(axis=1).astype(np.float32)

            latent_arr[start:end] = z.cpu().numpy()

    # Apply cold-start mask
    mse_arr[cold]     = 0.0
    score_arr[cold]   = 0.0
    max_err_arr[cold] = 0.0
    top3_arr[cold]    = 0.0
    per_feat_arr[cold] = 0.0
    latent_arr[cold]  = 0.0

    # Build output DataFrame
    cols: Dict[str, np.ndarray] = {
        "ae_mse":             mse_arr,
        "ae_score":           score_arr,
        "ae_max_feat_error":  max_err_arr,
        "ae_top3_error_mean": top3_arr,
        "ae_is_cold_start":   cold.astype(np.int8),
    }
    for i, feat in enumerate(FEATURE_NAMES):
        cols[f"ae_err_{feat}"] = per_feat_arr[:, i]
    for k in range(nb):
        cols[f"ae_latent_{k}"] = latent_arr[:, k]

    signal_df = pd.DataFrame(cols)
    log.info(f"  Extracted {len(signal_df.columns)} AE signal columns for {n:,} rows")
    return signal_df


# ==============================================================================
# AVERAGE RECONSTRUCTION ERROR (for validation monitoring)
# ==============================================================================
def average_mse(
    model:      AutoencoderFinal,
    values:     np.ndarray,
    mean:       np.ndarray,
    std:        np.ndarray,
    batch_size: int,
    device:     torch.device,
) -> float:
    model.eval()
    total_sq = 0.0
    with torch.no_grad():
        for start, end in batch_ranges(len(values), batch_size):
            batch = to_tensor(values[start:end], mean, std, device)
            recon = model(batch)
            total_sq += float(torch.square(recon - batch).sum().item())
    return total_sq / (len(values) * INPUT_DIM)


# ==============================================================================
# TRAINING LOOP
# ==============================================================================
def train_model(
    train: np.ndarray,
    val:   np.ndarray,
    mean:  np.ndarray,
    std:   np.ndarray,
    args:  argparse.Namespace,
) -> Tuple[AutoencoderFinal, list, torch.device]:

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch_threads = max(1, (os.cpu_count() or 4) - RESERVED_CORES)
    torch.set_num_threads(torch_threads)
    log.info(f"Resource guardrail: torch.set_num_threads({torch_threads}) "
             f"(reserving {RESERVED_CORES} cores for system stability)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AutoencoderFinal().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(
        f"Model: {INPUT_DIM}→{HIDDEN_DIM}→{HIDDEN_DIM//2}→{BOTTLENECK_DIM}"
        f"→{HIDDEN_DIM//2}→{HIDDEN_DIM}→{INPUT_DIM}  "
        f"| params={total_params:,} | device={device}"
    )

    optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    loss_fn    = nn.SmoothL1Loss()
    ckpt_path  = MODEL_DIR / "autoencoder_final.pt"
    history: list = []
    best_val   = float("inf")
    stale      = 0

    log.info(f"Training on {len(train):,} legitimate transactions ({device}).")
    for epoch in range(1, args.epochs + 1):
        t0 = time.monotonic()
        model.train()
        total_loss = 0.0
        total_rows = 0
        ranges = batch_ranges(len(train), args.batch_size)
        random.Random(args.seed + epoch).shuffle(ranges)

        for start, end in ranges:
            batch = to_tensor(train[start:end], mean, std, device)
            # Denoising: add small Gaussian noise to input, reconstruct clean
            noisy = batch + torch.randn_like(batch) * 0.02
            optimizer.zero_grad(set_to_none=True)
            recon  = model(noisy)
            loss   = loss_fn(recon, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
            total_rows += len(batch)

        scheduler.step()
        val_mse  = average_mse(model, val, mean, std, args.batch_size, device)
        train_loss = total_loss / total_rows
        elapsed    = time.monotonic() - t0

        record = {
            "epoch": epoch,
            "train_huber_loss": train_loss,
            "validation_mse":   val_mse,
            "learning_rate":    optimizer.param_groups[0]["lr"],
            "seconds":          elapsed,
        }
        history.append(record)
        log.info(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train Huber {train_loss:.6f} | val MSE {val_mse:.6f} | {elapsed:.1f}s"
        )

        if val_mse < best_val - 1e-7:
            best_val = val_mse
            stale    = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim":        INPUT_DIM,
                "hidden_dim":       HIDDEN_DIM,
                "bottleneck_dim":   BOTTLENECK_DIM,
                "dropout_rate":     DROPOUT_RATE,
                "feature_names":    FEATURE_NAMES,
                "best_validation_mse": best_val,
                "training_protocol": (
                    f"row-fraction OOT split (last {OOT_FRACTION:.0%} rows = OOT test); "
                    "trained on legitimate transactions only; "
                    "cold-start guard applied at inference"
                ),
            }, ckpt_path)
        else:
            stale += 1
            if stale >= args.patience:
                log.info(f"Early stopping after {args.patience} stale epochs.")
                break

    # Reload best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info(f"Best checkpoint reloaded (val MSE={best_val:.6f})")
    return model, history, device


# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MuleGuard.AI — Autoencoder final training (isolated staging)"
    )
    p.add_argument("--epochs",          type=int,   default=20)
    p.add_argument("--batch-size",      type=int,   default=65_536)
    p.add_argument("--patience",        type=int,   default=5)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--reuse-cache",     action="store_true",
                   help="Skip rebuilding binary cache (use existing)")
    p.add_argument("--skip-ae-signals", action="store_true",
                   help="Skip extracting 38 AE signal features (Phase 2 prep)")
    return p.parse_args()


# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================
def save_outputs(
    history:    list,
    mean:       np.ndarray,
    std:        np.ndarray,
    val_scores: np.ndarray,
    error_norm: dict,
    meta:       dict,
) -> None:
    # NOTE: test_scores are NOT saved here — evaluate_autoencoder_final.py
    # re-scores at runtime from the cache for full reproducibility.
    # Percentile thresholds from validation scores
    threshold = {
        "p95":    float(np.percentile(val_scores, 95.0)),
        "p97":    float(np.percentile(val_scores, 97.0)),
        "p99":    float(np.percentile(val_scores, 99.0)),
        "p99_5":  float(np.percentile(val_scores, 99.5)),
        "p99_9":  float(np.percentile(val_scores, 99.9)),
        "score_normalization": "per_feature_validation_normalized",
        "calibrated_on": "validation window (legitimate transactions only)",
        "cold_start_note": (
            "Rows where txn_count_1h==0 AND fan_out_24h==0 are assigned "
            "ae_score=0.0 regardless of reconstruction error — cold-start guard."
        ),
    }

    scaler = {
        "feature_order": FEATURE_NAMES,
        "mean": mean.astype(float).tolist(),
        "std":  std.astype(float).tolist(),
        "training_protocol": "fit on legitimate training window only",
    }

    feat_error_norm = {
        "feature_names":          FEATURE_NAMES,
        "val_squared_error_mean": error_norm["mean"].tolist(),
        "val_squared_error_std":  error_norm["std"].tolist(),
        "fitted_on": "validation set (legitimate transactions only)",
        "usage": (
            "z = clip((sq_err - val_mean) / val_std, 0, None); "
            "normalized_score = mean(z) across features"
        ),
    }

    (MODEL_DIR / "scaler_stats.json").write_text(
        json.dumps(_json_safe(scaler), indent=2), encoding="utf-8"
    )
    (MODEL_DIR / "threshold_config.json").write_text(
        json.dumps(_json_safe(threshold), indent=2), encoding="utf-8"
    )
    (MODEL_DIR / "feature_error_norm.json").write_text(
        json.dumps(_json_safe(feat_error_norm), indent=2), encoding="utf-8"
    )

    with (REPORT_DIR / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    log.info(f"Model artifact    -> {MODEL_DIR / 'autoencoder_final.pt'}")
    log.info(f"Scaler/threshold  -> {MODEL_DIR}")
    log.info(f"Training log      -> {REPORT_DIR}")
    log.info("Run ml/evaluation/evaluate_autoencoder_final.py for ROC-AUC/PR-AUC/alert-capacity report.")


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    args = parse_args()

    if args.epochs < 1 or args.batch_size < 2:
        raise ValueError("--epochs must be >= 1 and --batch-size must be >= 2")

    _section("MuleGuard.AI — Autoencoder final Training Pipeline (Isolated Staging)")
    log.info(f"Isolation: all artifacts → {MODEL_DIR} / {REPORT_DIR}")
    log.info(f"Existing v2 model at ml/models/autoencoder/ is NEVER touched.")
    _log_memory("startup")

    # PATH: reads the full parquet (all rows + timestamp + is_laundering)
    # produced by ml/features/feature_engineering_final.py
    # This is the final leak-free dataset for production training.
    parquet_path = _PROJECT_ROOT / "ml" / "data" / "processed" / "features_autoencoder_v3.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"features_autoencoder_v2.parquet not found: {parquet_path}\n"
            "Run ml/features/feature_engineering.py first."
        )

    # ── Load / build cache ──────────────────────────────────────────────────
    if args.reuse_cache:
        log.info("--reuse-cache: loading existing binary cache ...")
        train, val, test_X, test_y, meta = load_cache()
    else:
        df_train, df_val, df_test, config = load_parquet_and_split(parquet_path)
        validate_feature_columns(df_train, "train")
        validate_feature_columns(df_val,   "validation")
        build_cache(df_train, df_val, df_test, config)
        del df_train, df_val, df_test
        gc.collect()
        train, val, test_X, test_y, meta = load_cache()

    log.info(
        f"Cache: train={len(train):,} | val={len(val):,} | test={len(test_X):,} "
        f"({int(test_y.sum()):,} fraud)"
    )
    _log_memory("after cache load")

    # ── Scaler ──────────────────────────────────────────────────────────────
    log.info("Fitting scaler from training-window legitimate transactions only.")
    mean, std = fit_scaler(train, args.batch_size)
    std[std == 0] = 1.0   # guard against zero-variance features

    # ── Train ───────────────────────────────────────────────────────────────
    model, history, device = train_model(train, val, mean, std, args)
    _log_memory("after training")

    # ── Per-feature error normalization (fit on val, never test) ────────────
    log.info("Fitting per-feature error-normalization statistics on validation set ...")
    feat_err_mean, feat_err_std = compute_feature_error_stats(
        model, val, mean, std, device, args.batch_size
    )
    error_norm = {"mean": feat_err_mean, "std": feat_err_std}

    # ── Score validation + OOT test ─────────────────────────────────────────
    log.info("Scoring validation and untouched OOT test windows ...")
    val_scores  = score_normalized(model, val,    mean, std, error_norm, device, args.batch_size)
    test_scores = score_normalized(model, test_X, mean, std, error_norm, device, args.batch_size)

    # Quick sanity check
    cold_count = int(cold_start_mask(test_X).sum())
    log.info(
        f"Cold-start guard applied to {cold_count:,} test rows "
        f"({cold_count/len(test_X)*100:.2f}% of OOT test) — scored 0.0"
    )

    # ── Save artifacts ───────────────────────────────────────────────────────
    save_outputs(history, mean, std, val_scores, error_norm, meta)

    # ── AE Signal Extraction (Phase 2 prep) ─────────────────────────────────
    if not args.skip_ae_signals:
        _section("Extracting 38 AE signals for Phase-2 LGBM integration")
        log.info("Loading full features_autoencoder.parquet for signal extraction ...")
        df_full = pd.read_parquet(
            parquet_path,
            columns=FEATURE_NAMES + ["timestamp", "is_laundering"],
            engine="pyarrow",
        )
        df_full = df_full.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

        signal_df = extract_ae_signals(
            model      = model,
            df_source  = df_full,
            mean       = mean,
            std        = std,
            error_norm = error_norm,
            device     = device,
            batch_size = args.batch_size,
        )

        # Attach timestamp + label for future join with features_lgbm.parquet
        signal_df["timestamp"]     = df_full["timestamp"].values
        signal_df["is_laundering"] = df_full["is_laundering"].values

        ae_signals_path = _PROJECT_ROOT / "ml" / "data" / "processed" / "features_ae_signals_final.parquet"
        signal_df.to_parquet(ae_signals_path, index=False)
        log.info(f"AE signals saved -> {ae_signals_path}  ({signal_df.shape[1]} cols, {len(signal_df):,} rows)")
        _log_memory("after ae signal extraction")
    else:
        log.info("--skip-ae-signals: skipping Phase-2 AE signal extraction.")

    _section("TRAINING COMPLETE — final")
    log.info(f"Train rows:      {len(train):,} (legitimate only)")
    log.info(f"Validation rows: {len(val):,}   (legitimate only)")
    log.info(f"OOT test rows:   {len(test_X):,} (all labels)")
    log.info(f"Model: {MODEL_DIR / 'autoencoder_final.pt'}")
    log.info(f"Next step: python -X utf8 ml/evaluation/evaluate_autoencoder_final.py")


if __name__ == "__main__":
    main()
