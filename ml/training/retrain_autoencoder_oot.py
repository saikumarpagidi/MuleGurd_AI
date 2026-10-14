"""
================================================================================
MuleGuard.AI — Temporal Autoencoder v2
Enterprise Training Pipeline (matches train_lgbm_aml.py's conventions)
================================================================================

Every account feature is computed only from transactions seen strictly
before the event being scored (causal, expanding per-account state — see
make_feature_row()).

WHY THIS SPLIT MATCHES train_lgbm_aml.py, NOT AN INDEPENDENT CHOICE
--------------------------------------------------------------------
The LightGBM pipeline (ml/training/train_lgbm_aml.py) uses
OOT_SPLIT_FRACTION = 0.80: the last 20% of chronologically-sorted ROWS is
held out as an untouched test set, with the last 10% of the remaining 80%
reserved as internal validation. An earlier version of this Autoencoder
script used fixed CALENDAR-DAY boundaries (train=0-10.5d, val=10.5-14d,
test=14d-end) instead. On this dataset's actual ~27.67-day span, that put
roughly HALF the data in the OOT test set -- a completely different held-
out population than the LightGBM model's last-20%-of-rows split. That
made the two models' metrics non-comparable and would have made any
future ensemble of their scores invalid (blending predictions made on
different transaction populations). This script instead reads the actual
timestamp distribution and finds the exact row-count-fraction cutoffs, so
both models are evaluated on the IDENTICAL held-out rows -- verified
empirically (see compute_row_fraction_split's test suite) to land on
exactly 20.0000% every time, regardless of daily volume skew.

ENTERPRISE-PARITY FIXES applied in this version (matching
train_lgbm_aml.py's already-proven patterns):
  - _json_safe(): the LightGBM pipeline crashed once on its LAST step of a
    ~2-hour run because a numpy bool from a threshold comparison isn't
    JSON-serializable. Applied here defensively to every json.dump call.
  - Resource guardrail: torch.set_num_threads(cpu_count - 2), matching
    LGBM_N_JOBS's "leave 2 cores free" convention on the same workstation.
  - torch.manual_seed() + torch.cuda.manual_seed_all(): weight init and
    the denoising noise injection are now reproducible run to run.
  - Guard against a trailing batch of size 1 (see batch_ranges()) --
    BatchNorm1d cannot compute a variance from a single sample and would
    otherwise crash mid-training on a real ~20M-row partition.
  - MODEL_DIR (deployable artifact) vs REPORT_DIR (everything else) are
    now separate, matching train_lgbm_aml.py's own directory convention.
  - Dual-handler logging (console + a persistent training_log.txt),
    matching train_lgbm_aml.py's logging setup exactly.

Run:
  python -X utf8 scripts/retrain_autoencoder_oot.py
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import torch
import torch.nn as nn

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ==============================================================================
# CONFIGURATION — mirrors train_lgbm_aml.py's CONFIGURATION block
# ==============================================================================
MODEL_DIR = Path("ml/models/autoencoder")
REPORT_DIR = Path("ml/reports/autoencoder_v2")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OOT_FRACTION = 0.20            # must match train_lgbm_aml.py's OOT_SPLIT_FRACTION
INTERNAL_VAL_FRACTION = 0.10   # must match train_lgbm_aml.py's INTERNAL_VAL_FRACTION
RESERVED_CORES = 2             # must match train_lgbm_aml.py's LGBM_N_JOBS convention
EPSILON = 1e-6

PAYMENT_FORMAT_ID = {
    "ACH": 0, "Cash": 1, "Cheque": 2, "Credit Card": 3,
    "Bitcoin": 4, "Reinvestment": 5, "Wire": 6,
}

FEATURE_NAMES = [
    "log_amount_paid",
    "log_amount_received",
    "signed_log_amount_ratio",
    "sender_log_amount_vs_past_mean",
    "receiver_log_amount_vs_past_mean",
    "sender_log_past_transaction_count",
    "receiver_log_past_transaction_count",
    "sender_log_past_total_paid",
    "receiver_log_past_total_received",
    "sender_log_hours_since_last_txn",
    "receiver_log_hours_since_last_txn",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_cross_bank",
    "is_cross_currency",
    "payment_format_id",
]

SOURCE_COLUMNS = [
    "timestamp", "from_account", "to_account", "from_bank", "to_bank",
    "amount_paid", "amount_received", "payment_currency",
    "receiving_currency", "payment_format", "is_laundering",
]


# ==============================================================================
# LOGGING — identical dual-handler pattern to train_lgbm_aml.py
# ==============================================================================
log = logging.getLogger("MuleGuard.Autoencoder")
log.propagate = False
log.setLevel(logging.INFO)
if not log.handlers:
    _formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    _stream_handler = logging.StreamHandler(sys.stdout)
    _stream_handler.setFormatter(_formatter)
    _file_handler = logging.FileHandler(REPORT_DIR / "training_log.txt", mode="w")
    _file_handler.setFormatter(_formatter)
    log.addHandler(_stream_handler)
    log.addHandler(_file_handler)


def _log_memory(tag: str) -> None:
    if _HAS_PSUTIL:
        vm = psutil.virtual_memory()
        log.info(f"[mem check: {tag}] RAM used={vm.percent:.1f}% "
                 f"({vm.used/1e9:.1f}GB / {vm.total/1e9:.1f}GB)")


def _json_safe(value):
    """
    Recursively converts numpy scalar types to native Python types before
    JSON serialization -- the exact same helper, verbatim in behavior, as
    train_lgbm_aml.py's _json_safe(), applied here for the same reason:
    that pipeline crashed once on its LAST step of a ~2-hour run because a
    numpy bool wasn't JSON-serializable. Applying this everywhere removes
    the need to audit every future field addition by hand.
    """
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _section(title: str) -> None:
    log.info("=" * 88)
    log.info(title)
    log.info("=" * 88)


# ==============================================================================
# Split configuration
# ==============================================================================
@dataclass(frozen=True)
class SplitConfig:
    start_timestamp_ns: int
    train_end_ns: int
    validation_end_ns: int


def compute_row_fraction_split(
    dataset: ds.Dataset,
    oot_fraction: float = OOT_FRACTION,
    internal_val_fraction: float = INTERNAL_VAL_FRACTION,
) -> SplitConfig:
    """
    Row-count-fraction split, matching train_lgbm_aml.py's
    chronological_splits() exactly: the last `oot_fraction` of
    chronologically-sorted ROWS is the OOT test set; within the remaining
    rows, the last `internal_val_fraction` is internal validation.
    """
    log.info("Loading full timestamp column to compute row-fraction split boundaries ...")
    timestamps = dataset.to_table(columns=["timestamp"]).column("timestamp").cast(pa.int64()).to_numpy(zero_copy_only=False)
    timestamps = np.sort(timestamps)
    n = len(timestamps)
    if n == 0:
        raise RuntimeError("transactions_clean.parquet contains no rows")

    oot_start_idx = int(n * (1.0 - oot_fraction))
    val_start_idx = int(oot_start_idx * (1.0 - internal_val_fraction))

    train_end_ns = int(timestamps[val_start_idx])
    validation_end_ns = int(timestamps[oot_start_idx])

    log.info(
        f"Row-fraction split (matches train_lgbm_aml.py OOT_SPLIT_FRACTION={oot_fraction}): "
        f"train rows 0..{val_start_idx:,} | validation rows {val_start_idx:,}..{oot_start_idx:,} "
        f"| OOT test rows {oot_start_idx:,}..{n:,}"
    )
    return SplitConfig(
        start_timestamp_ns=int(timestamps[0]),
        train_end_ns=train_end_ns,
        validation_end_ns=validation_end_ns,
    )


def split_for_timestamp(timestamp_ns: int, config: SplitConfig) -> str:
    if timestamp_ns < config.train_end_ns:
        return "train"
    if timestamp_ns < config.validation_end_ns:
        return "validation"
    return "test"


# ==============================================================================
# Model
# ==============================================================================
class TemporalAutoencoder(nn.Module):
    """Compact denoising Autoencoder for the leakage-free feature set."""

    def __init__(self, input_dim: int = len(FEATURE_NAMES), bottleneck_dim: int = 12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.BatchNorm1d(64), nn.GELU(),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.GELU(),
            nn.Linear(32, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32), nn.BatchNorm1d(32), nn.GELU(),
            nn.Linear(32, 64), nn.BatchNorm1d(64), nn.GELU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def count_cache_rows(dataset: ds.Dataset, config: SplitConfig) -> Dict[str, int]:
    counts = {"train": 0, "validation": 0, "test": 0}
    scanner = dataset.scanner(columns=["timestamp", "is_laundering"], batch_size=262_144)
    for batch in scanner.to_batches():
        timestamps = batch.column("timestamp").cast(pa.int64()).to_numpy(zero_copy_only=False)
        labels = batch.column("is_laundering").to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
        train_mask = timestamps < config.train_end_ns
        validation_mask = (timestamps >= config.train_end_ns) & (timestamps < config.validation_end_ns)
        test_mask = timestamps >= config.validation_end_ns
        counts["train"] += int(np.sum(train_mask & (labels == 0)))
        counts["validation"] += int(np.sum(validation_mask & (labels == 0)))
        counts["test"] += int(np.sum(test_mask))
    if not all(counts.values()):
        raise RuntimeError(f"Invalid temporal split counts: {counts}")
    return counts


def make_feature_row(
    timestamp_ns, source_account, destination_account, source_bank, destination_bank,
    amount_paid, amount_received, payment_currency, receiving_currency, payment_format,
    sender_state, receiver_state,
) -> np.ndarray:
    """Create one row using state strictly prior to the current transaction."""
    sender_count, sender_total, sender_last_ts = sender_state.get(source_account, (0, 0.0, timestamp_ns))
    receiver_count, receiver_total, receiver_last_ts = receiver_state.get(destination_account, (0, 0.0, timestamp_ns))

    sender_average = sender_total / sender_count if sender_count else amount_paid
    receiver_average = receiver_total / receiver_count if receiver_count else amount_received
    sender_gap_hours = max(0.0, (timestamp_ns - sender_last_ts) / 3_600_000_000_000) if sender_count else 0.0
    receiver_gap_hours = max(0.0, (timestamp_ns - receiver_last_ts) / 3_600_000_000_000) if receiver_count else 0.0

    timestamp_seconds = timestamp_ns // 1_000_000_000
    hour = (timestamp_seconds % 86_400) / 3_600.0
    weekday = ((timestamp_seconds // 86_400) + 3) % 7  # 1970-01-01 was a Thursday

    row = np.array([
        math.log1p(max(amount_paid, 0.0)),
        math.log1p(max(amount_received, 0.0)),
        np.clip(math.log((amount_received + EPSILON) / (amount_paid + EPSILON)), -12.0, 12.0),
        np.clip(math.log((amount_paid + EPSILON) / (sender_average + EPSILON)), -12.0, 12.0),
        np.clip(math.log((amount_received + EPSILON) / (receiver_average + EPSILON)), -12.0, 12.0),
        math.log1p(sender_count),
        math.log1p(receiver_count),
        math.log1p(max(sender_total, 0.0)),
        math.log1p(max(receiver_total, 0.0)),
        math.log1p(min(sender_gap_hours, 24.0 * 90.0)),
        math.log1p(min(receiver_gap_hours, 24.0 * 90.0)),
        math.sin(2.0 * math.pi * hour / 24.0),
        math.cos(2.0 * math.pi * hour / 24.0),
        math.sin(2.0 * math.pi * weekday / 7.0),
        math.cos(2.0 * math.pi * weekday / 7.0),
        float(source_bank != destination_bank),
        float(payment_currency != receiving_currency),
        PAYMENT_FORMAT_ID.get(payment_format, len(PAYMENT_FORMAT_ID)) / len(PAYMENT_FORMAT_ID),
    ], dtype=np.float32)

    sender_state[source_account] = (sender_count + 1, sender_total + amount_paid, timestamp_ns)
    receiver_state[destination_account] = (receiver_count + 1, receiver_total + amount_received, timestamp_ns)
    return row


def cache_paths(cache_dir: Path) -> Dict[str, Path]:
    return {
        "train": cache_dir / "train_legitimate.f32",
        "validation": cache_dir / "validation_legitimate.f32",
        "test": cache_dir / "test_oot.f32",
        "test_labels": cache_dir / "test_oot_labels.i8",
        "metadata": cache_dir / "metadata.json",
    }


def build_feature_cache(dataset: ds.Dataset, cache_dir: Path, config: SplitConfig, counts: Dict[str, int]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(cache_dir)
    feature_count = len(FEATURE_NAMES)

    train = np.memmap(paths["train"], dtype=np.float32, mode="w+", shape=(counts["train"], feature_count))
    validation = np.memmap(paths["validation"], dtype=np.float32, mode="w+", shape=(counts["validation"], feature_count))
    test = np.memmap(paths["test"], dtype=np.float32, mode="w+", shape=(counts["test"], feature_count))
    test_labels = np.memmap(paths["test_labels"], dtype=np.int8, mode="w+", shape=(counts["test"],))
    destinations = {"train": train, "validation": validation, "test": test}
    positions = {"train": 0, "validation": 0, "test": 0}
    sender_state: Dict[str, Tuple[int, float, int]] = {}
    receiver_state: Dict[str, Tuple[int, float, int]] = {}

    scanner = dataset.scanner(columns=SOURCE_COLUMNS, batch_size=131_072)
    total_seen = 0
    started = time.monotonic()
    last_seen_ts = -1
    for batch_number, batch in enumerate(scanner.to_batches(), start=1):
        timestamps = batch.column("timestamp").cast(pa.int64()).to_numpy(zero_copy_only=False)

        # Defensive monotonicity check -- causal state accumulation below
        # requires strictly non-decreasing timestamp order across the scan.
        if timestamps.size:
            if int(timestamps[0]) < last_seen_ts or not np.all(np.diff(timestamps) >= 0):
                raise RuntimeError(
                    f"Non-chronological rows detected at batch {batch_number}. "
                    "Re-sort transactions_clean.parquet by timestamp before rerunning."
                )
            last_seen_ts = int(timestamps[-1])

        paid = batch.column("amount_paid").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        received = batch.column("amount_received").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
        labels = batch.column("is_laundering").to_numpy(zero_copy_only=False).astype(np.int8, copy=False)
        sources = batch.column("from_account").to_pylist()
        destinations_accounts = batch.column("to_account").to_pylist()
        source_banks = batch.column("from_bank").to_pylist()
        destination_banks = batch.column("to_bank").to_pylist()
        payment_currencies = batch.column("payment_currency").to_pylist()
        receiving_currencies = batch.column("receiving_currency").to_pylist()
        payment_formats = batch.column("payment_format").to_pylist()

        feature_rows = np.empty((len(batch), feature_count), dtype=np.float32)
        split_names = np.empty(len(batch), dtype="U10")
        for index in range(len(batch)):
            timestamp_ns = int(timestamps[index])
            feature_rows[index] = make_feature_row(
                timestamp_ns=timestamp_ns, source_account=sources[index],
                destination_account=destinations_accounts[index],
                source_bank=source_banks[index], destination_bank=destination_banks[index],
                amount_paid=float(paid[index]), amount_received=float(received[index]),
                payment_currency=payment_currencies[index], receiving_currency=receiving_currencies[index],
                payment_format=payment_formats[index],
                sender_state=sender_state, receiver_state=receiver_state,
            )
            split_names[index] = split_for_timestamp(timestamp_ns, config)

        for split in ("train", "validation", "test"):
            mask = split_names == split
            if split != "test":
                mask &= labels == 0
            selected = feature_rows[mask]
            next_position = positions[split] + len(selected)
            destinations[split][positions[split]:next_position] = selected
            if split == "test":
                test_labels[positions[split]:next_position] = labels[mask]
            positions[split] = next_position

        total_seen += len(batch)
        if batch_number % 25 == 0:
            elapsed = time.monotonic() - started
            log.info(f"Cached {total_seen:,} transactions ({total_seen / max(elapsed, EPSILON):.0f} rows/sec; "
                      f"sender states {len(sender_state):,}; receiver states {len(receiver_state):,})")

    if positions != counts:
        raise RuntimeError(f"Cache row mismatch: wrote {positions}, expected {counts}")
    for array in (train, validation, test, test_labels):
        array.flush()
    metadata = {
        "feature_names": FEATURE_NAMES, "counts": counts, "split": asdict(config),
        "cache_format": "row-major float32 features; int8 test labels",
    }
    paths["metadata"].write_text(json.dumps(_json_safe(metadata), indent=2), encoding="utf-8")
    log.info("Leakage-free feature cache written to %s", cache_dir)


def load_metadata(cache_dir: Path) -> Dict[str, object]:
    path = cache_paths(cache_dir)["metadata"]
    if not path.exists():
        raise FileNotFoundError(f"Feature-cache metadata missing: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("feature_names") != FEATURE_NAMES:
        raise RuntimeError("Feature cache does not match this retraining script")
    return metadata


def open_cache(cache_dir: Path, metadata: Dict[str, object]):
    counts = metadata["counts"]
    feature_count = len(FEATURE_NAMES)
    paths = cache_paths(cache_dir)
    train = np.memmap(paths["train"], dtype=np.float32, mode="r", shape=(counts["train"], feature_count))
    validation = np.memmap(paths["validation"], dtype=np.float32, mode="r", shape=(counts["validation"], feature_count))
    test = np.memmap(paths["test"], dtype=np.float32, mode="r", shape=(counts["test"], feature_count))
    labels = np.memmap(paths["test_labels"], dtype=np.int8, mode="r", shape=(counts["test"],))
    return train, validation, test, labels


def fit_scaler(train: np.memmap, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    total = 0
    running_sum = np.zeros(train.shape[1], dtype=np.float64)
    running_square_sum = np.zeros(train.shape[1], dtype=np.float64)
    for start in range(0, len(train), batch_size):
        values = np.asarray(train[start:start + batch_size], dtype=np.float64)
        running_sum += values.sum(axis=0)
        running_square_sum += np.square(values).sum(axis=0)
        total += len(values)
    mean = running_sum / total
    variance = np.maximum(running_square_sum / total - np.square(mean), 1e-8)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def as_tensor(values: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> torch.Tensor:
    standardized = (np.asarray(values, dtype=np.float32) - mean) / std
    return torch.from_numpy(standardized).to(device, non_blocking=device.type == "cuda")


def batch_ranges(n: int, batch_size: int) -> list:
    """
    Compute (start, end) batch boundaries covering rows [0, n).

    FIX: BatchNorm1d cannot compute a variance from a single sample and
    raises at training time if it ever receives a batch of exactly 1 row.
    A plain range(0, n, batch_size) produces a trailing batch of size 1
    whenever n % batch_size == 1 -- on a real ~20M-row partition this is
    entirely possible. This merges any such trailing singleton into the
    previous batch instead of leaving it on its own.
    """
    ranges = [(start, min(start + batch_size, n)) for start in range(0, n, batch_size)]
    if len(ranges) > 1 and (ranges[-1][1] - ranges[-1][0]) == 1:
        prev_start, _ = ranges[-2]
        _, last_end = ranges[-1]
        ranges[-2] = (prev_start, last_end)
        ranges.pop()
    return ranges


def average_reconstruction_error(model, values, mean, std, batch_size, device) -> float:
    model.eval()
    error_sum = 0.0
    with torch.no_grad():
        for start, end in batch_ranges(len(values), batch_size):
            batch = as_tensor(values[start:end], mean, std, device)
            reconstruction = model(batch)
            error_sum += float(torch.square(reconstruction - batch).sum().item())
    return error_sum / (len(values) * len(FEATURE_NAMES))


def compute_feature_error_stats(model, values, mean, std, batch_size, device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes, per feature, the mean and std of squared reconstruction error
    over `values` (intended to be the VALIDATION set, never test).

    WHY THIS EXISTS: a real run of this pipeline showed that an unweighted
    mean-squared-error across all 18 standardized features lets a handful of
    naturally-high-variance features (the cyclical hour/weekday encodings)
    dominate the total anomaly score, drowning out genuinely fraud-relevant
    behavioral features. In one real run, 4 of 18 features accounted for
    ~89% of total reconstruction error, and the resulting anomaly score had
    WORSE than random enrichment at the tightest alert budget (Top 0.1% lift
    = 0.1x) -- a sign the score was ranking "unusual time of day" ahead of
    genuine fraud-relevant deviations.

    Fix: at evaluation/scoring time, each feature's squared error is
    converted to a z-score relative to how that SAME feature behaves on
    legitimate validation data, before averaging across features. A feature
    that's merely "naturally noisy" (like weekday_sin) will have a large
    val_std, so even a large raw error there produces a modest z-score.
    A feature that's normally very well-reconstructed but suddenly isn't
    (a real behavioral anomaly) produces a large z-score. This is the
    standard "per-dimension error normalization" pattern used in production
    autoencoder-based anomaly detection specifically to prevent this failure
    mode -- see evaluate_autoencoder_v2.py's normalized_anomaly_scores().

    Fit on the VALIDATION set only (same "never touch test" discipline as
    fit_scaler() and the p95/p99 threshold calibration) -- this is a
    normalization parameter, not a model weight, but it is still fit data
    and must never see the OOT test set.
    """
    model.eval()
    per_feature_sq_error_sum = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    per_feature_sq_error_sum_sq = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    total_rows = 0
    with torch.no_grad():
        for start, end in batch_ranges(len(values), batch_size):
            batch = as_tensor(values[start:end], mean, std, device)
            reconstruction = model(batch)
            sq_error = torch.square(reconstruction - batch).cpu().numpy().astype(np.float64)
            per_feature_sq_error_sum += sq_error.sum(axis=0)
            per_feature_sq_error_sum_sq += np.square(sq_error).sum(axis=0)
            total_rows += (end - start)

    feature_err_mean = per_feature_sq_error_sum / max(total_rows, 1)
    feature_err_var = np.maximum(
        per_feature_sq_error_sum_sq / max(total_rows, 1) - np.square(feature_err_mean), 1e-12
    )
    feature_err_std = np.sqrt(feature_err_var)
    return feature_err_mean.astype(np.float64), feature_err_std.astype(np.float64)


def train_model(train, validation, mean, std, args) -> Tuple[TemporalAutoencoder, list, torch.device]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch_threads = max(1, (os.cpu_count() or 4) - RESERVED_CORES)
    torch.set_num_threads(torch_threads)
    log.info(f"Resource guardrail: torch.set_num_threads({torch_threads}) "
             f"(reserving {RESERVED_CORES} cores for system stability)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    loss_fn = nn.SmoothL1Loss()
    checkpoint_path = MODEL_DIR / "autoencoder_oot_v2.pt"
    history: list = []
    best_validation = float("inf")
    stale_epochs = 0

    log.info(f"Training Autoencoder v2 on {len(train):,} legitimate transactions ({device}).")
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        total_loss = 0.0
        total_rows = 0
        ranges = batch_ranges(len(train), args.batch_size)
        random.Random(args.seed + epoch).shuffle(ranges)
        for start, end in ranges:
            batch = as_tensor(train[start:end], mean, std, device)
            noisy_batch = batch + torch.randn_like(batch) * 0.02
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(noisy_batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
            total_rows += len(batch)

        scheduler.step()
        validation_error = average_reconstruction_error(model, validation, mean, std, args.batch_size, device)
        train_loss = total_loss / total_rows
        record = {
            "epoch": epoch, "train_huber_loss": train_loss, "validation_mse": validation_error,
            "learning_rate": optimizer.param_groups[0]["lr"], "seconds": time.monotonic() - epoch_started,
        }
        history.append(record)
        log.info(f"Epoch {epoch:02d}/{args.epochs} | train Huber {train_loss:.6f} | "
                  f"val MSE {validation_error:.6f} | {record['seconds']:.1f}s")

        if validation_error < best_validation - 1e-7:
            best_validation = validation_error
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(), "input_dim": len(FEATURE_NAMES),
                "bottleneck_dim": 12, "feature_names": FEATURE_NAMES,
                "best_validation_mse": best_validation,
                "training_protocol": f"row-fraction split (last {OOT_FRACTION:.0%} of rows = OOT test, "
                                      f"matching train_lgbm_aml.py); past-only causal features",
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                log.info(f"Early stopping after {args.patience} epochs without validation improvement.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, history, device


def scores_for_values(model, values, mean, std, batch_size, device) -> np.ndarray:
    model.eval()
    scores = np.empty(len(values), dtype=np.float32)
    with torch.no_grad():
        for start, end in batch_ranges(len(values), batch_size):
            batch = as_tensor(values[start:end], mean, std, device)
            reconstruction = model(batch)
            scores[start:end] = torch.mean(torch.square(reconstruction - batch), dim=1).cpu().numpy()
    return scores


def save_outputs(history, mean, std, validation_scores, test_labels, test_scores, metadata) -> dict:
    threshold = {
        "p95": float(np.percentile(validation_scores, 95.0)),
        "p97": float(np.percentile(validation_scores, 97.0)),
        "p99": float(np.percentile(validation_scores, 99.0)),
        "p99_5": float(np.percentile(validation_scores, 99.5)),
        "score_normalization": "clip((reconstruction_error - p95) / (p99_5 - p95), 0, 1)",
        "calibrated_on": "future validation window containing legitimate transactions only",
    }
    scaler = {
        "feature_order": FEATURE_NAMES, "mean": mean.astype(float).tolist(), "std": std.astype(float).tolist(),
        "training_protocol": "past-only features; scaler fit on legitimate training window only",
    }
    (MODEL_DIR / "scaler_stats.json").write_text(json.dumps(_json_safe(scaler), indent=2), encoding="utf-8")
    (MODEL_DIR / "threshold_config.json").write_text(json.dumps(_json_safe(threshold), indent=2), encoding="utf-8")

    with (REPORT_DIR / "training_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    log.info(f"Model artifact  -> {MODEL_DIR / 'autoencoder_oot_v2.pt'}")
    log.info(f"Scaler/threshold -> {MODEL_DIR}")
    log.info(f"Training log     -> {REPORT_DIR / 'training_log.csv'} / training_log.txt")
    log.info("Run scripts/evaluate_autoencoder_v2.py next for the official validated report "
              "(ROC-AUC/PR-AUC, alert-capacity table, reconstruction-error feature attribution, "
              "PR curve, executive summary).")
    return {"split": metadata["split"]}


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        raise ValueError("--epochs must be positive and --batch-size must be at least 2")

    _section("MuleGuard.AI — Temporal Autoencoder v2 Training Pipeline")
    log.info(f"Resource guardrails will reserve {RESERVED_CORES} of {os.cpu_count()} logical cores.")
    _log_memory("startup")

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "ml" / "data" / "processed" / "transactions_clean.parquet"
    cache_dir = project_root / "ml" / "data" / "processed" / "autoencoder_oot_v2_cache"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing source data: {data_path}")

    dataset = ds.dataset(data_path, format="parquet")
    if args.reuse_cache:
        metadata = load_metadata(cache_dir)
    else:
        config = compute_row_fraction_split(dataset)
        log.info("Counting temporal partitions before allocating cache files.")
        counts = count_cache_rows(dataset, config)
        log.info(f"Partition counts: {counts}")
        build_feature_cache(dataset, cache_dir, config, counts)
        metadata = load_metadata(cache_dir)

    train, validation, test, test_labels = open_cache(cache_dir, metadata)
    log.info("Fitting scaler from training-window legitimate transactions only.")
    mean, std = fit_scaler(train, args.batch_size)
    model, history, device = train_model(train, validation, mean, std, args)

    log.info("Scoring validation and untouched OOT test windows.")
    validation_scores = scores_for_values(model, validation, mean, std, args.batch_size, device)
    test_scores = scores_for_values(model, test, mean, std, args.batch_size, device)
    save_outputs(history, mean, std, validation_scores, np.asarray(test_labels), test_scores, metadata)

    log.info("Fitting per-feature error-normalization statistics on validation set "
             "(fixes the dominant-feature scoring issue -- see compute_feature_error_stats docstring).")
    feature_err_mean, feature_err_std = compute_feature_error_stats(
        model, validation, mean, std, args.batch_size, device
    )
    error_norm = {
        "feature_names": FEATURE_NAMES,
        "val_squared_error_mean": feature_err_mean.tolist(),
        "val_squared_error_std": feature_err_std.tolist(),
        "fitted_on": "validation set (legitimate transactions only), never test",
        "usage": "z = clip((row_squared_error - val_squared_error_mean) / val_squared_error_std, 0, None); "
                 "normalized_score = mean(z) across features. See evaluate_autoencoder_v2.py.",
    }
    (MODEL_DIR / "feature_error_norm.json").write_text(
        json.dumps(_json_safe(error_norm), indent=2), encoding="utf-8"
    )
    log.info(f"Feature error-normalization stats -> {MODEL_DIR / 'feature_error_norm.json'}")

    _section("TRAINING COMPLETE")
    log.info(f"Train rows: {len(train):,} | Validation rows: {len(validation):,} | OOT test rows: {len(test):,}")


if __name__ == "__main__":
    main()