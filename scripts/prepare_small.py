"""
scripts/prepare_small.py
========================
MuleGuard.AI — HI-Small dataset preparation (one-shot)

Steps performed:
  1. CSV -> cleaned Parquet  (same transforms as HI-Medium data_prep.py)
  2. 13 TGN edge features    (delegates to build_tgn_edge_features.py)

Output files:
  ml/data/processed/transactions_clean_small.parquet
  ml/features/tgn_edge_features_small.parquet

Usage:
  python scripts/prepare_small.py

Then train TGN with:
  python ml/training/tgn_train.py ^
      --transactions-path ml/data/processed/transactions_clean_small.parquet ^
      --edge-features-path ml/features/tgn_edge_features_small.parquet ^
      --model-out ml/models/tgn_small_best.pt ^
      --epochs 15
"""

from __future__ import annotations
import os, sys, logging, time, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("prepare_small")

RAW_CSV       = ROOT / "ml" / "data" / "raw" / "HI-Small_Trans.csv"
CLEAN_PARQUET = ROOT / "ml" / "data" / "processed" / "transactions_clean_small.parquet"
EDGE_PARQUET  = ROOT / "ml" / "features" / "tgn_edge_features_small.parquet"
SPEC_PATH     = ROOT / "ml" / "features" / "tgn_edge_feature_spec.json"

CURRENCY_TO_USD = {
    "US Dollar": 1.0, "Euro": 1.00, "UK Pound": 1.12, "Bitcoin": 19500.0,
    "Swiss Franc": 1.03, "Yen": 0.0069, "Yuan": 0.14, "Ruble": 0.016,
    "Australian Dollar": 0.65, "Canadian Dollar": 0.74, "Saudi Riyal": 0.27,
    "Mexican Peso": 0.05, "Brazil Real": 0.19, "Rupee": 0.012, "Shekel": 0.29,
}


def step1_clean() -> None:
    if CLEAN_PARQUET.exists():
        log.info("SKIP Step 1 -- %s already exists.", CLEAN_PARQUET.name)
        return
    log.info("Step 1: Loading %s (%.0f MB) ...", RAW_CSV.name, RAW_CSV.stat().st_size / 1e6)
    t0 = time.time()
    df = pd.read_csv(RAW_CSV)
    log.info("  %s rows loaded.", f"{len(df):,}")

    df = df.rename(columns={
        "Timestamp": "timestamp", "From Bank": "from_bank", "Account": "from_account",
        "To Bank": "to_bank", "Account.1": "to_account",
        "Amount Received": "amount_received_raw", "Receiving Currency": "receiving_currency",
        "Amount Paid": "amount_paid_raw", "Payment Currency": "payment_currency",
        "Payment Format": "payment_format", "Is Laundering": "is_laundering",
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M").dt.tz_localize("UTC")
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["amount_paid"]     = df["amount_paid_raw"]     * df["payment_currency"].map(CURRENCY_TO_USD).fillna(1.0)
    df["amount_received"] = df["amount_received_raw"] * df["receiving_currency"].map(CURRENCY_TO_USD).fillna(1.0)
    df = df.drop(columns=["amount_paid_raw", "amount_received_raw"])

    assert df["timestamp"].isna().sum() == 0
    assert df["is_laundering"].isin([0, 1]).all()

    fraud_rate = df["is_laundering"].mean()
    log.info("  PASS -- %s rows | fraud=%.4f%% | span %s -> %s",
             f"{len(df):,}", fraud_rate*100,
             df["timestamp"].min().date(), df["timestamp"].max().date())

    CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PARQUET, index=False)
    log.info("  Saved -> %s  (%.1fs)", CLEAN_PARQUET.name, time.time() - t0)


def step2_edge_features() -> None:
    if EDGE_PARQUET.exists():
        log.info("SKIP Step 2 -- %s already exists.", EDGE_PARQUET.name)
        return
    log.info("Step 2: Building 13 TGN edge features ...")
    t0 = time.time()
    result = subprocess.run([
        sys.executable,
        str(ROOT / "ml" / "features" / "build_tgn_edge_features.py"),
        "--data-path",   str(CLEAN_PARQUET),
        "--output-path", str(EDGE_PARQUET),
        "--spec-path",   str(SPEC_PATH),
    ], cwd=str(ROOT))
    if result.returncode != 0:
        log.error("Edge feature builder FAILED (returncode=%d)", result.returncode)
        sys.exit(1)
    log.info("  Step 2 done in %.1fs", time.time() - t0)


def print_summary() -> None:
    df = pd.read_parquet(CLEAN_PARQUET, columns=["is_laundering", "timestamp"])
    n = len(df)
    fraud_pct = df["is_laundering"].mean() * 100
    ratio = n / 28_869_479
    est_min = 44 * ratio
    print()
    print("=" * 62)
    print("  HI-Small Dataset Ready")
    print("=" * 62)
    print(f"  Transactions : {n:,}  ({fraud_pct:.3f}% fraud)")
    print(f"  Est. per epoch on CPU: ~{est_min:.1f} min  (HI-Medium = ~44 min)")
    print()
    print("  Run TGN on HI-Small:")
    print("  python ml/training/tgn_train.py ^")
    print("      --transactions-path ml/data/processed/transactions_clean_small.parquet ^")
    print("      --edge-features-path ml/features/tgn_edge_features_small.parquet ^")
    print("      --model-out ml/models/tgn_small_best.pt ^")
    print("      --epochs 15")
    print()
    print("  Run TGN on HI-Medium (when ready):")
    print("  python ml/training/tgn_train.py")
    print("=" * 62)


if __name__ == "__main__":
    log.info("HI-Small data preparation ...")
    step1_clean()
    step2_edge_features()
    print_summary()
