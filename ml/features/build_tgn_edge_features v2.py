"""
ml/features/build_tgn_edge_features.py
=========================================
MuleGuard.AI — Sprint 5, Step 5.0 (prerequisite to tgn_train.py)

Builds the 13 dynamic, per-transaction edge features that feed the TGN's
message function. This is the "WHAT happened in this transaction" layer --
distinct from and complementary to the GDS "WHO is this account" layer
built in Step 4A (pagerank/community_id/in_scc_ring).

Without these features, the TGN's message function e_uv has nothing real to
encode per transaction -- it degrades to seeing only "an edge exists between
A and B," blind to amount, channel, or structuring behavior. See the Sprint
5 section of the Forward Plan for why this step is a hard prerequisite, not
an optional enhancement.

Features, grouped by build cost (see Forward Plan Step 5.0 for the full table):

Tier 1 -- cheap, single-row derivable:
    log_amount_paid, payment_format_encoded, is_cross_bank, is_cash_payment,
    is_bitcoin_transaction, structuring_threshold_flag

Tier 2 -- requires a per-account lookback window, NOT optional (protects the
    product's flagship "transit mule" detection use case):
    amount_ratio, amount_difference

Remaining (time/velocity, same tiering logic as the two groups above):
    hour_sin, hour_cos, weekday_sin, weekday_cos, is_cross_currency,
    log_time_since_last_txn

Design notes
------------
* Vectorized throughout (pandas groupby/shift/cumsum), following the same
  performance lesson learned in scripts/retrain_autoencoder_oot.py -- a
  32M-row Python for-loop is not viable at this scale.
* amount_ratio / amount_difference specifically detect the "transit mule"
  peeling-chain signature (money in, ~95-98% out immediately). They are
  computed here as: for each OUTBOUND transaction from an account, look at
  that account's most recent INBOUND transaction within a configurable
  lookback window (default 24h). This directly operationalizes "did this
  account just forward money it very recently received."
* Output is aligned to the same (from_account, to_account, timestamp, amount)
  composite key used as the SENT relationship's MERGE key in seed_neo4j.py,
  so these features can be joined back onto the Neo4j graph edges without
  ambiguity.

Usage
-----
    python ml/features/build_tgn_edge_features.py \\
        --data-path data/processed/transactions_clean.parquet \\
        --output-path ml/features/tgn_edge_features.parquet \\
        --spec-path ml/features/tgn_edge_feature_spec.json

Exit codes: 0 success, 1 input/config error.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level="INFO", format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
log = logging.getLogger("build_tgn_edge_features")

EPSILON = 1e-6
US_PER_HOUR = 3_600_000_000
SECONDS_PER_DAY = 86_400

PAYMENT_FORMAT_ID = {
    "ACH": 0, "Cash": 1, "Cheque": 2, "Credit Card": 3,
    "Bitcoin": 4, "Reinvestment": 5, "Wire": 6,
}

REQUIRED_COLUMNS = {
    "from_account", "to_account", "amount", "timestamp",
}
# Optional columns used if present, degrade gracefully if absent (see load_transactions).
OPTIONAL_COLUMNS = {"from_bank", "to_bank", "payment_currency", "receiving_currency", "payment_format"}

EDGE_FEATURE_NAMES = [
    "log_amount_paid",
    "payment_format_encoded",
    "is_cross_bank",
    "is_cash_payment",
    "is_bitcoin_transaction",
    "structuring_threshold_flag",
    "amount_ratio",
    "amount_difference",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_cross_currency",
]
assert len(EDGE_FEATURE_NAMES) == 13, "Forward Plan Step 5.0 specifies exactly 13 dynamic edge features"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-path", type=Path, default=Path("ml/data/processed/transactions_clean.parquet"))
    p.add_argument("--output-path", type=Path, default=Path("ml/features/tgn_edge_features.parquet"))
    p.add_argument("--spec-path", type=Path, default=Path("ml/features/tgn_edge_feature_spec.json"))
    p.add_argument("--structuring-min", type=float, default=8_000.0)
    p.add_argument("--structuring-max", type=float, default=9_999.99)
    p.add_argument("--transit-lookback-hours", type=float, default=24.0,
                    help="Window for matching an outbound txn to a recent inbound txn "
                         "when computing amount_ratio/amount_difference (transit-mule signature).")
    return p.parse_args(argv)


def to_epoch_us(ts: pd.Series) -> np.ndarray:
    if pd.api.types.is_datetime64_any_dtype(ts):
        unit = ts.values.dtype.str
        as_int = ts.astype("int64").to_numpy()
        if "ns" in unit:
            return as_int // 1_000
        if "us" in unit:
            return as_int
        if "ms" in unit:
            return as_int * 1_000
        return as_int * 1_000_000
    return ts.astype("int64").to_numpy()


def load_transactions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing source data: {path}")
    log.info("Reading %s ...", path)
    t0 = time.time()
    df = pd.read_parquet(path)
    if "amount_paid" in df.columns and "amount" not in df.columns:
        df = df.rename(columns={"amount_paid": "amount"})
    log.info("Loaded %s rows in %.1fs", f"{len(df):,}", time.time() - t0)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            log.warning("Optional column '%s' not found; the feature(s) depending on it will default to 0.", col)
            df[col] = None

    df["timestamp_us"] = to_epoch_us(df["timestamp"])
    if not df["timestamp_us"].is_monotonic_increasing:
        log.warning("Source data not chronologically ordered; sorting (required for causal, past-only lookback).")
        df = df.sort_values("timestamp_us", kind="mergesort").reset_index(drop=True)

    df["amount"] = df["amount"].astype("float64")
    # FIX #2: assign a guaranteed-unique integer row ID based on the sorted
    # position in this parquet. This becomes the join key in tgn_train.py,
    # replacing the (from_account, to_account, timestamp, amount) composite key
    # that caused many-to-many fan-out when the same account pair transacted the
    # same amount twice within the same minute (negative 'dropped rows' bug).
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    return df


def compute_tier1_features(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Single-row-derivable features -- no cross-row lookback required."""
    amount = df["amount"].to_numpy()
    payment_format = df["payment_format"].fillna("Unknown")
    from_bank = df["from_bank"]
    to_bank = df["to_bank"]
    payment_currency = df["payment_currency"]
    receiving_currency = df["receiving_currency"]

    features = pd.DataFrame(index=df.index)
    features["log_amount_paid"] = np.log1p(np.maximum(amount, 0.0))
    features["payment_format_encoded"] = payment_format.map(
        lambda fmt: PAYMENT_FORMAT_ID.get(fmt, len(PAYMENT_FORMAT_ID))
    ).astype(np.float64)
    features["is_cross_bank"] = (from_bank != to_bank).astype(np.float64)
    features["is_cash_payment"] = (payment_format == "Cash").astype(np.float64)
    features["is_bitcoin_transaction"] = (payment_format == "Bitcoin").astype(np.float64)
    features["structuring_threshold_flag"] = (
        (amount >= args.structuring_min) & (amount <= args.structuring_max)
    ).astype(np.float64)
    features["is_cross_currency"] = (payment_currency != receiving_currency).astype(np.float64)

    ts_us = df["timestamp_us"].to_numpy()
    seconds_of_day = (ts_us // 1_000_000) % SECONDS_PER_DAY
    hour = seconds_of_day / 3_600.0
    day_index = (ts_us // 1_000_000) // SECONDS_PER_DAY
    weekday = (day_index + 3) % 7  # epoch day 0 (1970-01-01) was a Thursday

    features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    features["weekday_sin"] = np.sin(2 * np.pi * weekday / 7.0)
    features["weekday_cos"] = np.cos(2 * np.pi * weekday / 7.0)
    return features


def compute_tier2_transit_features(df: pd.DataFrame, lookback_hours: float) -> pd.DataFrame:
    """amount_ratio / amount_difference -- the transit-mule "peeling chain" signature.

    For every OUTBOUND transaction from account X at time t, find X's most
    recent INBOUND transaction within [t - lookback_hours, t). If X received
    money and immediately forwarded ~all of it (amount_ratio ~= 1.0), that's
    the classic transit-mule pattern. If no qualifying inbound transaction
    exists in the window, amount_ratio defaults to 0.0 (no evidence of
    pass-through behavior) rather than a fabricated value.

    FIX #3: inbound side uses `amount_received` when present (the amount the
    receiving account actually received after FX conversion), falling back to
    `amount` (amount_paid) when not available. For cross-currency transit chains
    this gives the correct pass-through ratio -- using the sender's paid amount
    for the inbound side systematically understates or overstates the ratio by
    the FX rate, which matters for the product's flagship transit-mule detection.

    NOT optional per the Forward Plan -- cutting this removes the reason a
    per-transaction TGN exists over an account-level aggregate model.
    """
    t0 = time.time()
    n = len(df)
    ts_us = df["timestamp_us"].to_numpy()
    lookback_us = int(lookback_hours * US_PER_HOUR)

    # FIX #3: use amount_received for the inbound side when available.
    # amount_received = what this account actually received after FX conversion.
    # amount (amount_paid) = what the sender paid in their currency.
    # These differ for cross-currency transactions; using the wrong one gives
    # a systematically biased amount_ratio for the highest-value mule pattern.
    inbound_amount_col = "amount_received" if "amount_received" in df.columns else "amount"
    if inbound_amount_col == "amount":
        log.warning(
            "amount_received column not found -- falling back to amount_paid for inbound side "
            "of transit ratio. FX-chain transit ratios may be slightly biased."
        )

    # Inbound events, indexed per-account, sorted by time (already globally
    # sorted, so per-account order is preserved).
    inbound = df[["to_account", "timestamp_us", inbound_amount_col]].rename(
        columns={"to_account": "account", inbound_amount_col: "inbound_amount"}
    )
    # merge_asof requires a global sort by the "on" key AND by the "by" key
    # having monotonic "on" within each group -- already satisfied since df
    # is fully chronologically sorted.
    outbound = pd.DataFrame({
        "account": df["from_account"].to_numpy(),
        "timestamp_us": ts_us,
        "outbound_amount": df["amount"].to_numpy(),
        "_row_order": np.arange(n),
    })

    matched = pd.merge_asof(
        outbound.sort_values("timestamp_us", kind="mergesort"),
        inbound.sort_values("timestamp_us", kind="mergesort"),
        on="timestamp_us",
        by="account",
        direction="backward",
        allow_exact_matches=False,  # the inbound event must be strictly before the outbound one
        tolerance=lookback_us,      # enforce the lookback window: no match beyond this many microseconds back
    )
    matched = matched.sort_values("_row_order").reset_index(drop=True)

    has_match = matched["inbound_amount"].notna()
    inbound_amount = matched["inbound_amount"].fillna(0.0).to_numpy()
    outbound_amount = matched["outbound_amount"].to_numpy()

    amount_ratio = np.where(has_match, outbound_amount / (inbound_amount + EPSILON), 0.0)
    amount_difference = np.where(has_match, inbound_amount - outbound_amount, 0.0)
    # Cap ratio to a sane range -- a match against a tiny historical inbound
    # trickle shouldn't produce an extreme, uninformative outlier ratio.
    amount_ratio = np.clip(amount_ratio, 0.0, 5.0)

    out = pd.DataFrame({"amount_ratio": amount_ratio, "amount_difference": amount_difference})
    log.info(
        "Tier-2 transit features computed in %.1fs (%.2f%% of outbound txns matched an inbound txn within %sh)",
        time.time() - t0, 100.0 * has_match.mean(), lookback_hours,
    )
    return out


def build_features(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    tier1 = compute_tier1_features(df, args)
    tier2 = compute_tier2_transit_features(df, args.transit_lookback_hours)
    features = pd.concat([tier1.reset_index(drop=True), tier2.reset_index(drop=True)], axis=1)
    features = features[EDGE_FEATURE_NAMES]  # enforce canonical column order

    # FIX #2: include row_id as first column -- this becomes the guaranteed-unique
    # join key in tgn_train.py, replacing the composite key that caused fan-out.
    key_cols = ["row_id", "from_account", "to_account", "timestamp", "amount"]
    result = pd.concat([df[key_cols].reset_index(drop=True), features], axis=1)
    return result


def run_spot_checks(result: pd.DataFrame) -> None:
    """Mirrors the Forward Plan's 'Done when' criteria for Step 5.0."""
    structuring_hits = int(result["structuring_threshold_flag"].sum())
    log.info("[CHECK] structuring_threshold_flag hits: %s rows in the $8,000-$9,999.99 band", f"{structuring_hits:,}")

    transit_like = int((result["amount_ratio"] >= 0.95).sum())
    log.info(
        "[CHECK] transit-mule-like rows (amount_ratio >= 0.95, i.e. >=95%% forwarded): %s",
        f"{transit_like:,}",
    )
    if structuring_hits == 0:
        log.warning("No structuring-band transactions found at all -- verify --structuring-min/--structuring-max against your data's currency/scale.")


def write_spec(spec_path: Path, args: argparse.Namespace) -> None:
    spec = {
        "feature_names": EDGE_FEATURE_NAMES,
        "feature_count": len(EDGE_FEATURE_NAMES),
        "definitions": {
            "log_amount_paid": "log1p(amount)",
            "payment_format_encoded": f"categorical id, mapping={PAYMENT_FORMAT_ID}",
            "is_cross_bank": "1.0 if from_bank != to_bank else 0.0",
            "is_cash_payment": "1.0 if payment_format == 'Cash' else 0.0",
            "is_bitcoin_transaction": "1.0 if payment_format == 'Bitcoin' else 0.0",
            "structuring_threshold_flag": f"1.0 if {args.structuring_min} <= amount <= {args.structuring_max} else 0.0 (CTR evasion band)",
            "amount_ratio": (
                f"outbound_amount / (most_recent_inbound_amount_within_{args.transit_lookback_hours}h + eps), "
                f"clipped to [0, 5]; 0.0 if no qualifying inbound txn found. Transit-mule 'peeling chain' signature."
            ),
            "amount_difference": "most_recent_qualifying_inbound_amount - outbound_amount; 0.0 if no match.",
            "hour_sin": "sin(2*pi*hour_of_day/24)",
            "hour_cos": "cos(2*pi*hour_of_day/24)",
            "weekday_sin": "sin(2*pi*weekday/7)",
            "weekday_cos": "cos(2*pi*weekday/7)",
            "is_cross_currency": "1.0 if payment_currency != receiving_currency else 0.0",
        },
        "used_for": "RBI explainability documentation (Sprint 11) and TGN message function e_uv (Sprint 5)",
    }
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)
    log.info("Wrote feature spec to %s", spec_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        df = load_transactions(args.data_path)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    result = build_features(df, args)
    run_spot_checks(result)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output_path, index=False)
    log.info("Wrote %s rows x %s edge features to %s", f"{len(result):,}", len(EDGE_FEATURE_NAMES), args.output_path)

    write_spec(args.spec_path, args)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())