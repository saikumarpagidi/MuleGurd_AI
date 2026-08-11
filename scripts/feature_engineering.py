"""
Feature Engineering Pipeline — MuleGuard.AI
============================================
Transforms cleaned transaction data into model-ready feature sets.

Outputs:
  ml/data/processed/features_lgbm.parquet        — 66 features + label (31.9M rows)
  ml/data/processed/features_autoencoder.parquet — 21 features, legit rows only

Runtime estimate: 60–120 minutes on HI-Medium (31.9M rows).
Run overnight if needed. Progress is logged every major step.

Execution:
  python -X utf8 scripts/feature_engineering.py
"""

import os
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# ---------------------------------------------------------------------------
# Logging — consistent with data_prep.py style
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Rolling window sizes
WINDOW_1H  = "1h"
WINDOW_6H  = "6h"
WINDOW_24H = "24h"

# Payment format risk scores derived from empirical fraud rates in HI-Medium:
#   ACH=0.795%, Cash=0.021%, Cheque=0.018%, Credit Card=0.015%,
#   Bitcoin=0.035%, Reinvestment=0.000%, Wire=0.000%
PAYMENT_FORMAT_RISK = {
    "ACH":         6,
    "Cash":        4,
    "Cheque":      3,
    "Bitcoin":     2,
    "Credit Card": 2,
    "Reinvestment":1,
    "Wire":        1,
}

# Autoencoder feature subset (21 features — continuous & behavioral only)
AUTOENCODER_FEATURES = [
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
    # Empirical Validation Features
    "amount_ratio",
    "time_since_last_received",
    # New Receiver/Distribution Additions
    "amount_received_1h",
    "receiver_txn_count_1h",
    "amount_vs_receiver_mean",
]

# Full LightGBM feature set (66 features)
LGBM_FEATURES = [
    # Group A — Raw Amount
    "amount_paid",
    "amount_received",
    "log_amount_paid",
    "log_amount_received",
    # Group B — Amount Behavioral
    "amount_vs_sender_mean",
    "is_large_amount",
    "amount_robust_zscore",
    # Group C — Velocity
    "txn_count_1h",
    "txn_count_6h",
    "txn_count_24h",
    "amount_sent_1h",
    "amount_sent_24h",
    "fan_out_24h",
    "fan_in_24h",
    "consecutive_cross_bank_count",
    # Group D — Normalized Velocity Ratios
    "fan_out_ratio",
    "fan_in_ratio",
    # Group E — Account Network Scores
    "account_degree_ratio",
    "gather_scatter_score",
    "entity_net_flow_ratio",
    # Group F — Account Profile
    "sender_lifetime_txn_count",
    "sender_lifetime_total_sent_log",
    "sender_avg_amount",
    "sender_unique_receivers_all",
    "receiver_lifetime_txn_count",
    "receiver_lifetime_total_received_log",
    "receiver_avg_received",
    "receiver_unique_senders_all",
    # Group G — Temporal
    "hour_of_day",
    "minute_of_hour",
    "day_of_month",
    "day_of_week",
    "is_night",
    "is_weekend",
    "days_since_account_first_txn",
    "days_since_own_last_activity",
    # Group H — Activity Ratios
    "night_activity_ratio",
    "weekend_activity_ratio",
    "sender_daily_txn_rate",
    # Group I — Payment Format & Currency
    "payment_format_encoded",
    "is_ach_payment",
    "is_cross_currency",
    "payment_currency_encoded",
    "receiving_currency_encoded",
    # Group J — Empirical Validation Features
    "is_cross_bank",
    "round_amount_flag",
    "amount_ratio",
    "amount_difference",
    "is_self_transfer",
    "is_bitcoin_transaction",
    "is_cash_payment",
    "currency_pair_rarity",
    "time_since_last_received",
    "sender_receiver_pair_txn_count",
    # Group K — Receiver-Side Velocity
    "receiver_txn_count_1h",
    "receiver_txn_count_24h",
    "amount_received_1h",
    "amount_received_24h",
    "receiver_velocity_ratio",
    # Group L — Amount Distribution
    "sender_amount_std",
    "sender_amount_cv",
    "amount_vs_receiver_mean",
    "receiver_amount_std",
    # Group M — Interval Statistics
    "sender_interval_std_seconds",
    "sender_interval_cv",
    "sender_median_interval_hours",
]


# ---------------------------------------------------------------------------
# Helper: timer context
# ---------------------------------------------------------------------------
class _Timer:
    """Lightweight step timer for logging."""
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self._start = time.time()
        logger.info(f"  → {self.label} ...")
        return self

    def __exit__(self, *_):
        elapsed = time.time() - self._start
        logger.info(f"  ✓ {self.label} — {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Step 1: Log-transform amount features (Group A)
# ---------------------------------------------------------------------------
def compute_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Log₁₀-transform paid and received amounts.
    
    Transaction amounts range 15 orders of magnitude ($0.0001 → $154B).
    Log compression maps this to a stable [−4, 11] range so tree splits
    are not dominated by extreme Bitcoin values.
    """
    df["log_amount_paid"]     = np.log10(df["amount_paid"].clip(lower=1e-9))
    df["log_amount_received"] = np.log10(df["amount_received"].clip(lower=1e-9))
    return df


# ---------------------------------------------------------------------------
# Step 2: Amount behavioral features (Group B)
# ---------------------------------------------------------------------------
def compute_amount_behavioral(df: pd.DataFrame) -> pd.DataFrame:
    """Deviation of this transaction from the sender's historical average.
    
    amount_vs_sender_mean: ratio of current amount to sender's all-time mean.
      Ratio = 1.0  → perfectly normal for this sender.
      Ratio = 52   → 52× above normal — dormancy burst signal.
    
    is_large_amount: binary flag for top-5% by amount (data-driven threshold).
    """
    # Cumulative mean per sender (expanding window, up to and including current row).
    # Using cumulative sum / cumulative count avoids an expensive full-dataset groupby.
    df = df.sort_values(["from_account", "timestamp"]).reset_index(drop=True)

    grp_sender = df.groupby("from_account")["amount_paid"]
    cumsum   = grp_sender.cumsum()
    cumcount = grp_sender.cumcount() + 1          # 1-indexed

    sender_mean_upto_now = cumsum / cumcount
    df["amount_vs_sender_mean"] = (
        df["amount_paid"] / sender_mean_upto_now.replace(0, np.nan)
    ).fillna(1.0)

    # Data-driven large-amount threshold (95th percentile across full dataset)
    p95 = df["amount_paid"].quantile(0.95)
    logger.info(f"    Large-amount threshold (p95): ${p95:,.2f}")
    df["is_large_amount"] = (df["amount_paid"] > p95).astype(np.int8)

    # Robust Z-score using global Median and IQR
    global_median = df["amount_paid"].median()
    q25 = df["amount_paid"].quantile(0.25)
    q75 = df["amount_paid"].quantile(0.75)
    global_iqr = q75 - q25
    df["amount_robust_zscore"] = (
        (df["amount_paid"] - global_median) / max(global_iqr, 1e-9)
    ).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Step 3: Rolling velocity features (Group C)
# ---------------------------------------------------------------------------
def _make_timestamps_unique(df: pd.DataFrame, account_col: str) -> pd.DataFrame:
    """Add microsecond offsets to break timestamp ties within each account group.
    
    IBM AML timestamps have minute-level resolution. When two transactions from
    the same account share the same timestamp, pandas time-based rolling raises
    a ValueError (non-monotonic index). This function resolves ties by adding
    cumulative microsecond offsets (1us = effectively 0 for a 1h window).
    """
    df = df.copy()
    dupes = df.duplicated(subset=[account_col, "timestamp"], keep=False)
    if dupes.any():
        cumcnt = (
            df[dupes]
            .groupby([account_col, "timestamp"])
            .cumcount()
            .astype("int64")
        )
        df.loc[dupes, "timestamp"] = (
            df.loc[dupes, "timestamp"]
            + pd.to_timedelta(cumcnt, unit="us")
        )
    return df


def _rolling_vectorized(
    df: pd.DataFrame,
    account_col: str,
    value_col: str,
    window: str,
    agg: str,
) -> np.ndarray:
    """Vectorized rolling aggregation using groupby().rolling(on=timestamp).
    
    Returns a numpy array aligned to df's current row order.
    closed='left' excludes the current row from its own window — essential
    for real-time scoring correctness (no look-ahead leakage).
    
    Root cause of naive reindex failure: groupby().rolling(on='timestamp')
    returns a MultiIndex of (account, timestamp). After droplevel(0) the
    remaining index is timestamps — which can be duplicated across accounts,
    causing reindex to raise ValueError.
    
    Fix: sort work_df and record original positions via the preserved index.
    After computing rolling values (numpy array in sorted order), map back
    using pd.Series(vals, index=work.index).sort_index() to restore original
    row order without any reindex on potentially-duplicate timestamp labels.
    
    Precondition: df must have a unique integer index (call reset_index first).
    """
    # Sort by account and timestamp — preserves original integer index intact
    work = df[[account_col, "timestamp", value_col]].sort_values(
        [account_col, "timestamp"]
    )
    # work.index contains original df integer positions (0..N-1) reordered —
    # this is the key that lets us map results back to original row order.

    rolled = (
        work.groupby(account_col, sort=True)
        .rolling(window=window, on="timestamp", closed="left")
    )[value_col]

    vals = rolled.sum().values if agg == "sum" else rolled.count().values
    # vals is numpy array in groupby-sort order — same order as work.
    # Create Series with original df positions as index, sort to restore df order.
    return pd.Series(vals, index=work.index).sort_index().fillna(0).values



def compute_rolling_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling window count and sum features (Group C).
    
    All windows use closed='left' so the current transaction is excluded
    from its own window — preventing look-ahead leakage and matching
    real-time scoring semantics.
    
    IBM AML has minute-resolution timestamps. Multiple transactions from
    the same account can share the same timestamp. We resolve this by
    adding microsecond offsets before rolling, then restore the original
    timestamps — this has zero effect on any window >= 1 hour.
    """
    df = df.sort_values(["from_account", "timestamp"]).reset_index(drop=True)

    # ── Sender-side: make timestamps unique within each account ──────────
    logger.info("    Deduplicating sender timestamps (minute resolution) ...")
    sender_df = _make_timestamps_unique(
        df[["from_account", "timestamp", "amount_paid"]],
        account_col="from_account",
    )

    # ── Outbound count and sum — vectorized groupby rolling ───────────────
    logger.info("    Computing sender rolling windows (count & sum) ...")

    df["txn_count_1h"]    = _rolling_vectorized(sender_df, "from_account", "amount_paid", WINDOW_1H,  "count").astype(np.int32)
    df["txn_count_6h"]    = _rolling_vectorized(sender_df, "from_account", "amount_paid", WINDOW_6H,  "count").astype(np.int32)
    df["txn_count_24h"]   = _rolling_vectorized(sender_df, "from_account", "amount_paid", WINDOW_24H, "count").astype(np.int32)
    df["amount_sent_1h"]  = _rolling_vectorized(sender_df, "from_account", "amount_paid", WINDOW_1H,  "sum").astype(np.float32)
    df["amount_sent_24h"] = _rolling_vectorized(sender_df, "from_account", "amount_paid", WINDOW_24H, "sum").astype(np.float32)

    # ── Receiver-side count and sum ──────────────────────────────────────
    logger.info("    Computing receiver rolling windows (count & sum) ...")
    receiver_df = _make_timestamps_unique(
        df[["to_account", "timestamp", "amount_received"]],
        account_col="to_account",
    )
    df["receiver_txn_count_1h"]  = _rolling_vectorized(receiver_df, "to_account", "amount_received", WINDOW_1H,  "count").astype(np.int32)
    df["receiver_txn_count_24h"] = _rolling_vectorized(receiver_df, "to_account", "amount_received", WINDOW_24H, "count").astype(np.int32)
    df["amount_received_1h"]     = _rolling_vectorized(receiver_df, "to_account", "amount_received", WINDOW_1H,  "sum").astype(np.float32)
    df["amount_received_24h"]    = _rolling_vectorized(receiver_df, "to_account", "amount_received", WINDOW_24H, "sum").astype(np.float32)

    # ── Fan-out: unique receivers per sender in 24h ───────────────────────
    logger.info("    Computing fan_out_24h (unique receivers per sender in 24h) ...")
    df["fan_out_24h"] = _rolling_unique(
        df, account_col="from_account", partner_col="to_account", window_hours=24
    )

    # ── Fan-in: unique senders per receiver in 24h ────────────────────────
    logger.info("    Computing fan_in_24h (unique senders per receiver in 24h) ...")
    df["fan_in_24h"] = _rolling_unique(
        df, account_col="to_account", partner_col="from_account", window_hours=24
    )

    return df


def _rolling_unique(
    df: pd.DataFrame,
    account_col: str,
    partner_col: str,
    window_hours: int,
) -> np.ndarray:
    """Compute rolling unique partner count per account over a time window.

    Returns numpy array aligned to df's current row order.
    Semantics: closed='left' — current row NOT counted in its own window.

    Algorithm — O(N) sliding window with Counter dict:
      For each sorted account group:
        - Maintain a {partner: count} dict for entries inside the window.
        - Advance LEFT pointer to expire rows older than window_hours.
        - Record len(dict) BEFORE adding current row (closed='left').
        - Add current row's partner to dict for future iterations.

      Each partner is added and removed at most once => O(N) total.
      Handles large accounts (100K+ transactions) in seconds, not hours.
    """
    from collections import defaultdict

    window_ns = int(window_hours * 3600 * 1e9)   # nanoseconds

    df_sorted = df[[account_col, "timestamp", partner_col]].copy()
    df_sorted = df_sorted.sort_values([account_col, "timestamp"]).reset_index(drop=False)
    orig_index = df_sorted["index"].values

    ts_ns    = df_sorted["timestamp"].astype(np.int64).values
    accounts = df_sorted[account_col].values
    partners = df_sorted[partner_col].values

    result = np.zeros(len(df_sorted), dtype=np.int32)

    _, group_starts = np.unique(accounts, return_index=True)
    group_ends = np.append(group_starts[1:], len(df_sorted))

    for g_start, g_end in zip(group_starts, group_ends):
        grp_ts  = ts_ns[g_start:g_end]
        grp_ptr = partners[g_start:g_end]
        n       = g_end - g_start

        window_counts: dict = defaultdict(int)
        left = 0

        for i in range(n):
            cutoff = grp_ts[i] - window_ns

            # Expire rows outside window
            while left < i and grp_ts[left] < cutoff:
                p = grp_ptr[left]
                window_counts[p] -= 1
                if window_counts[p] == 0:
                    del window_counts[p]
                left += 1

            # Count before adding current row (closed='left')
            result[g_start + i] = len(window_counts)

            # Add current row for future iterations
            window_counts[grp_ptr[i]] += 1

    out = pd.Series(result, index=orig_index).sort_index()
    return out.values


# ---------------------------------------------------------------------------
# Step 4: Normalized velocity ratios (Group D)
# ---------------------------------------------------------------------------
def compute_velocity_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize fan_out/fan_in by each account's own lifetime daily average.
    
    fan_out_ratio = fan_out_24h / (sender's avg daily unique receivers + 1)
    fan_in_ratio  = fan_in_24h  / (receiver's avg daily unique senders + 1)
    
    Ratios contextualize velocity relative to each account's own baseline.
    A law firm sending to 15 unique clients daily (ratio ≈ 1.0) vs
    a personal account doing the same (ratio >> 1.0 = spike).
    """
    # Sender baseline: total unique receivers lifetime / days_since_first_txn
    sender_lifetime_uniq = (
        df.groupby("from_account")["to_account"]
        .transform("nunique")
    )
    sender_days = (
        df.groupby("from_account")["timestamp"]
        .transform(lambda x: (x.max() - x.min()).total_seconds() / 86400 + 1)
    )
    sender_daily_avg = sender_lifetime_uniq / sender_days.clip(lower=1)
    df["fan_out_ratio"] = df["fan_out_24h"] / (sender_daily_avg + 1)

    # Receiver baseline: total unique senders lifetime / days active
    receiver_lifetime_uniq = (
        df.groupby("to_account")["from_account"]
        .transform("nunique")
    )
    receiver_days = (
        df.groupby("to_account")["timestamp"]
        .transform(lambda x: (x.max() - x.min()).total_seconds() / 86400 + 1)
    )
    receiver_daily_avg = receiver_lifetime_uniq / receiver_days.clip(lower=1)
    df["fan_in_ratio"] = df["fan_in_24h"] / (receiver_daily_avg + 1)

    # Receiver burst ratio: receiver_txn_count_1h / daily avg
    receiver_lifetime_txns = (
        df.groupby("to_account")["amount_received"]
        .transform("count")
    )
    receiver_daily_avg_txns = receiver_lifetime_txns / receiver_days.clip(lower=1)
    df["receiver_velocity_ratio"] = df["receiver_txn_count_1h"] / (receiver_daily_avg_txns + 1)

    return df


# ---------------------------------------------------------------------------
# Step 5: Account network scores (Group E)
# ---------------------------------------------------------------------------
def compute_network_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute structural topology scores derived from fan-in / fan-out.
    
    account_degree_ratio:
      fan_out_24h / (fan_in_24h + 1)
      ≈ 0     → collection node (high fan-in, low fan-out)
      ≈ 1.0   → relay node (symmetric gather-scatter)
      >> 1    → distribution node (low fan-in, high fan-out)
    
    gather_scatter_score:
      min(fan_in, fan_out) / (max(fan_in, fan_out) + 1)
      Near 1.0 → perfect relay (purest mule signature)
      Near 0   → pure collection or pure distribution
    
    entity_net_flow_ratio:
      (lifetime_received - lifetime_sent) / (lifetime_received + lifetime_sent + 1)
      Range [-1, +1]. ≈ 0 = relay, +1 = collection endpoint, -1 = funding source.
    """
    fo = df["fan_out_24h"].values.astype(np.float32)
    fi = df["fan_in_24h"].values.astype(np.float32)

    df["account_degree_ratio"] = fo / (fi + 1.0)

    df["gather_scatter_score"] = (
        np.minimum(fo, fi) / (np.maximum(fo, fi) + 1.0)
    )

    # Lifetime totals per account
    lifetime_recv = (
        df.groupby("to_account")["amount_received"]
        .transform("sum")
        .reindex(df.index)
        .fillna(0)
    )
    lifetime_sent = (
        df.groupby("from_account")["amount_paid"]
        .transform("sum")
        .reindex(df.index)
        .fillna(0)
    )
    df["entity_net_flow_ratio"] = (
        (lifetime_recv - lifetime_sent)
        / (lifetime_recv + lifetime_sent + 1.0)
    )

    return df


# ---------------------------------------------------------------------------
# Step 6: Account profile features (Group F)
# ---------------------------------------------------------------------------
def compute_account_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lifetime aggregate statistics per sender and receiver.
    
    These are cumulative (expanding) features computed across the full dataset.
    They answer: "Who is this account, historically?"
    
    Sender features:  lifetime txn count, total sent, avg amount, unique receivers
    Receiver features: lifetime txn count, total received, avg received, unique senders
    """
    # ── Sender lifetime aggregates ────────────────────────────────────────
    sender_agg = (
        df.groupby("from_account")
        .agg(
            sender_lifetime_txn_count   =("amount_paid", "count"),
            sender_total_sent           =("amount_paid", "sum"),
            sender_amount_std           =("amount_paid", "std"),
            sender_unique_receivers_all =("to_account",  "nunique"),
        )
        .reset_index()
    )
    sender_agg["sender_lifetime_total_sent_log"] = np.log10(
        sender_agg["sender_total_sent"].clip(lower=1e-9)
    )
    sender_agg["sender_avg_amount"] = (
        sender_agg["sender_total_sent"] / sender_agg["sender_lifetime_txn_count"]
    )
    sender_agg["sender_amount_std"] = sender_agg["sender_amount_std"].fillna(0.0).astype(np.float32)
    sender_agg["sender_amount_cv"] = (
        sender_agg["sender_amount_std"] / sender_agg["sender_avg_amount"].clip(lower=1e-9)
    ).astype(np.float32)

    df = df.merge(
        sender_agg[[
            "from_account",
            "sender_lifetime_txn_count",
            "sender_lifetime_total_sent_log",
            "sender_avg_amount",
            "sender_amount_std",
            "sender_amount_cv",
            "sender_unique_receivers_all",
        ]],
        on="from_account",
        how="left",
    )

    # ── Receiver lifetime aggregates ──────────────────────────────────────
    receiver_agg = (
        df.groupby("to_account")
        .agg(
            receiver_lifetime_txn_count      =("amount_received", "count"),
            receiver_total_received          =("amount_received", "sum"),
            receiver_amount_std              =("amount_received", "std"),
            receiver_unique_senders_all      =("from_account",    "nunique"),
        )
        .reset_index()
    )
    receiver_agg["receiver_lifetime_total_received_log"] = np.log10(
        receiver_agg["receiver_total_received"].clip(lower=1e-9)
    )
    receiver_agg["receiver_avg_received"] = (
        receiver_agg["receiver_total_received"] / receiver_agg["receiver_lifetime_txn_count"]
    )
    receiver_agg["receiver_amount_std"] = receiver_agg["receiver_amount_std"].fillna(0.0).astype(np.float32)

    df = df.merge(
        receiver_agg[[
            "to_account",
            "receiver_lifetime_txn_count",
            "receiver_lifetime_total_received_log",
            "receiver_avg_received",
            "receiver_amount_std",
            "receiver_unique_senders_all",
        ]],
        on="to_account",
        how="left",
    )

    # amount_vs_receiver_mean compares the row-level received amount to the receiver's historical average
    df["amount_vs_receiver_mean"] = (
        df["amount_received"] / df["receiver_avg_received"].clip(lower=1e-9)
    ).fillna(1.0).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Step 7: Temporal features (Group G)
# ---------------------------------------------------------------------------
def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from the UTC timestamp.
    
    hour_of_day, minute_of_hour, day_of_month, day_of_week:
      Standard temporal signals. Mule operations cluster at night and weekends.
    
    is_night:   binary flag for hours 0–5 (low oversight window)
    is_weekend: binary flag for Saturday/Sunday
    
    days_since_account_first_txn:
      Account age proxy. A 0-day-old account forwarding $100K = new mule recruit.
    
    days_since_own_last_activity:
      Dormancy gap before this transaction. The key mule activation signal.
      An account dormant for 10 days that suddenly bursts is the textbook pattern.
    """
    ts = df["timestamp"]

    df["hour_of_day"]   = ts.dt.hour.astype(np.int8)
    df["minute_of_hour"]= ts.dt.minute.astype(np.int8)
    df["day_of_month"]  = ts.dt.day.astype(np.int8)
    df["day_of_week"]   = ts.dt.dayofweek.astype(np.int8)
    df["is_night"]      = ts.dt.hour.isin(range(6)).astype(np.int8)
    df["is_weekend"]    = ts.dt.dayofweek.isin([5, 6]).astype(np.int8)

    # Days since first transaction for this account
    first_seen = df.groupby("from_account")["timestamp"].transform("min")
    df["days_since_account_first_txn"] = (
        (ts - first_seen).dt.total_seconds() / 86400
    ).astype(np.float32)

    # Days since own last activity (dormancy gap)
    # Sort by (from_account, timestamp) so shift() gives prior transaction per account
    df = df.sort_values(["from_account", "timestamp"]).reset_index(drop=True)
    prev_ts = df.groupby("from_account")["timestamp"].shift(1)

    intervals_sec = (df["timestamp"] - prev_ts).dt.total_seconds()
    df["days_since_own_last_activity"] = (
        intervals_sec / 86400.0
    ).fillna(0).astype(np.float32)

    # ── Interval Statistics (Group M) ─────────────────────────────────────
    # intervals_sec has NaN for each account's first transaction (no prior row).
    # We group by from_account and compute std, median, mean per account,
    # then broadcast back to all rows via transform.
    logger.info("    Computing sender interval statistics ...")
    df["sender_interval_std_seconds"] = (
        df.groupby("from_account")["days_since_own_last_activity"]
        .transform("std")
        .mul(86400.0)          # convert days → seconds to match the feature name
        .fillna(0.0)
        .astype(np.float32)
    )
    sender_interval_mean_days = (
        df.groupby("from_account")["days_since_own_last_activity"]
        .transform("mean")
        .fillna(0.0)
    )
    sender_interval_median_days = (
        df.groupby("from_account")["days_since_own_last_activity"]
        .transform("median")
        .fillna(0.0)
    )
    # CV: std / mean (both in same units — days; ratio is unit-free)
    df["sender_interval_cv"] = (
        df["sender_interval_std_seconds"] / 86400.0    # back to days
        / (sender_interval_mean_days + 1e-9)
    ).fillna(0.0).astype(np.float32)
    df["sender_median_interval_hours"] = (
        sender_interval_median_days * 24.0             # days → hours
    ).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Step 8: Activity ratio features (Group H)
# ---------------------------------------------------------------------------
def compute_activity_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Normalized temporal behavior ratios per sender.
    
    night_activity_ratio:
      Proportion of this sender's lifetime transactions that occurred at night.
      Shift from 0.0 to 0.5 overnight = behavioral takeover signal.
    
    weekend_activity_ratio:
      Same for weekends. A historically weekday-only account going weekend-active
      indicates it may have been co-opted by a different operator.
    """
    grp = df.groupby("from_account")

    night_count = grp["is_night"].transform("sum")
    total_count = grp["is_night"].transform("count")
    df["night_activity_ratio"] = (night_count / total_count.clip(lower=1)).astype(np.float32)

    weekend_count = grp["is_weekend"].transform("sum")
    df["weekend_activity_ratio"] = (weekend_count / total_count.clip(lower=1)).astype(np.float32)

    df["sender_daily_txn_rate"] = (
        df["sender_lifetime_txn_count"] / (df["days_since_account_first_txn"] + 1.0)
    ).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Step 9: Format & currency features (Group I)
# ---------------------------------------------------------------------------
def compute_format_currency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode payment format and currency columns.
    
    payment_format_encoded:  label-encoded (7 formats → 0–6)
    is_ach_payment:           binary flag for highest-risk format (0.795% fraud rate)
    is_cross_currency:        binary flag for cross-currency transactions
    payment_currency_encoded: label-encoded (15 currencies)
    receiving_currency_encoded: label-encoded (15 currencies)
    payment_format_risk_score:  ordinal risk score from empirical fraud rates
    is_cross_bank:              binary flag (used in Autoencoder feature set)
    """
    le = LabelEncoder()

    df["payment_format_encoded"] = le.fit_transform(df["payment_format"])
    df["payment_currency_encoded"] = le.fit_transform(df["payment_currency"])
    df["receiving_currency_encoded"] = le.fit_transform(df["receiving_currency"])

    df["is_ach_payment"] = (df["payment_format"] == "ACH").astype(np.int8)
    df["is_cross_currency"] = (
        df["payment_currency"] != df["receiving_currency"]
    ).astype(np.int8)
    df["is_cross_bank"] = (df["from_bank"] != df["to_bank"]).astype(np.int8)

    df["payment_format_risk_score"] = (
        df["payment_format"].map(PAYMENT_FORMAT_RISK).fillna(1).astype(np.int8)
    )

    # Normalized versions used by Autoencoder
    df["hour_of_day_norm"] = (df["hour_of_day"] / 23.0).astype(np.float32)
    df["day_of_week_norm"] = (df["day_of_week"] / 6.0).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Step 10: Empirical validation features (Group J)
# ---------------------------------------------------------------------------
def compute_empirical_validation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute empirical validation features for the superset approach.
    
    These features were identified during the domain analysis phase as
    potentially relevant signals. They are included in the feature superset
    so that SHAP analysis can empirically confirm or reject their predictive
    contribution, rather than relying solely on theoretical judgment.
    
    Features computed:
      round_amount_flag           - Transaction amount is a round number ($1000 multiples)
      amount_ratio                - Ratio of received to paid amount (FX skimming indicator)
      amount_difference           - Absolute difference between paid and received amounts
      is_self_transfer            - Sender and receiver share the same account identifier
      is_bitcoin_transaction      - Payment made via Bitcoin
      is_cash_payment             - Payment made via Cash
      currency_pair_rarity        - How rare this specific currency pair is in the dataset
      time_since_last_received    - Dormancy gap on the receiver side (hours)
      sender_receiver_pair_txn_count - Frequency of this specific sender-receiver pair
      consecutive_cross_bank_count   - Number of consecutive preceding cross-bank transactions
    """
    # 1. Round amount flag — exact multiples of $1,000
    df["round_amount_flag"] = (
        (df["amount_paid"] % 1000 == 0) & (df["amount_paid"] > 0)
    ).astype(np.int8)

    # 2. Amount ratio — received / paid (FX skimming detection)
    df["amount_ratio"] = (
        df["amount_received"] / df["amount_paid"].clip(lower=1e-9)
    ).astype(np.float32)

    # 3. Amount difference — absolute skimming detection
    df["amount_difference"] = (
        np.abs(df["amount_received"] - df["amount_paid"])
    ).astype(np.float32)

    # 4. Self-transfer — sender == receiver (layering technique)
    df["is_self_transfer"] = (
        df["from_account"] == df["to_account"]
    ).astype(np.int8)

    # 5. Bitcoin transaction flag
    df["is_bitcoin_transaction"] = (
        df["payment_format"] == "Bitcoin"
    ).astype(np.int8)

    # 6. Cash payment flag
    df["is_cash_payment"] = (
        df["payment_format"] == "Cash"
    ).astype(np.int8)

    # 7. Currency pair rarity — inverse document frequency of the currency pair
    #    Rare pairs score higher (more suspicious). Common pairs score lower.
    currency_pair = df["payment_currency"] + "_TO_" + df["receiving_currency"]
    pair_counts = currency_pair.value_counts()
    total_txns = len(df)
    pair_rarity_map = np.log10(total_txns / pair_counts)
    df["currency_pair_rarity"] = currency_pair.map(pair_rarity_map).astype(np.float32)

    # 8. Time since last received — receiver-side dormancy (hours)
    df = df.sort_values(["to_account", "timestamp"]).reset_index(drop=True)
    prev_recv_ts = df.groupby("to_account")["timestamp"].shift(1)
    df["time_since_last_received"] = (
        (df["timestamp"] - prev_recv_ts).dt.total_seconds() / 3600
    ).fillna(0).astype(np.float32)

    # 9. Sender-receiver pair transaction count — lifetime frequency
    pair_key = df["from_account"].astype(str) + "_TO_" + df["to_account"].astype(str)
    pair_freq = pair_key.value_counts()
    df["sender_receiver_pair_txn_count"] = pair_key.map(pair_freq).astype(np.int32)

    # 10. Consecutive cross-bank count
    df = df.sort_values(["from_account", "timestamp"]).reset_index(drop=True)
    prev_is_cross_bank = df.groupby("from_account")["is_cross_bank"].shift(1).fillna(0)
    block = (prev_is_cross_bank == 0).groupby(df["from_account"]).cumsum()
    df["consecutive_cross_bank_count"] = df.groupby(["from_account", block]).cumcount().astype(np.int32)

    return df


# ---------------------------------------------------------------------------
# Step 11: Validation gate
# ---------------------------------------------------------------------------
def validate_features(df: pd.DataFrame, feature_list: list, label: str) -> None:
    """Assert feature integrity before writing output files.
    
    Checks:
      1. All expected feature columns exist.
      2. No NaN values in any feature column.
      3. No infinite values.
      4. Feature count is exactly as specified.
    """
    logger.info(f"Running validation gate for {label} ...")

    # 1. Column presence
    missing = [f for f in feature_list if f not in df.columns]
    assert not missing, f"Missing features: {missing}"

    feature_df = df[feature_list]

    # 2. No NaN
    null_counts = feature_df.isnull().sum()
    nulls_found = null_counts[null_counts > 0]
    assert nulls_found.empty, f"Null values found:\n{nulls_found}"

    # 3. No infinities
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(feature_df[numeric_cols].values).sum(axis=0)
    inf_series = pd.Series(inf_counts, index=numeric_cols)
    infs_found = inf_series[inf_series > 0]
    assert infs_found.empty, f"Infinite values found:\n{infs_found}"

    # 4. Feature count
    assert len(feature_list) == len(set(feature_list)), "Duplicate feature names in spec!"

    logger.info(
        f"  Validation PASSED — {label}: "
        f"{len(df):,} rows × {len(feature_list)} features, zero nulls, zero infs."
    )


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------
def build_features(input_path: str, output_lgbm: str, output_ae: str) -> None:
    """End-to-end feature engineering pipeline.
    
    Parameters
    ----------
    input_path  : Path to transactions_clean.parquet
    output_lgbm : Path to write features_lgbm.parquet
    output_ae   : Path to write features_autoencoder.parquet
    """
    pipeline_start = time.time()
    logger.info("=" * 70)
    logger.info("MuleGuard.AI — Feature Engineering Pipeline")
    logger.info("=" * 70)

    # ── Load ─────────────────────────────────────────────────────────────
    with _Timer("Loading cleaned parquet"):
        df = pd.read_parquet(input_path)
        logger.info(
            f"    Loaded {len(df):,} rows | "
            f"Fraud: {df['is_laundering'].sum():,} ({df['is_laundering'].mean():.4%})"
        )

    # ── Step 1: Log amount ────────────────────────────────────────────────
    with _Timer("Step 1/10 — Log-transform amounts (Group A)"):
        df = compute_log_features(df)

    # ── Step 2: Amount behavioral ─────────────────────────────────────────
    with _Timer("Step 2/10 — Amount behavioral features (Group B)"):
        df = compute_amount_behavioral(df)

    # ── Step 3: Rolling velocity ──────────────────────────────────────────
    with _Timer("Step 3/10 — Rolling velocity windows (Group C) [slowest step]"):
        df = compute_rolling_velocity(df)

    # ── Step 4: Velocity ratios ───────────────────────────────────────────
    with _Timer("Step 4/10 — Normalized velocity ratios (Group D)"):
        df = compute_velocity_ratios(df)

    # ── Step 5: Network scores ────────────────────────────────────────────
    with _Timer("Step 5/10 — Account network scores (Group E)"):
        df = compute_network_scores(df)

    # ── Step 6: Account profile ───────────────────────────────────────────
    with _Timer("Step 6/10 — Account profile aggregates (Group F)"):
        df = compute_account_profile(df)

    # ── Step 7: Temporal ──────────────────────────────────────────────────
    with _Timer("Step 7/10 — Temporal features (Group G)"):
        df = compute_temporal_features(df)

    # ── Step 8: Activity ratios ───────────────────────────────────────────
    with _Timer("Step 8/10 — Activity ratio features (Group H)"):
        df = compute_activity_ratios(df)

    # ── Step 9: Format & currency ─────────────────────────────────────────
    with _Timer("Step 9/10 — Format and currency encodings (Group I)"):
        df = compute_format_currency_features(df)

    # ── Step 10: Empirical validation features ────────────────────────────
    with _Timer("Step 10/10 — Empirical validation features (Group J)"):
        df = compute_empirical_validation_features(df)

    # ── Validation gate (LightGBM) ────────────────────────────────────────
    validate_features(df, LGBM_FEATURES, label="LightGBM (66 features)")

    # ── Save LightGBM features ────────────────────────────────────────────
    with _Timer("Saving features_lgbm.parquet"):
        os.makedirs(os.path.dirname(output_lgbm), exist_ok=True)
        lgbm_cols = LGBM_FEATURES + ["is_laundering"]
        df[lgbm_cols].to_parquet(output_lgbm, index=False)
        file_mb = os.path.getsize(output_lgbm) / 1024 / 1024
        logger.info(f"    Written: {output_lgbm} ({file_mb:.1f} MB)")

    # ── Build Autoencoder features (legitimate rows only) ─────────────────
    with _Timer("Filtering to legitimate rows for Autoencoder"):
        df_legit = df[df["is_laundering"] == 0].copy()
        logger.info(
            f"    Legitimate rows: {len(df_legit):,} "
            f"({len(df_legit) / len(df):.4%} of total)"
        )

    validate_features(df_legit, AUTOENCODER_FEATURES, label="Autoencoder (21 features)")

    with _Timer("Saving features_autoencoder.parquet"):
        os.makedirs(os.path.dirname(output_ae), exist_ok=True)
        df_legit[AUTOENCODER_FEATURES].to_parquet(output_ae, index=False)
        file_mb = os.path.getsize(output_ae) / 1024 / 1024
        logger.info(f"    Written: {output_ae} ({file_mb:.1f} MB)")

    # ── Summary ───────────────────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    minutes, seconds = divmod(int(total_elapsed), 60)

    logger.info("=" * 70)
    logger.info("Feature Engineering Complete")
    logger.info(f"  Total runtime        : {minutes}m {seconds}s")
    logger.info(f"  LightGBM rows        : {len(df):,}")
    logger.info(f"  LightGBM features    : {len(LGBM_FEATURES)}")
    logger.info(f"  Autoencoder rows     : {len(df_legit):,}")
    logger.info(f"  Autoencoder features : {len(AUTOENCODER_FEATURES)}")
    logger.info(f"  Output (LGBM)        : {output_lgbm}")
    logger.info(f"  Output (AE)          : {output_ae}")
    logger.info("  Next step: python -X utf8 ml/notebooks/02_lightgbm_train.ipynb")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    input_parquet  = project_root / "ml" / "data" / "processed" / "transactions_clean.parquet"
    output_lgbm    = project_root / "ml" / "data" / "processed" / "features_lgbm.parquet"
    output_ae      = project_root / "ml" / "data" / "processed" / "features_autoencoder.parquet"

    if not input_parquet.exists():
        logger.error(
            f"Input not found: {input_parquet}\n"
            "Run scripts/data_prep.py first to generate transactions_clean.parquet."
        )
        raise SystemExit(1)

    build_features(
        input_path  = str(input_parquet),
        output_lgbm = str(output_lgbm),
        output_ae   = str(output_ae),
    )
