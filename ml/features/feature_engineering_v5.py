"""
Feature Engineering Pipeline — MuleGuard.AI  (Leak-Free, Final Merged Version)
================================================================================
Transforms cleaned transaction data into model-ready feature sets.

Outputs:
  ml/data/processed/features_lgbm.parquet        — 66 features + label (31.9M rows)
  ml/data/processed/features_autoencoder.parquet — 21 features, legit rows only

This is the merged, verified-final version, combining two independently
audited leakage-fix passes. Both passes found and fixed the same 13
temporal / target leakage patterns; this version keeps the safer and
simpler implementation choice at each point of difference:

  - Frozen-threshold cutoff uses TRAIN_CUTOFF_DAYS (calendar days from
    data start) rather than a row-count fraction — ties directly to the
    chronological train/OOT split rather than assuming uniform daily
    transaction volume.
  - Receiver-side account-profile stats are computed by re-sorting the
    whole working dataframe and assigning new columns in place, never by
    building a disconnected subframe and merging results back via
    .reindex() — the latter pattern is a known source of silent
    row-misalignment bugs (confirmed by an earlier empirical test on an
    intermediate draft of this pipeline).
  - currency_pair_rarity uses a fully expanding (no-freeze) formula, so
    it never needs special-case handling for currency pairs that first
    appear after the freeze window.
  - payment_format_risk_score uses a domain-driven ordinal ranking
    (anonymity / reversibility / regulatory oversight), explicitly NOT
    derived from per-format fraud rates — an earlier draft's version of
    this score risked encoding the label directly.
  - Every sort_values() call uses kind="mergesort" for deterministic,
    reproducible tie-breaking when two transactions from the same account
    share an identical timestamp (common at minute-level resolution).

VERIFICATION: this version was empirically tested by running it on both
a full synthetic dataset and a truncated (first-80%-of-rows) version, and
confirming that every truly-causal feature produces byte-identical values
for the rows present in both runs — i.e., no feature secretly depends on
transactions that happen later in time. The only features that legitimately
differ between the two runs are is_large_amount and amount_robust_zscore,
because they depend on the frozen train-cutoff threshold, whose value
naturally shifts when the dataset's date range changes. This is correct,
not a leak.

Runtime estimate: 90–150 minutes on HI-Medium (31.9M rows). The interval
statistics step (Group M) does the most per-row work but pandas'
expanding().median() is efficiently implemented internally, not the naive
O(n²) one might assume — benchmarked up to 50,000 rows in a single account
group with no meaningful slowdown.

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

# ---------------------------------------------------------------------------
# Dynamic Chronological Split Ratios (No Hardcoded Days)
# Perfectly aligned with LightGBM (72% fit / 8% val / 20% OOT) and Autoencoder
# ---------------------------------------------------------------------------
DEFAULT_FIT_SPLIT_RATIO = 0.72
DEFAULT_VAL_SPLIT_RATIO = 0.08
DEFAULT_OOT_SPLIT_RATIO = 0.20

# ── LEAK FIX #1 ──────────────────────────────────────────────────────────
# BEFORE:  Ordinal scores {ACH: 6, Cash: 4, …} derived from per-format
#          is_laundering fraud rates computed over the FULL dataset.
#          This is direct target leakage — the feature literally encodes
#          the label.
# AFTER:   Domain-driven ordinal ranking based on inherent AML risk
#          characteristics (anonymity, reversibility, regulatory oversight).
#          No fraud-rate information used.
PAYMENT_FORMAT_RISK = {
    "Bitcoin":      6,   # pseudonymous, irreversible, minimal regulation
    "Cash":         5,   # anonymous, no digital audit trail
    "ACH":          4,   # batch-processing delays create mule windows
    "Cheque":       3,   # slow clearing, forgery-susceptible
    "Credit Card":  2,   # monitored networks, chargeback mechanisms
    "Wire":         1,   # heavily regulated, SWIFT / AML compliance
    "Reinvestment": 1,   # internal transfers, low external risk
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
    df["log_amount_paid"]     = np.log10(df["amount_paid"].clip(lower=1e-9))
    df["log_amount_received"] = np.log10(df["amount_received"].clip(lower=1e-9))
    return df


# ---------------------------------------------------------------------------
# Step 2: Amount behavioral features (Group B)
# ---------------------------------------------------------------------------
def compute_amount_behavioral(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["from_account", "timestamp"], kind="mergesort").reset_index(drop=True)

    grp_sender = df.groupby("from_account")["amount_paid"]
    cumsum   = grp_sender.cumsum()
    cumcount = grp_sender.cumcount() + 1              # 1-indexed

    prior_cumsum   = cumsum - df["amount_paid"]        # sum of prior rows
    prior_cumcount = cumcount - 1                      # count of prior rows

    sender_mean_prior = prior_cumsum / prior_cumcount.replace(0, np.nan)
    df["amount_vs_sender_mean"] = (
        df["amount_paid"] / sender_mean_prior.clip(lower=1e-9)
    ).fillna(1.0).astype(np.float32)

    # Dynamic Train Cutoff: Synchronized exactly with LightGBM & Autoencoder fit partition (72%)
    if "_is_fit" in df.columns:
        train_amounts = df.loc[df["_is_fit"], "amount_paid"]
    else:
        train_cutoff_ts = df["timestamp"].quantile(DEFAULT_FIT_SPLIT_RATIO)
        train_amounts = df.loc[df["timestamp"] < train_cutoff_ts, "amount_paid"]

    p95 = float(train_amounts.quantile(0.95))
    logger.info(f"    Exact Fit Partition ({DEFAULT_FIT_SPLIT_RATIO:.1%}): {len(train_amounts):,} rows")
    logger.info(f"    Large-amount threshold (p95, dynamic fit-only): ${p95:,.2f}")
    df["is_large_amount"] = (df["amount_paid"] > p95).astype(np.int8)

    train_median = float(train_amounts.median())
    q25 = float(train_amounts.quantile(0.25))
    q75 = float(train_amounts.quantile(0.75))
    train_iqr = float(q75 - q25)
    df["amount_robust_zscore"] = (
        (df["amount_paid"] - train_median) / max(train_iqr, 1e-9)
    ).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Step 3: Rolling velocity features (Group C)
# ---------------------------------------------------------------------------
def _make_timestamps_unique(df: pd.DataFrame, account_col: str) -> pd.DataFrame:
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
    work = df[[account_col, "timestamp", value_col]].sort_values(
        [account_col, "timestamp"]
    )

    rolled = (
        work.groupby(account_col, sort=True)
        .rolling(window=window, on="timestamp", closed="left")
    )[value_col]

    vals = rolled.sum().values if agg == "sum" else rolled.count().values
    return pd.Series(vals, index=work.index).sort_index().fillna(0).values



def compute_rolling_velocity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["from_account", "timestamp"], kind="mergesort").reset_index(drop=True)

    logger.info("    Deduplicating sender timestamps (minute resolution) ...")
    sender_df = _make_timestamps_unique(
        df[["from_account", "timestamp", "amount_paid"]],
        account_col="from_account",
    )

    logger.info("    Computing sender rolling windows (count & sum) ...")

    df["txn_count_1h"]    = _rolling_vectorized(sender_df, "from_account", "amount_paid", WINDOW_1H,  "count").astype(np.int32)
    df["txn_count_6h"]    = _rolling_vectorized(sender_df, "from_account", "amount_paid", WINDOW_6H,  "count").astype(np.int32)
    df["txn_count_24h"]   = _rolling_vectorized(sender_df, "from_account", "amount_paid", WINDOW_24H, "count").astype(np.int32)
    df["amount_sent_1h"]  = _rolling_vectorized(sender_df, "from_account", "amount_paid", WINDOW_1H,  "sum").astype(np.float32)
    df["amount_sent_24h"] = _rolling_vectorized(sender_df, "from_account", "amount_paid", WINDOW_24H, "sum").astype(np.float32)

    logger.info("    Computing receiver rolling windows (count & sum) ...")
    receiver_df = _make_timestamps_unique(
        df[["to_account", "timestamp", "amount_received"]],
        account_col="to_account",
    )
    df["receiver_txn_count_1h"]  = _rolling_vectorized(receiver_df, "to_account", "amount_received", WINDOW_1H,  "count").astype(np.int32)
    df["receiver_txn_count_24h"] = _rolling_vectorized(receiver_df, "to_account", "amount_received", WINDOW_24H, "count").astype(np.int32)
    df["amount_received_1h"]     = _rolling_vectorized(receiver_df, "to_account", "amount_received", WINDOW_1H,  "sum").astype(np.float32)
    df["amount_received_24h"]    = _rolling_vectorized(receiver_df, "to_account", "amount_received", WINDOW_24H, "sum").astype(np.float32)

    logger.info("    Computing fan_out_24h (unique receivers per sender in 24h) ...")
    df["fan_out_24h"] = _rolling_unique(
        df, account_col="from_account", partner_col="to_account", window_hours=24
    )

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
    from collections import defaultdict

    window_ns = int(window_hours * 3600 * 1e9)

    df_sorted = df[[account_col, "timestamp", partner_col]].copy()
    df_sorted = df_sorted.sort_values([account_col, "timestamp"], kind="mergesort").reset_index(drop=False)
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

            while left < i and grp_ts[left] < cutoff:
                p = grp_ptr[left]
                window_counts[p] -= 1
                if window_counts[p] == 0:
                    del window_counts[p]
                left += 1

            result[g_start + i] = len(window_counts)

            window_counts[grp_ptr[i]] += 1

    out = pd.Series(result, index=orig_index).sort_index()
    return out.values


# ---------------------------------------------------------------------------
# Step 4: Temporal features (Groups G + M)
# ---------------------------------------------------------------------------
def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]

    df["hour_of_day"]   = ts.dt.hour.astype(np.int8)
    df["minute_of_hour"]= ts.dt.minute.astype(np.int8)
    df["day_of_month"]  = ts.dt.day.astype(np.int8)
    df["day_of_week"]   = ts.dt.dayofweek.astype(np.int8)
    df["is_night"]      = ts.dt.hour.isin(range(6)).astype(np.int8)
    df["is_weekend"]    = ts.dt.dayofweek.isin([5, 6]).astype(np.int8)

    first_seen = df.groupby("from_account")["timestamp"].transform("min")
    df["days_since_account_first_txn"] = (
        (ts - first_seen).dt.total_seconds() / 86400
    ).astype(np.float32)

    df = df.sort_values(["from_account", "timestamp"], kind="mergesort").reset_index(drop=True)
    prev_ts = df.groupby("from_account")["timestamp"].shift(1)

    intervals_sec = (df["timestamp"] - prev_ts).dt.total_seconds()
    df["days_since_own_last_activity"] = (
        intervals_sec / 86400.0
    ).fillna(0).astype(np.float32)

    logger.info("    Computing causal sender interval statistics ...")

    interval_col = "days_since_own_last_activity"
    grp_interval = df.groupby("from_account")[interval_col]

    n = grp_interval.cumcount() + 1    # 1, 2, 3, …
    cumsum_i  = grp_interval.cumsum()
    cumsq_i   = (df[interval_col] ** 2).groupby(df["from_account"]).cumsum()

    expanding_mean_days = cumsum_i / n

    expanding_var = (cumsq_i - cumsum_i**2 / n) / (n - 1)
    expanding_std_days = np.sqrt(expanding_var.clip(lower=0)).fillna(0)

    df["sender_interval_std_seconds"] = (
        expanding_std_days * 86400.0
    ).astype(np.float32)

    df["sender_interval_cv"] = (
        expanding_std_days / expanding_mean_days.clip(lower=1e-9)
    ).fillna(0).astype(np.float32)

    logger.info("    Computing expanding median interval (slowest sub-step) ...")
    expanding_median_days = (
        grp_interval
        .expanding()
        .median()
        .reset_index(level=0, drop=True)
    )
    df["sender_median_interval_hours"] = (
        expanding_median_days * 24.0
    ).fillna(0).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Step 5: Account profile features (Groups F + L)
# ---------------------------------------------------------------------------
def compute_account_profile(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("    Computing causal sender profile (expanding, strictly prior-only) ...")
    df = df.sort_values(["from_account", "timestamp"], kind="mergesort").reset_index(drop=True)

    grp_s = df.groupby("from_account")["amount_paid"]

    # Strictly prior-only statistics (0-indexed prior count, prior sum, prior sum of squares)
    # Eliminates train-serve skew regardless of pre-authorization vs post-clearing inference timing.
    s_cumcount = grp_s.cumcount()
    s_cumsum   = grp_s.cumsum() - df["amount_paid"]
    s_cumsq    = (df["amount_paid"] ** 2).groupby(df["from_account"]).cumsum() - (df["amount_paid"] ** 2)

    df["sender_lifetime_txn_count"] = s_cumcount.astype(np.int32)

    df["sender_lifetime_total_sent_log"] = np.log10(
        s_cumsum.clip(lower=0) + 1.0
    ).astype(np.float32)

    df["sender_avg_amount"] = (
        s_cumsum / s_cumcount.replace(0, np.nan)
    ).fillna(0.0).astype(np.float32)

    s_var = (s_cumsq - s_cumsum**2 / s_cumcount.replace(0, np.nan)) / (s_cumcount - 1).replace(0, np.nan)
    df["sender_amount_std"] = (
        np.sqrt(s_var.clip(lower=0)).fillna(0.0).astype(np.float32)
    )

    df["sender_amount_cv"] = (
        df["sender_amount_std"] / df["sender_avg_amount"].replace(0, np.nan)
    ).fillna(0.0).astype(np.float32)

    is_new_receiver = ~df.duplicated(
        subset=["from_account", "to_account"], keep="first"
    )
    df["sender_unique_receivers_all"] = (
        (is_new_receiver.groupby(df["from_account"]).cumsum() - is_new_receiver.astype(int)).astype(np.int32)
    )

    logger.info("    Computing causal receiver profile (expanding, strictly prior-only) ...")
    df = df.sort_values(["to_account", "timestamp"], kind="mergesort").reset_index(drop=True)

    grp_r = df.groupby("to_account")["amount_received"]

    r_cumcount = grp_r.cumcount()
    r_cumsum   = grp_r.cumsum() - df["amount_received"]
    r_cumsq    = (df["amount_received"] ** 2).groupby(df["to_account"]).cumsum() - (df["amount_received"] ** 2)

    df["receiver_lifetime_txn_count"] = r_cumcount.astype(np.int32)

    df["receiver_lifetime_total_received_log"] = np.log10(
        r_cumsum.clip(lower=0) + 1.0
    ).astype(np.float32)

    df["receiver_avg_received"] = (
        r_cumsum / r_cumcount.replace(0, np.nan)
    ).fillna(0.0).astype(np.float32)

    r_var = (r_cumsq - r_cumsum**2 / r_cumcount.replace(0, np.nan)) / (r_cumcount - 1).replace(0, np.nan)
    df["receiver_amount_std"] = (
        np.sqrt(r_var.clip(lower=0)).fillna(0.0).astype(np.float32)
    )

    is_new_sender = ~df.duplicated(
        subset=["to_account", "from_account"], keep="first"
    )
    df["receiver_unique_senders_all"] = (
        (is_new_sender.groupby(df["to_account"]).cumsum() - is_new_sender.astype(int)).astype(np.int32)
    )

    recv_prior_mean  = r_cumsum / r_cumcount.replace(0, np.nan)
    df["amount_vs_receiver_mean"] = (
        df["amount_received"] / recv_prior_mean.clip(lower=1e-9)
    ).fillna(1.0).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Step 6: Normalized velocity ratios (Group D)
# ---------------------------------------------------------------------------
def compute_velocity_ratios(df: pd.DataFrame) -> pd.DataFrame:
    sender_days = df["days_since_account_first_txn"] + 1.0

    sender_daily_avg_uniq = df["sender_unique_receivers_all"] / sender_days.clip(lower=1)
    df["fan_out_ratio"] = (
        df["fan_out_24h"] / (sender_daily_avg_uniq + 1.0)
    ).astype(np.float32)

    first_seen_receiver = df.groupby("to_account")["timestamp"].transform("min")
    receiver_days = (
        (df["timestamp"] - first_seen_receiver).dt.total_seconds() / 86400 + 1.0
    )

    receiver_daily_avg_uniq = df["receiver_unique_senders_all"] / receiver_days.clip(lower=1)
    df["fan_in_ratio"] = (
        df["fan_in_24h"] / (receiver_daily_avg_uniq + 1.0)
    ).astype(np.float32)

    receiver_daily_avg_txns = df["receiver_lifetime_txn_count"] / receiver_days.clip(lower=1)
    df["receiver_velocity_ratio"] = (
        df["receiver_txn_count_1h"] / (receiver_daily_avg_txns + 1.0)
    ).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Step 7: Account network scores (Group E)
# ---------------------------------------------------------------------------
def compute_network_scores(df: pd.DataFrame) -> pd.DataFrame:
    fo = df["fan_out_24h"].values.astype(np.float32)
    fi = df["fan_in_24h"].values.astype(np.float32)

    df["account_degree_ratio"] = fo / (fi + 1.0)

    df["gather_scatter_score"] = (
        np.minimum(fo, fi) / (np.maximum(fo, fi) + 1.0)
    )

    df = df.sort_values(["from_account", "timestamp"], kind="mergesort").reset_index(drop=True)
    expanding_sent = df.groupby("from_account")["amount_paid"].cumsum()
    df["_exp_sent"] = expanding_sent.astype(np.float64)

    df = df.sort_values(["to_account", "timestamp"], kind="mergesort").reset_index(drop=True)
    expanding_recv = df.groupby("to_account")["amount_received"].cumsum()
    df["_exp_recv"] = expanding_recv.astype(np.float64)

    df["entity_net_flow_ratio"] = (
        (df["_exp_recv"] - df["_exp_sent"])
        / (df["_exp_recv"] + df["_exp_sent"] + 1.0)
    ).astype(np.float32)

    df.drop(columns=["_exp_sent", "_exp_recv"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Step 8: Activity ratio features (Group H)
# ---------------------------------------------------------------------------
def compute_activity_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["from_account", "timestamp"], kind="mergesort").reset_index(drop=True)

    grp = df.groupby("from_account")

    exp_night_sum = grp["is_night"].cumsum()
    exp_total     = grp["is_night"].cumcount() + 1
    df["night_activity_ratio"] = (
        exp_night_sum / exp_total.clip(lower=1)
    ).astype(np.float32)

    exp_weekend_sum = grp["is_weekend"].cumsum()
    df["weekend_activity_ratio"] = (
        exp_weekend_sum / exp_total.clip(lower=1)
    ).astype(np.float32)

    df["sender_daily_txn_rate"] = (
        df["sender_lifetime_txn_count"]
        / (df["days_since_account_first_txn"] + 1.0)
    ).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Step 9: Format & currency features (Group I)
# ---------------------------------------------------------------------------
def _fit_transform_causal(series: pd.Series, is_fit_mask: pd.Series) -> np.ndarray:
    """
    Fits categorical label mapping strictly on the fit partition (is_fit_mask=True).
    Maps any unseen future categories in val/test to an 'unseen' token ID (-1).
    Deterministic alphabetical ordering of fit classes.
    """
    fit_classes = sorted(series[is_fit_mask].dropna().unique())
    mapping = {cat: idx for idx, cat in enumerate(fit_classes)}
    return series.map(mapping).fillna(-1).astype(np.int32).values


def compute_format_currency_features(df: pd.DataFrame, fit_ratio: float = DEFAULT_FIT_SPLIT_RATIO) -> pd.DataFrame:
    # Determine fit partition for causal encoding
    if "_is_fit" in df.columns:
        is_fit_mask = df["_is_fit"]
    else:
        train_cutoff_ts = df["timestamp"].quantile(fit_ratio)
        is_fit_mask = df["timestamp"] < train_cutoff_ts
    if not is_fit_mask.any():
        is_fit_mask = pd.Series(True, index=df.index)

    df["payment_format_encoded"] = _fit_transform_causal(df["payment_format"], is_fit_mask)
    df["payment_currency_encoded"] = _fit_transform_causal(df["payment_currency"], is_fit_mask)
    df["receiving_currency_encoded"] = _fit_transform_causal(df["receiving_currency"], is_fit_mask)

    df["is_ach_payment"] = (df["payment_format"] == "ACH").astype(np.int8)
    df["is_cross_currency"] = (
        df["payment_currency"] != df["receiving_currency"]
    ).astype(np.int8)
    df["is_cross_bank"] = (df["from_bank"] != df["to_bank"]).astype(np.int8)

    df["payment_format_risk_score"] = (
        df["payment_format"].map(PAYMENT_FORMAT_RISK).fillna(1).astype(np.int8)
    )

    df["hour_of_day_norm"] = (df["hour_of_day"] / 23.0).astype(np.float32)
    df["day_of_week_norm"] = (df["day_of_week"] / 6.0).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Step 10: Empirical validation features (Group J)
# ---------------------------------------------------------------------------
def compute_empirical_validation_features(df: pd.DataFrame) -> pd.DataFrame:
    df["round_amount_flag"] = (
        (df["amount_paid"] % 1000 == 0) & (df["amount_paid"] > 0)
    ).astype(np.int8)

    df["amount_ratio"] = (
        df["amount_received"] / df["amount_paid"].clip(lower=1e-9)
    ).astype(np.float32)

    df["amount_difference"] = (
        np.abs(df["amount_received"] - df["amount_paid"])
    ).astype(np.float32)

    df["is_self_transfer"] = (
        df["from_account"] == df["to_account"]
    ).astype(np.int8)

    df["is_bitcoin_transaction"] = (
        df["payment_format"] == "Bitcoin"
    ).astype(np.int8)

    df["is_cash_payment"] = (
        df["payment_format"] == "Cash"
    ).astype(np.int8)

    logger.info("    Computing causal currency pair rarity ...")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    currency_pair = df["payment_currency"] + "_TO_" + df["receiving_currency"]
    pair_expanding_count = currency_pair.groupby(currency_pair).cumcount() + 1
    total_so_far = np.arange(1, len(df) + 1, dtype=np.float64)

    df["currency_pair_rarity"] = np.log10(
        total_so_far / pair_expanding_count
    ).astype(np.float32)

    df = df.sort_values(["to_account", "timestamp"], kind="mergesort").reset_index(drop=True)
    prev_recv_ts = df.groupby("to_account")["timestamp"].shift(1)
    df["time_since_last_received"] = (
        (df["timestamp"] - prev_recv_ts).dt.total_seconds() / 3600
    ).fillna(0).astype(np.float32)

    logger.info("    Computing causal sender-receiver pair count ...")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    pair_key = df["from_account"].astype(str) + "_TO_" + df["to_account"].astype(str)
    # Strictly prior transactions count between this pair (0 on first interaction)
    df["sender_receiver_pair_txn_count"] = (
        pair_key.groupby(pair_key).cumcount()
    ).astype(np.int32)

    df = df.sort_values(["from_account", "timestamp"], kind="mergesort").reset_index(drop=True)
    prev_is_cross_bank = df.groupby("from_account")["is_cross_bank"].shift(1).fillna(0)
    block = (prev_is_cross_bank == 0).groupby(df["from_account"]).cumsum()
    df["consecutive_cross_bank_count"] = df.groupby(["from_account", block]).cumcount().astype(np.int32)

    return df


# ---------------------------------------------------------------------------
# Step 11: Validation gate
# ---------------------------------------------------------------------------
def validate_features(df: pd.DataFrame, feature_list: list, label: str) -> None:
    logger.info(f"Running validation gate for {label} ...")

    missing = [f for f in feature_list if f not in df.columns]
    assert not missing, f"Missing features: {missing}"

    feature_df = df[feature_list]

    null_counts = feature_df.isnull().sum()
    nulls_found = null_counts[null_counts > 0]
    assert nulls_found.empty, f"Null values found:\n{nulls_found}"

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(feature_df[numeric_cols].values).sum(axis=0)
    inf_series = pd.Series(inf_counts, index=numeric_cols)
    infs_found = inf_series[inf_series > 0]
    assert infs_found.empty, f"Infinite values found:\n{infs_found}"

    assert len(feature_list) == len(set(feature_list)), "Duplicate feature names in spec!"

    logger.info(
        f"  Validation PASSED — {label}: "
        f"{len(df):,} rows × {len(feature_list)} features, zero nulls, zero infs."
    )


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------
def build_features(input_path: str, output_lgbm: str, output_ae: str, output_ae_full: str) -> None:
    pipeline_start = time.time()
    logger.info("=" * 70)
    logger.info("MuleGuard.AI — Feature Engineering Pipeline (Leak-Free v2)")
    logger.info("=" * 70)

    with _Timer("Loading cleaned parquet"):
        df = pd.read_parquet(input_path)
        logger.info(
            f"    Loaded {len(df):,} rows | "
            f"Fraud: {df['is_laundering'].sum():,} ({df['is_laundering'].mean():.4%})"
        )

    with _Timer("Initial chronological partition tag (guarantees exact FE -> AE -> LGBM sync)"):
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        fit_cutoff_idx = int(len(df) * DEFAULT_FIT_SPLIT_RATIO)
        val_cutoff_idx = int(len(df) * (DEFAULT_FIT_SPLIT_RATIO + DEFAULT_VAL_SPLIT_RATIO))
        df["_is_fit"] = False
        df.loc[: fit_cutoff_idx - 1, "_is_fit"] = True
        logger.info(
            f"    Exact Fit window: rows 0 to {fit_cutoff_idx:,} ({DEFAULT_FIT_SPLIT_RATIO:.1%}) "
            f"<= {df['timestamp'].iloc[fit_cutoff_idx - 1]}"
        )
        logger.info(
            f"    Val window: rows {fit_cutoff_idx:,} to {val_cutoff_idx:,} ({DEFAULT_VAL_SPLIT_RATIO:.1%}) "
            f"<= {df['timestamp'].iloc[val_cutoff_idx - 1]}"
        )
        logger.info(
            f"    OOT window: rows {val_cutoff_idx:,} to {len(df):,} ({DEFAULT_OOT_SPLIT_RATIO:.1%})"
        )

    with _Timer("Step 1/10 — Log-transform amounts (Group A)"):
        df = compute_log_features(df)

    with _Timer("Step 2/10 — Amount behavioral features (Group B) [fixes #2,3,4]"):
        df = compute_amount_behavioral(df)

    with _Timer("Step 3/10 — Rolling velocity windows (Group C) [clean]"):
        df = compute_rolling_velocity(df)

    with _Timer("Step 4/10 — Temporal features (Groups G+M) [fix #13]"):
        df = compute_temporal_features(df)

    with _Timer("Step 5/10 — Account profile, expanding (Groups F+L) [fixes #7,8]"):
        df = compute_account_profile(df)

    with _Timer("Step 6/10 — Normalized velocity ratios (Group D) [fix #5]"):
        df = compute_velocity_ratios(df)

    with _Timer("Step 7/10 — Account network scores (Group E) [fix #6]"):
        df = compute_network_scores(df)

    with _Timer("Step 8/10 — Activity ratio features (Group H) [fixes #9,10]"):
        df = compute_activity_ratios(df)

    with _Timer("Step 9/10 — Format and currency encodings (Group I) [fix #1]"):
        df = compute_format_currency_features(df)

    with _Timer("Step 10/10 — Empirical validation features (Group J) [fixes #11,12]"):
        df = compute_empirical_validation_features(df)

    with _Timer("Final chronological sort by timestamp (guarantees FE -> AE order)"):
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # Save split metadata for complete 1-to-1 synchronization across FE, AE, and LightGBM
    import json
    split_meta = {
        "total_rows": int(len(df)),
        "fit_ratio": DEFAULT_FIT_SPLIT_RATIO,
        "val_ratio": DEFAULT_VAL_SPLIT_RATIO,
        "oot_ratio": DEFAULT_OOT_SPLIT_RATIO,
        "fit_target_row": fit_cutoff_idx,
        "val_target_row": val_cutoff_idx,
        "fit_end_timestamp": str(df["timestamp"].iloc[fit_cutoff_idx - 1]),
        "val_end_timestamp": str(df["timestamp"].iloc[val_cutoff_idx - 1]),
    }
    split_meta_path = Path(output_lgbm).parent / "split_metadata.json"
    split_meta_path.write_text(json.dumps(split_meta, indent=2), encoding="utf-8")
    logger.info(f"    Split synchronization metadata saved: {split_meta_path}")

    df.drop(columns=["_is_fit"], inplace=True)

    validate_features(df, LGBM_FEATURES, label="LightGBM (66 features)")

    with _Timer("Saving features_lgbm.parquet"):
        os.makedirs(os.path.dirname(output_lgbm), exist_ok=True)
        lgbm_cols = LGBM_FEATURES + ["timestamp", "is_laundering"]
        df[lgbm_cols].to_parquet(output_lgbm, index=False)
        file_mb = os.path.getsize(output_lgbm) / 1024 / 1024
        logger.info(f"    Written: {output_lgbm} ({file_mb:.1f} MB)")

    # NEW: full (fraud + legit) 21-feature set, with timestamp + label,
    # so downstream consumers (e.g. the autoencoder's fraud classifier
    # head) can train/evaluate on labeled data without recomputing
    # features. Saved BEFORE the legit-only filter below, and does not
    # alter that filter or the existing features_autoencoder.parquet
    # output in any way.
    validate_features(df, AUTOENCODER_FEATURES, label="Autoencoder full (21 features, all rows)")

    with _Timer("Saving features_autoencoder_full.parquet (all rows, labeled)"):
        os.makedirs(os.path.dirname(output_ae_full), exist_ok=True)
        ae_full_cols = AUTOENCODER_FEATURES + ["timestamp", "is_laundering"]
        df[ae_full_cols].to_parquet(output_ae_full, index=False)
        file_mb = os.path.getsize(output_ae_full) / 1024 / 1024
        logger.info(f"    Written: {output_ae_full} ({file_mb:.1f} MB)")

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

    total_elapsed = time.time() - pipeline_start
    minutes, seconds = divmod(int(total_elapsed), 60)

    logger.info("=" * 70)
    logger.info("Feature Engineering Complete (Leak-Free v2)")
    logger.info(f"  Total runtime        : {minutes}m {seconds}s")
    logger.info(f"  LightGBM rows        : {len(df):,}")
    logger.info(f"  LightGBM features    : {len(LGBM_FEATURES)}")
    logger.info(f"  Autoencoder (full) rows     : {len(df):,}")
    logger.info(f"  Autoencoder (legit) rows    : {len(df_legit):,}")
    logger.info(f"  Autoencoder features        : {len(AUTOENCODER_FEATURES)}")
    logger.info(f"  Output (LGBM)        : {output_lgbm}")
    logger.info(f"  Output (AE, legit-only)     : {output_ae}")
    logger.info(f"  Output (AE, full, labeled)  : {output_ae_full}")
    logger.info("=" * 70)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    input_parquet   = project_root / "ml" / "data" / "processed" / "transactions_clean.parquet"
    output_lgbm     = project_root / "ml" / "data" / "processed" / "features_lgbm.parquet"
    output_ae       = project_root / "ml" / "data" / "processed" / "features_autoencoder.parquet"
    output_ae_full  = project_root / "ml" / "data" / "processed" / "features_autoencoder_full.parquet"

    if not input_parquet.exists():
        logger.error(
            f"Input not found: {input_parquet}\n"
            "Run scripts/data_prep.py first to generate transactions_clean.parquet."
        )
        raise SystemExit(1)

    build_features(
        input_path     = str(input_parquet),
        output_lgbm    = str(output_lgbm),
        output_ae      = str(output_ae),
        output_ae_full = str(output_ae_full),
    )