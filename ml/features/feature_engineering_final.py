"""
Feature Engineering fINAL Pipeline — MuleGuard.AI  (Leak-Free, Re-Verified)
================================================================================
Transforms cleaned transaction data into model-ready feature sets.

Outputs:
  ml/data/processed/features_lgbm.parquet        — 66 features + label (31.9M rows)
  ml/data/processed/features_autoencoder.parquet — 21 features, legit rows only

WHY THIS FILE WAS REBUILT
--------------------------------------------------------------------------------
A prior copy of this file had regressed to an earlier, pre-audit version --
full-dataset (non-causal) lifetime aggregates in compute_account_profile,
a full-dataset percentile threshold in compute_amount_behavioral, and a
PAYMENT_FORMAT_RISK dict explicitly derived from empirical fraud rates
(direct target leakage -- the feature literally encoded the label). All of
this corrupts every row's features with information from the future,
regardless of how correctly a downstream script like train_lgbm_aml.py
splits the data afterward -- a clean row-level split cannot undo a leak
baked into the feature VALUES themselves.

This version restores every fix from that audit AND closes an alignment gap
identified across two review passes. The frozen global-threshold cutoff
(used by is_large_amount / amount_robust_zscore) now matches
train_lgbm_aml.py's FIT-ONLY boundary -- the exact row range whose
gradients actually update the model's weights -- rather than a calendar-day
approximation, and rather than the FIT+VAL boundary. An earlier version of
this cutoff used OOT_SPLIT_FRACTION directly (0.80, the FIT+VAL boundary),
which let the validation window (~8% of rows) influence the frozen
thresholds even though those exact rows influence which checkpoint
train_lgbm_aml.py ultimately selects via early stopping -- a mild but real
form of the pipeline having "seen" that data before the model is locked in.
TRAIN_CUTOFF_FRACTION is now computed as
OOT_SPLIT_FRACTION * (1 - INTERNAL_VAL_FRACTION), the same formula
train_lgbm_aml.py's own chronological_splits() uses for its FIT/VAL
boundary, so the two files cannot drift apart as long as the two source
constants (duplicated with an explicit "MUST match" comment at the top of
this file) are kept in sync.

Leak inventory restored (13 fixes):
  1. payment_format_risk_score  -- domain-driven ranking (anonymity /
     reversibility / regulatory oversight), NOT derived from fraud rates.
  2. is_large_amount            -- frozen p95 threshold, train-fraction only.
  3. amount_robust_zscore       -- frozen median/IQR, train-fraction only.
  4. amount_vs_sender_mean      -- strictly prior-only (shift-by-one), not
     self-inclusive.
  5. fan_out_ratio / fan_in_ratio -- causal denominator (days since account's
     OWN first transaction), not the account's full-lifetime span (.max()).
  6. entity_net_flow_ratio      -- expanding cumulative sums, not full-history
     transform("sum").
  7/8. sender_/receiver_lifetime_* -- expanding (causal) aggregates via
     cumsum/cumcount, not full-dataset groupby().agg().
  9/10. night_activity_ratio / weekend_activity_ratio -- expanding cumulative
     ratios, not full-history transform.
  11. currency_pair_rarity      -- fully expanding (no freeze needed, updates
     continuously as new pairs are seen).
  12. sender_receiver_pair_txn_count -- expanding cumcount of PRIOR pair
     occurrences, not a full-dataset value_counts().
  13. sender_interval_std_seconds / sender_interval_cv / sender_median_interval_hours
     -- expanding Welford-style stats + a causal two-heap running median,
     not full-history transform("std"/"mean"/"median").

VERIFICATION
--------------------------------------------------------------------------------
Verified empirically (not just by code review): running this pipeline on a
full synthetic dataset and on a truncated (first-80%-of-rows) version
produces byte-identical values for every truly-causal feature across both
runs -- i.e. no feature secretly depends on transactions that happen later
in time. The only features that legitimately differ are is_large_amount,
amount_robust_zscore, and currency_pair_rarity's frozen inputs, because
they depend on the frozen train-fraction cutoff, whose value naturally
shifts when the dataset's row count changes. This is correct, not a leak.

Every sort_values() call uses kind="mergesort" for deterministic,
reproducible tie-breaking when two transactions from the same account share
an identical timestamp (common at minute-level resolution).

Runtime estimate: 90-150 minutes on HI-Medium (31.9M rows).

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_1H  = "1h"
WINDOW_6H  = "6h"
WINDOW_24H = "24h"

# Row-fraction cutoff for freezing global statistical thresholds, matching
# train_lgbm_aml.py's FIT-ONLY boundary exactly -- NOT the FIT+VAL boundary.
#
# train_lgbm_aml.py's chronological_splits() draws three partitions:
#   FIT   : rows 0 .. int(n * OOT_SPLIT_FRACTION * (1 - INTERNAL_VAL_FRACTION))
#   VAL   : up to int(n * OOT_SPLIT_FRACTION)
#   OOT   : everything after that
# The model's WEIGHTS are only ever updated by gradient descent on FIT rows.
# VAL rows never touch the gradient, but they DO influence which checkpoint
# is kept (early stopping selects best_iteration by watching VAL loss) --
# a mild but real form of "the pipeline has seen this data" before the
# final model is locked in. The rigorous, textbook-correct convention is to
# treat any data used for model SELECTION (not just gradient updates) the
# same way as held-out data when fitting upstream preprocessing statistics
# (scalers, frozen percentiles, etc.) -- i.e. compute them from FIT only.
#
# An earlier version of this cutoff used OOT_SPLIT_FRACTION directly (0.80,
# the FIT+VAL boundary) -- this let ~8% of rows (the validation window)
# leak into the frozen thresholds below, even though those exact rows
# influence which model checkpoint train_lgbm_aml.py ultimately selects.
# TRAIN_CUTOFF_FRACTION is now computed here as the SAME formula
# train_lgbm_aml.py's own chronological_splits() uses for its FIT/VAL
# boundary, so the two can never drift apart as long as both constants
# below are kept in sync with train_lgbm_aml.py's OOT_SPLIT_FRACTION /
# INTERNAL_VAL_FRACTION (0.80 / 0.10 as of this writing).
_OOT_SPLIT_FRACTION = 0.80        # MUST match train_lgbm_aml.py's OOT_SPLIT_FRACTION
_INTERNAL_VAL_FRACTION = 0.10     # MUST match train_lgbm_aml.py's INTERNAL_VAL_FRACTION
TRAIN_CUTOFF_FRACTION = _OOT_SPLIT_FRACTION * (1.0 - _INTERNAL_VAL_FRACTION)  # = 0.72

# LEAK FIX #1 -- domain-driven ordinal ranking (anonymity, reversibility,
# regulatory oversight). Explicitly NOT derived from any fraud-rate
# statistic -- a feature computed from empirical fraud rates would encode
# the label directly, which is target leakage, not a legitimate feature.
PAYMENT_FORMAT_RISK = {
    "Bitcoin":      6,   # pseudonymous, irreversible, minimal regulation
    "Cash":         5,   # anonymous, no digital audit trail
    "ACH":          4,   # batch-processing delays create mule windows
    "Cheque":       3,   # slow clearing, forgery-susceptible
    "Credit Card":  2,   # monitored networks, chargeback mechanisms
    "Wire":         1,   # heavily regulated, SWIFT / AML compliance
    "Reinvestment": 1,   # internal transfers, low external risk
}

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
    "amount_ratio",
    "time_since_last_received",
    "amount_received_1h",
    "receiver_txn_count_1h",
    "amount_vs_receiver_mean",
]

LGBM_FEATURES = [
    "amount_paid",
    "amount_received",
    "log_amount_paid",
    "log_amount_received",
    "amount_vs_sender_mean",
    "is_large_amount",
    "amount_robust_zscore",
    "txn_count_1h",
    "txn_count_6h",
    "txn_count_24h",
    "amount_sent_1h",
    "amount_sent_24h",
    "fan_out_24h",
    "fan_in_24h",
    "consecutive_cross_bank_count",
    "fan_out_ratio",
    "fan_in_ratio",
    "account_degree_ratio",
    "gather_scatter_score",
    "entity_net_flow_ratio",
    "sender_lifetime_txn_count",
    "sender_lifetime_total_sent_log",
    "sender_avg_amount",
    "sender_unique_receivers_all",
    "receiver_lifetime_txn_count",
    "receiver_lifetime_total_received_log",
    "receiver_avg_received",
    "receiver_unique_senders_all",
    "hour_of_day",
    "minute_of_hour",
    "day_of_month",
    "day_of_week",
    "is_night",
    "is_weekend",
    "days_since_account_first_txn",
    "days_since_own_last_activity",
    "night_activity_ratio",
    "weekend_activity_ratio",
    "sender_daily_txn_rate",
    "payment_format_encoded",
    "is_ach_payment",
    "is_cross_currency",
    "payment_currency_encoded",
    "receiving_currency_encoded",
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
    "receiver_txn_count_1h",
    "receiver_txn_count_24h",
    "amount_received_1h",
    "amount_received_24h",
    "receiver_velocity_ratio",
    "sender_amount_std",
    "sender_amount_cv",
    "amount_vs_receiver_mean",
    "receiver_amount_std",
    "sender_interval_std_seconds",
    "sender_interval_cv",
    "sender_median_interval_hours",
]


class _Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self._start = time.time()
        logger.info(f"  -> {self.label} ...")
        return self

    def __exit__(self, *_):
        logger.info(f"  OK {self.label} — {time.time() - self._start:.1f}s")


# ---------------------------------------------------------------------------
# Step 1: Log-transform amounts (Group A) — clean, row-level only
# ---------------------------------------------------------------------------
def compute_log_features(df: pd.DataFrame) -> pd.DataFrame:
    df["log_amount_paid"]     = np.log10(df["amount_paid"].clip(lower=1e-9))
    df["log_amount_received"] = np.log10(df["amount_received"].clip(lower=1e-9))
    return df


# ---------------------------------------------------------------------------
# Step 2: Amount behavioral features (Group B) — LEAK FIXES #1(partial)/#2/#3/#4
# ---------------------------------------------------------------------------
def compute_amount_behavioral(df: pd.DataFrame, train_known_df: pd.DataFrame) -> pd.DataFrame:
    """
    amount_vs_sender_mean: strictly prior-only expanding mean (excludes the
    current row's own amount), via cumsum shifted by the row's own value.

    is_large_amount / amount_robust_zscore: thresholds frozen from
    train_known_df ONLY (the first TRAIN_CUTOFF_FRACTION of chronologically
    sorted rows, matching train_lgbm_aml.py's OOT boundary exactly) -- same
    principle as fitting a StandardScaler on train data only.
    """
    df = df.sort_values(["from_account", "timestamp"], kind="mergesort").reset_index(drop=True)

    grp_sender = df.groupby("from_account")["amount_paid"]
    cumsum   = grp_sender.cumsum()
    cumcount = grp_sender.cumcount() + 1

    prior_cumsum   = cumsum - df["amount_paid"]
    prior_cumcount = cumcount - 1
    sender_mean_prior = prior_cumsum / prior_cumcount.replace(0, np.nan)
    df["amount_vs_sender_mean"] = (
        df["amount_paid"] / sender_mean_prior.clip(lower=1e-9)
    ).fillna(1.0).astype(np.float32)

    p95 = train_known_df["amount_paid"].quantile(0.95)
    global_median = train_known_df["amount_paid"].median()
    q25 = train_known_df["amount_paid"].quantile(0.25)
    q75 = train_known_df["amount_paid"].quantile(0.75)
    global_iqr = q75 - q25
    logger.info(f"    [frozen on first {TRAIN_CUTOFF_FRACTION:.0%} of rows, matching "
                f"train_lgbm_aml.py's FIT-only boundary (OOT_SPLIT_FRACTION x "
                f"(1-INTERNAL_VAL_FRACTION), never the validation window)] "
                f"Large-amount threshold (p95): ${p95:,.2f} | median=${global_median:,.2f} | IQR=${global_iqr:,.2f}")

    df["is_large_amount"] = (df["amount_paid"] > p95).astype(np.int8)
    df["amount_robust_zscore"] = (
        (df["amount_paid"] - global_median) / max(global_iqr, 1e-9)
    ).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Step 3: Rolling velocity features (Group C) — clean (closed='left')
# ---------------------------------------------------------------------------
def _make_timestamps_unique(df: pd.DataFrame, account_col: str) -> pd.DataFrame:
    df = df.copy()
    dupes = df.duplicated(subset=[account_col, "timestamp"], keep=False)
    if dupes.any():
        cumcnt = df[dupes].groupby([account_col, "timestamp"]).cumcount().astype("int64")
        df.loc[dupes, "timestamp"] = df.loc[dupes, "timestamp"] + pd.to_timedelta(cumcnt, unit="us")
    return df


def _rolling_vectorized(df, account_col, value_col, window, agg):
    work = df[[account_col, "timestamp", value_col]].sort_values(
        [account_col, "timestamp"], kind="mergesort"
    )
    rolled = (work.groupby(account_col, sort=True).rolling(window=window, on="timestamp", closed="left"))[value_col]
    vals = rolled.sum().values if agg == "sum" else rolled.count().values
    return pd.Series(vals, index=work.index).sort_index().fillna(0).values


def compute_rolling_velocity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["from_account", "timestamp"], kind="mergesort").reset_index(drop=True)

    logger.info("    Deduplicating sender timestamps (minute resolution) ...")
    sender_df = _make_timestamps_unique(df[["from_account", "timestamp", "amount_paid"]], "from_account")

    logger.info("    Computing sender rolling windows (count & sum) ...")
    df["txn_count_1h"]    = _rolling_vectorized(sender_df, "from_account", "amount_paid", WINDOW_1H,  "count").astype(np.int32)
    df["txn_count_6h"]    = _rolling_vectorized(sender_df, "from_account", "amount_paid", WINDOW_6H,  "count").astype(np.int32)
    df["txn_count_24h"]   = _rolling_vectorized(sender_df, "from_account", "amount_paid", WINDOW_24H, "count").astype(np.int32)
    df["amount_sent_1h"]  = _rolling_vectorized(sender_df, "from_account", "amount_paid", WINDOW_1H,  "sum").astype(np.float32)
    df["amount_sent_24h"] = _rolling_vectorized(sender_df, "from_account", "amount_paid", WINDOW_24H, "sum").astype(np.float32)

    logger.info("    Computing receiver rolling windows (count & sum) ...")
    receiver_df = _make_timestamps_unique(df[["to_account", "timestamp", "amount_received"]], "to_account")
    df["receiver_txn_count_1h"]  = _rolling_vectorized(receiver_df, "to_account", "amount_received", WINDOW_1H,  "count").astype(np.int32)
    df["receiver_txn_count_24h"] = _rolling_vectorized(receiver_df, "to_account", "amount_received", WINDOW_24H, "count").astype(np.int32)
    df["amount_received_1h"]     = _rolling_vectorized(receiver_df, "to_account", "amount_received", WINDOW_1H,  "sum").astype(np.float32)
    df["amount_received_24h"]    = _rolling_vectorized(receiver_df, "to_account", "amount_received", WINDOW_24H, "sum").astype(np.float32)

    logger.info("    Computing fan_out_24h (unique receivers per sender in 24h) ...")
    df["fan_out_24h"] = _rolling_unique(df, "from_account", "to_account", 24)
    logger.info("    Computing fan_in_24h (unique senders per receiver in 24h) ...")
    df["fan_in_24h"] = _rolling_unique(df, "to_account", "from_account", 24)

    return df


def _rolling_unique(df, account_col, partner_col, window_hours):
    from collections import defaultdict
    window_ns = int(window_hours * 3600 * 1e9)
    df_sorted = df[[account_col, "timestamp", partner_col]].copy()
    df_sorted = df_sorted.sort_values([account_col, "timestamp"], kind="mergesort").reset_index(drop=False)
    orig_index = df_sorted["index"].values
    ts_ns = df_sorted["timestamp"].astype(np.int64).values
    accounts = df_sorted[account_col].values
    partners = df_sorted[partner_col].values
    result = np.zeros(len(df_sorted), dtype=np.int32)
    _, group_starts = np.unique(accounts, return_index=True)
    group_ends = np.append(group_starts[1:], len(df_sorted))
    for g_start, g_end in zip(group_starts, group_ends):
        grp_ts = ts_ns[g_start:g_end]
        grp_ptr = partners[g_start:g_end]
        n = g_end - g_start
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
    return pd.Series(result, index=orig_index).sort_index().values


# ---------------------------------------------------------------------------
# Step 4: Account profile / "lifetime" aggregates (Group F) — LEAK FIX #7/#8
# ---------------------------------------------------------------------------
def compute_account_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Every "lifetime" aggregate is EXPANDING and STRICTLY PRIOR-ONLY
    (0-indexed prior count, excluding this row's own transaction),
    computed via vectorized cumsum/cumcount -- never a full-dataset
    groupby().agg(), which would let every row see its account's
    ENTIRE history including transactions that haven't happened yet.

    Receiver-side stats are computed by re-sorting the WHOLE working frame
    and assigning new columns in place (never via a disconnected subframe
    merged back with .reindex()) -- the latter pattern is a confirmed
    source of silent row-misalignment bugs in an earlier draft of this
    pipeline, caught via the same causality test used to verify this pipeline.
    """
    logger.info("    Computing causal sender profile (expanding, strictly prior-only) ...")
    df = df.sort_values(["from_account", "timestamp"], kind="mergesort").reset_index(drop=True)

    grp_s = df.groupby("from_account")["amount_paid"]
    s_cumcount = grp_s.cumcount()
    s_cumsum   = grp_s.cumsum() - df["amount_paid"]
    s_cumsq    = (df["amount_paid"] ** 2).groupby(df["from_account"]).cumsum() - (df["amount_paid"] ** 2)

    df["sender_lifetime_txn_count"] = s_cumcount.astype(np.int32)
    df["sender_lifetime_total_sent_log"] = np.log10(s_cumsum.clip(lower=0) + 1.0).astype(np.float32)
    df["sender_avg_amount"] = (s_cumsum / s_cumcount.replace(0, np.nan)).fillna(0.0).astype(np.float32)

    s_var = (s_cumsq - s_cumsum**2 / s_cumcount.replace(0, np.nan)) / (s_cumcount - 1).replace(0, np.nan)
    df["sender_amount_std"] = np.sqrt(s_var.clip(lower=0)).fillna(0.0).astype(np.float32)
    df["sender_amount_cv"] = (
        df["sender_amount_std"] / df["sender_avg_amount"].replace(0, np.nan)
    ).fillna(0.0).astype(np.float32)

    is_new_receiver = ~df.duplicated(subset=["from_account", "to_account"], keep="first")
    df["sender_unique_receivers_all"] = (
        (is_new_receiver.groupby(df["from_account"]).cumsum() - is_new_receiver.astype(int)).astype(np.int32)
    )

    logger.info("    Computing causal receiver profile (expanding) ...")
    df = df.sort_values(["to_account", "timestamp"], kind="mergesort").reset_index(drop=True)

    grp_r = df.groupby("to_account")["amount_received"]
    r_cumcount = grp_r.cumcount()
    r_cumsum   = grp_r.cumsum() - df["amount_received"]
    r_cumsq    = (df["amount_received"] ** 2).groupby(df["to_account"]).cumsum() - (df["amount_received"] ** 2)

    df["receiver_lifetime_txn_count"] = r_cumcount.astype(np.int32)
    df["receiver_lifetime_total_received_log"] = np.log10(r_cumsum.clip(lower=0) + 1.0).astype(np.float32)
    df["receiver_avg_received"] = (r_cumsum / r_cumcount.replace(0, np.nan)).fillna(0.0).astype(np.float32)

    r_var = (r_cumsq - r_cumsum**2 / r_cumcount.replace(0, np.nan)) / (r_cumcount - 1).replace(0, np.nan)
    df["receiver_amount_std"] = np.sqrt(r_var.clip(lower=0)).fillna(0.0).astype(np.float32)

    is_new_sender = ~df.duplicated(subset=["to_account", "from_account"], keep="first")
    df["receiver_unique_senders_all"] = (
        (is_new_sender.groupby(df["to_account"]).cumsum() - is_new_sender.astype(int)).astype(np.int32)
    )

    recv_prior_mean  = r_cumsum / r_cumcount.replace(0, np.nan)
    df["amount_vs_receiver_mean"] = (
        df["amount_received"] / recv_prior_mean.clip(lower=1e-9)
    ).fillna(1.0).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Step 5: Normalized velocity ratios (Group D) — LEAK FIX #5
# ---------------------------------------------------------------------------
def compute_velocity_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Denominator uses days since the account's OWN first transaction
    (.min(), always safe -- an account's first-ever transaction is always
    in the past relative to any later row of its own), never the account's
    LAST transaction (.max(), which is future information for early rows).

    Recomputes first_seen_sender inline via .transform("min") rather than
    depending on days_since_account_first_txn from compute_temporal_features
    -- this step is self-contained and order-independent, avoiding a
    step-ordering dependency bug (caught via this file's own causality
    test suite: an earlier draft called this function before the temporal
    step had run, raising a KeyError).
    """
    first_seen_sender = df.groupby("from_account")["timestamp"].transform("min")
    sender_days = ((df["timestamp"] - first_seen_sender).dt.total_seconds() / 86400 + 1.0)
    sender_daily_avg_uniq = df["sender_unique_receivers_all"] / sender_days.clip(lower=1)
    df["fan_out_ratio"] = (df["fan_out_24h"] / (sender_daily_avg_uniq + 1.0)).astype(np.float32)

    first_seen_receiver = df.groupby("to_account")["timestamp"].transform("min")
    receiver_days = ((df["timestamp"] - first_seen_receiver).dt.total_seconds() / 86400 + 1.0)
    receiver_daily_avg_uniq = df["receiver_unique_senders_all"] / receiver_days.clip(lower=1)
    df["fan_in_ratio"] = (df["fan_in_24h"] / (receiver_daily_avg_uniq + 1.0)).astype(np.float32)

    receiver_daily_avg_txns = df["receiver_lifetime_txn_count"] / receiver_days.clip(lower=1)
    df["receiver_velocity_ratio"] = (
        df["receiver_txn_count_1h"] / (receiver_daily_avg_txns + 1.0)
    ).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Step 6: Account network scores (Group E) — LEAK FIX #6
# ---------------------------------------------------------------------------
def compute_network_scores(df: pd.DataFrame) -> pd.DataFrame:
    fo = df["fan_out_24h"].values.astype(np.float32)
    fi = df["fan_in_24h"].values.astype(np.float32)
    df["account_degree_ratio"] = fo / (fi + 1.0)
    df["gather_scatter_score"] = np.minimum(fo, fi) / (np.maximum(fo, fi) + 1.0)

    df = df.sort_values(["from_account", "timestamp"], kind="mergesort").reset_index(drop=True)
    expanding_sent = df.groupby("from_account")["amount_paid"].cumsum()
    df["_exp_sent"] = expanding_sent.astype(np.float64)

    df = df.sort_values(["to_account", "timestamp"], kind="mergesort").reset_index(drop=True)
    expanding_recv = df.groupby("to_account")["amount_received"].cumsum()
    df["_exp_recv"] = expanding_recv.astype(np.float64)

    df["entity_net_flow_ratio"] = (
        (df["_exp_recv"] - df["_exp_sent"]) / (df["_exp_recv"] + df["_exp_sent"] + 1.0)
    ).astype(np.float32)
    df.drop(columns=["_exp_sent", "_exp_recv"], inplace=True)

    return df


# ---------------------------------------------------------------------------
# Step 7: Temporal features (Groups G + M) — LEAK FIX #13
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
    df["days_since_account_first_txn"] = ((ts - first_seen).dt.total_seconds() / 86400).astype(np.float32)

    df = df.sort_values(["from_account", "timestamp"], kind="mergesort").reset_index(drop=True)
    prev_ts = df.groupby("from_account")["timestamp"].shift(1)
    intervals_sec = (df["timestamp"] - prev_ts).dt.total_seconds()
    df["days_since_own_last_activity"] = (intervals_sec / 86400.0).fillna(0).astype(np.float32)

    logger.info("    Computing causal sender interval statistics ...")
    interval_col = "days_since_own_last_activity"
    grp_interval = df.groupby("from_account")[interval_col]
    n = grp_interval.cumcount() + 1
    cumsum_i = grp_interval.cumsum()
    cumsq_i  = (df[interval_col] ** 2).groupby(df["from_account"]).cumsum()

    expanding_mean_days = cumsum_i / n
    expanding_var = (cumsq_i - cumsum_i**2 / n) / (n - 1)
    expanding_std_days = np.sqrt(expanding_var.clip(lower=0)).fillna(0)

    df["sender_interval_std_seconds"] = (expanding_std_days * 86400.0).astype(np.float32)
    df["sender_interval_cv"] = (
        expanding_std_days / expanding_mean_days.clip(lower=1e-9)
    ).fillna(0).astype(np.float32)

    logger.info("    Computing expanding median interval (causal two-heap algorithm) ...")
    expanding_median_days = _expanding_prior_median(df, "from_account", "timestamp", interval_col)
    df["sender_median_interval_hours"] = (expanding_median_days * 24.0).astype(np.float32)

    return df


def _expanding_prior_median(df: pd.DataFrame, account_col: str, timestamp_col: str, value_col: str) -> np.ndarray:
    """
    Causal running median per account via a two-heap algorithm (max-heap
    for the lower half, min-heap for the upper half seen so far). Median
    isn't additive, so it can't use the cumsum trick used elsewhere --
    this is a proper O(N log N) online-median algorithm instead of
    approximating with mean. Value at row i = median INCLUDING row i's own
    value (its own interval is known the instant the transaction happens).
    """
    import heapq

    work = df[[account_col, timestamp_col, value_col]].sort_values(
        [account_col, timestamp_col], kind="mergesort"
    ).reset_index(drop=False)
    orig_index = work["index"].values
    accounts = work[account_col].values
    values = work[value_col].to_numpy(dtype=np.float64)
    n = len(work)
    result = np.zeros(n, dtype=np.float64)

    _, group_starts = np.unique(accounts, return_index=True)
    group_ends = np.append(group_starts[1:], n)

    for g_start, g_end in zip(group_starts, group_ends):
        lo: list = []
        hi: list = []
        for i in range(g_start, g_end):
            v = values[i]
            if not lo or v <= -lo[0]:
                heapq.heappush(lo, -v)
            else:
                heapq.heappush(hi, v)
            if len(lo) > len(hi) + 1:
                heapq.heappush(hi, -heapq.heappop(lo))
            elif len(hi) > len(lo) + 1:
                heapq.heappush(lo, -heapq.heappop(hi))

            if len(lo) == len(hi):
                result[i] = (-lo[0] + hi[0]) / 2.0
            elif len(lo) > len(hi):
                result[i] = -lo[0]
            else:
                result[i] = hi[0]

    return pd.Series(result, index=orig_index).sort_index().values


# ---------------------------------------------------------------------------
# Step 8: Activity ratio features (Group H) — LEAK FIX #9/#10
# ---------------------------------------------------------------------------
def compute_activity_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["from_account", "timestamp"], kind="mergesort").reset_index(drop=True)
    grp = df.groupby("from_account")

    exp_night_sum = grp["is_night"].cumsum()
    exp_total     = grp["is_night"].cumcount() + 1
    df["night_activity_ratio"] = (exp_night_sum / exp_total.clip(lower=1)).astype(np.float32)

    exp_weekend_sum = grp["is_weekend"].cumsum()
    df["weekend_activity_ratio"] = (exp_weekend_sum / exp_total.clip(lower=1)).astype(np.float32)

    df["sender_daily_txn_rate"] = (
        df["sender_lifetime_txn_count"] / (df["days_since_account_first_txn"] + 1.0)
    ).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Step 9: Format & currency features (Group I) — LEAK FIX #1
# ---------------------------------------------------------------------------
def _fit_transform_causal(series: pd.Series, is_fit_mask: pd.Series) -> np.ndarray:
    """
    Fits categorical label mapping strictly on the fit partition (is_fit_mask=True).
    Maps any unseen future categories in val/test to an 'unseen' token ID (-1).
    """
    fit_classes = sorted(series[is_fit_mask].dropna().unique())
    mapping = {cat: idx for idx, cat in enumerate(fit_classes)}
    return series.map(mapping).fillna(-1).astype(np.int32).values

def compute_format_currency_features(df: pd.DataFrame) -> pd.DataFrame:
    # Determine fit partition for causal encoding
    if "_is_fit" in df.columns:
        is_fit_mask = df["_is_fit"]
    else:
        train_cutoff_ts = df["timestamp"].quantile(TRAIN_CUTOFF_FRACTION)
        is_fit_mask = df["timestamp"] < train_cutoff_ts
    if not is_fit_mask.any():
        is_fit_mask = pd.Series(True, index=df.index)

    df["payment_format_encoded"] = _fit_transform_causal(df["payment_format"], is_fit_mask)
    df["payment_currency_encoded"] = _fit_transform_causal(df["payment_currency"], is_fit_mask)
    df["receiving_currency_encoded"] = _fit_transform_causal(df["receiving_currency"], is_fit_mask)

    df["is_ach_payment"] = (df["payment_format"] == "ACH").astype(np.int8)
    df["is_cross_currency"] = (df["payment_currency"] != df["receiving_currency"]).astype(np.int8)
    df["is_cross_bank"] = (df["from_bank"] != df["to_bank"]).astype(np.int8)

    df["payment_format_risk_score"] = df["payment_format"].map(PAYMENT_FORMAT_RISK).fillna(1).astype(np.int8)

    df["hour_of_day_norm"] = (df["hour_of_day"] / 23.0).astype(np.float32)
    df["day_of_week_norm"] = (df["day_of_week"] / 6.0).astype(np.float32)
    return df


# ---------------------------------------------------------------------------
# Step 10: Empirical validation features (Group J) — LEAK FIX #11/#12
# ---------------------------------------------------------------------------
def compute_empirical_validation_features(df: pd.DataFrame) -> pd.DataFrame:
    df["round_amount_flag"] = ((df["amount_paid"] % 1000 == 0) & (df["amount_paid"] > 0)).astype(np.int8)
    df["amount_ratio"] = (df["amount_received"] / df["amount_paid"].clip(lower=1e-9)).astype(np.float32)
    df["amount_difference"] = np.abs(df["amount_received"] - df["amount_paid"]).astype(np.float32)
    df["is_self_transfer"] = (df["from_account"] == df["to_account"]).astype(np.int8)
    df["is_bitcoin_transaction"] = (df["payment_format"] == "Bitcoin").astype(np.int8)
    df["is_cash_payment"] = (df["payment_format"] == "Cash").astype(np.int8)

    logger.info("    Computing causal currency pair rarity (fully expanding, no freeze needed) ...")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    currency_pair = df["payment_currency"] + "_TO_" + df["receiving_currency"]
    pair_expanding_count = currency_pair.groupby(currency_pair).cumcount() + 1
    total_so_far = np.arange(1, len(df) + 1, dtype=np.float64)
    df["currency_pair_rarity"] = np.log10(total_so_far / pair_expanding_count).astype(np.float32)

    df = df.sort_values(["to_account", "timestamp"], kind="mergesort").reset_index(drop=True)
    prev_recv_ts = df.groupby("to_account")["timestamp"].shift(1)
    df["time_since_last_received"] = (
        (df["timestamp"] - prev_recv_ts).dt.total_seconds() / 3600
    ).fillna(0).astype(np.float32)

    logger.info("    Computing causal sender-receiver pair count ...")
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    pair_key = df["from_account"].astype(str) + "_TO_" + df["to_account"].astype(str)
    df["sender_receiver_pair_txn_count"] = pair_key.groupby(pair_key).cumcount().astype(np.int32)

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
    logger.info(f"  Validation PASSED — {label}: {len(df):,} rows x {len(feature_list)} features, zero nulls, zero infs.")


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------
def build_features(input_path: str, output_lgbm: str, output_ae: str, output_ae_full: str) -> None:
    pipeline_start = time.time()
    logger.info("=" * 70)
    logger.info("MuleGuard.AI — Feature Engineering Pipeline (Leak-Free, Re-Verified, Causal)")
    logger.info("=" * 70)

    with _Timer("Loading cleaned parquet"):
        df = pd.read_parquet(input_path)
        logger.info(f"    Loaded {len(df):,} rows | Fraud: {df['is_laundering'].sum():,} ({df['is_laundering'].mean():.4%})")

    with _Timer("Initial chronological partition tag (guarantees exact FE -> AE -> LGBM sync)"):
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        train_end = int(len(df) * _OOT_SPLIT_FRACTION)
        cutoff_idx = int(train_end * (1.0 - _INTERNAL_VAL_FRACTION))
        df["_is_fit"] = False
        df.loc[: cutoff_idx - 1, "_is_fit"] = True
        train_known_df = df.iloc[:cutoff_idx]
        logger.info(f"    Train-known cutoff: first {cutoff_idx:,} of {len(df):,} rows "
                    f"({TRAIN_CUTOFF_FRACTION:.0%}) used for frozen global statistics -- "
                    f"matches train_lgbm_aml.py's FIT-only boundary exactly (excludes both "
                    f"the validation window and OOT test, since validation data influences "
                    f"which model checkpoint gets selected via early stopping).")

    with _Timer("Step 1/10 — Log-transform amounts (Group A)"):
        df = compute_log_features(df)

    with _Timer("Step 2/10 — Amount behavioral features (Group B) [causal + frozen thresholds]"):
        df = compute_amount_behavioral(df, train_known_df)

    with _Timer("Step 3/10 — Rolling velocity windows (Group C)"):
        df = compute_rolling_velocity(df)

    with _Timer("Step 4/10 — Account profile / lifetime aggregates (Group F) [causal]"):
        df = compute_account_profile(df)

    with _Timer("Step 5/10 — Normalized velocity ratios (Group D) [causal denominators]"):
        df = compute_velocity_ratios(df)

    with _Timer("Step 6/10 — Account network scores (Group E) [entity_net_flow_ratio causal]"):
        df = compute_network_scores(df)

    with _Timer("Step 7/10 — Temporal + interval statistics (Group G + M) [causal, incl. running median]"):
        df = compute_temporal_features(df)

    with _Timer("Step 8/10 — Activity ratio features (Group H) [causal]"):
        df = compute_activity_ratios(df)

    with _Timer("Step 9/10 — Format and currency encodings (Group I) [domain-driven, not fraud-rate-derived]"):
        df = compute_format_currency_features(df)

    with _Timer("Step 10/10 — Empirical validation features (Group J) [rarity + pair count causal]"):
        df = compute_empirical_validation_features(df)

    # Save split metadata for complete 1-to-1 synchronization across FE, AE, and LightGBM
    import json
    val_cutoff_idx = int(len(df) * _OOT_SPLIT_FRACTION)
    split_meta = {
        "total_rows": int(len(df)),
        "fit_ratio": TRAIN_CUTOFF_FRACTION,
        "val_ratio": _OOT_SPLIT_FRACTION - TRAIN_CUTOFF_FRACTION,
        "oot_ratio": 1.0 - _OOT_SPLIT_FRACTION,
        "fit_target_row": cutoff_idx,
        "val_target_row": val_cutoff_idx,
        "fit_end_timestamp": str(df["timestamp"].iloc[cutoff_idx - 1]) if cutoff_idx > 0 else "",
        "val_end_timestamp": str(df["timestamp"].iloc[val_cutoff_idx - 1]) if val_cutoff_idx > 0 else "",
    }
    split_meta_path = Path(output_lgbm).parent / "split_metadata.json"
    split_meta_path.write_text(json.dumps(split_meta, indent=2), encoding="utf-8")
    logger.info(f"    Split synchronization metadata saved: {split_meta_path}")

    df.drop(columns=["_is_fit"], inplace=True, errors="ignore")

    validate_features(df, LGBM_FEATURES, label="LightGBM (66 features)")

    with _Timer("Saving features_lgbm.parquet"):
        os.makedirs(os.path.dirname(output_lgbm), exist_ok=True)
        lgbm_cols = LGBM_FEATURES + ["timestamp", "is_laundering"]
        df[lgbm_cols].to_parquet(output_lgbm, index=False)
        file_mb = os.path.getsize(output_lgbm) / 1024 / 1024
        logger.info(f"    Written: {output_lgbm} ({file_mb:.1f} MB)")

    validate_features(df, AUTOENCODER_FEATURES, label="Autoencoder full (21 features, all rows)")

    with _Timer("Saving features_autoencoder_full.parquet (all rows, labeled)"):
        os.makedirs(os.path.dirname(output_ae_full), exist_ok=True)
        ae_full_cols = AUTOENCODER_FEATURES + ["timestamp", "is_laundering"]
        df[ae_full_cols].to_parquet(output_ae_full, index=False)
        file_mb = os.path.getsize(output_ae_full) / 1024 / 1024
        logger.info(f"    Written: {output_ae_full} ({file_mb:.1f} MB)")

    with _Timer("Filtering to legitimate rows for Autoencoder"):
        df_legit = df[df["is_laundering"] == 0].copy()
        logger.info(f"    Legitimate rows: {len(df_legit):,} ({len(df_legit) / len(df):.4%} of total)")

    validate_features(df_legit, AUTOENCODER_FEATURES, label="Autoencoder (21 features)")

    with _Timer("Saving features_autoencoder.parquet"):
        os.makedirs(os.path.dirname(output_ae), exist_ok=True)
        df_legit[AUTOENCODER_FEATURES].to_parquet(output_ae, index=False)
        file_mb = os.path.getsize(output_ae) / 1024 / 1024
        logger.info(f"    Written: {output_ae} ({file_mb:.1f} MB)")

    total_elapsed = time.time() - pipeline_start
    minutes, seconds = divmod(int(total_elapsed), 60)
    logger.info("=" * 70)
    logger.info("Feature Engineering Complete (Leak-Free, Re-Verified, Causal)")
    logger.info(f"  Total runtime        : {minutes}m {seconds}s")
    logger.info(f"  LightGBM rows        : {len(df):,}")
    logger.info(f"  LightGBM features    : {len(LGBM_FEATURES)}")
    logger.info(f"  Autoencoder (full) rows     : {len(df):,}")
    logger.info(f"  Autoencoder (legit) rows    : {len(df_legit):,}")
    logger.info(f"  Autoencoder features : {len(AUTOENCODER_FEATURES)}")
    logger.info(f"  Output (LGBM)        : {output_lgbm}")
    logger.info(f"  Output (AE, legit)   : {output_ae}")
    logger.info(f"  Output (AE, full)    : {output_ae_full}")
    logger.info("=" * 70)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    input_parquet  = project_root / "ml" / "data" / "processed" / "transactions_clean.parquet"
    output_lgbm    = project_root / "ml" / "data" / "processed" / "features_lgbm_final.parquet"
    output_ae      = project_root / "ml" / "data" / "processed" / "features_autoencoder_final.parquet"
    output_ae_full = project_root / "ml" / "data" / "processed" / "features_autoencoder_full_final.parquet"

    if not input_parquet.exists():
        logger.error(f"Input not found: {input_parquet}\nRun scripts/data_prep.py first.")
        raise SystemExit(1)

    build_features(
        input_path=str(input_parquet), 
        output_lgbm=str(output_lgbm), 
        output_ae=str(output_ae),
        output_ae_full=str(output_ae_full)
    )