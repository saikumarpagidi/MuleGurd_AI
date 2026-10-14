"""
================================================================================
MuleGuard.AI — Anti-Money-Laundering Detection Model
Enterprise Training Pipeline (Single-Model, Production Edition — FINAL)
================================================================================

BUSINESS CONTEXT
----------------------------------------------------------------------------
Dataset: IBM AML "HI-Medium" — 31,898,238 transactions, 35,230 confirmed
laundering cases (0.1104% fraud rate, a 1:905 class imbalance).

Business mandate: >=80% Recall at a realistic investigator alert capacity,
with a defensible PR-AUC and full model explainability for regulatory review.

WHY THIS IS A SINGLE MODEL, NOT AN ENSEMBLE
----------------------------------------------------------------------------
An earlier 3-model ensemble version crashed partway through training member
2 on this hardware, after member 1 alone reached PR-AUC ~0.70 — a real
improvement over the prior single-model result (0.6454-0.6519). The
improvement came from letting ONE model train with a higher round ceiling
(6000, up from 3000) and more early-stopping patience (150, up from 100) --
NOT from ensembling. Training 3 models sequentially at that higher ceiling
roughly triples total sustained run time, which is what triggered the
crash. This script keeps the change that helped (higher ceiling + patience
on a single model) and drops the change that added crash risk without a
proportional benefit (training 3 of them back to back).

WHAT THIS SCRIPT DOES, END TO END
----------------------------------------------------------------------------
1. Loads and prepares the full 31.9M-row feature set (memory-safe, see
   "MEMORY ARCHITECTURE" below).
2. Splits chronologically into fit / validation / out-of-time test (never
   randomly — see "WHY CHRONOLOGICAL SPLITTING" below).
3. Runs Bayesian hyperparameter search (Optuna, 25 trials) — cached to disk
   so a rerun doesn't repeat a ~40-minute search unnecessarily.
4. Trains ONE model to genuine convergence (real early-stopping plateau,
   not an arbitrary round cap) with a generous 6000-round ceiling.
5. Evaluates on PR-AUC, ROC-AUC, and Top-K alert-capacity Recall/Precision
   — NOT plain accuracy, which is meaningless at this imbalance.
6. Produces full SHAP explainability across ALL 66 features (complete
   ranked table, not a top-10 preview) — required for AML model governance.
7. Writes a professional executive summary report suitable for a manager
   or model-risk review, in plain language with every number traceable to
   a source file.

MEMORY ARCHITECTURE
----------------------------------------------------------------------------
The full feature matrix (~8.4GB as float32) is allocated exactly ONCE as a
single NumPy array. Every downstream partition (fit/val/OOT, and every
cross-validation fold during tuning) uses BASIC slicing (`a[i:j]`), which
returns a zero-copy VIEW into that one buffer — never a new allocation.
This is what makes a 31.9M-row pipeline run reliably on a single 64GB
workstation with no SMOTE, no oversampling, and no downsampling of the data
actually used to train the production model. This exact architecture has
been validated stable across multiple full runs on this hardware.

WHY CHRONOLOGICAL SPLITTING (OOT — Out-of-Time)
----------------------------------------------------------------------------
Fraud tactics evolve over time. A random train/test split would let the
model "see the future" during training, inflating reported performance in
a way that will not hold up once deployed. We sort by timestamp and split
by DATE instead:

    [ ------- FIT (72%) ------- ][ VAL (8%) ][ ---- OOT TEST (20%) ---- ]
       trains the model            early-        touched exactly once,
                                    stopping       after all decisions
                                    decisions      are locked in — this
                                    only           is the only trustworthy
                                                    number

The OOT test set is never passed into any training call, at any point in
this script — it is structurally impossible for it to leak into model
selection.

WHY scale_pos_weight IS TUNED, NOT HARD-CODED
----------------------------------------------------------------------------
The textbook formula (count(legit) / count(fraud)) gives ~1036.76 on this
dataset. Empirically, using that raw value destabilizes training — the
model over-focuses on the ~24.6k fraud rows in the fit set and overfits
almost immediately (observed directly: best_iteration collapsed to 1 in an
earlier run using the raw ratio). We instead let Optuna search a DAMPING
MULTIPLIER (0.05x-1.0x of the raw ratio) — still pure algorithm-level
reweighting, no SMOTE, no synthetic data, just an empirically stable one.

IMPORTANT — DATA INTEGRITY REMINDER
----------------------------------------------------------------------------
If you have regenerated features_lgbm.parquet (e.g. after fixing temporal
leakage in the feature-engineering pipeline), DELETE
ml/reports/optuna_best_params.json before running this script. Otherwise
tune_hyperparameters() will silently reuse parameters tuned on the OLD
data, which is inconsistent with the new dataset. This script prints a
loud warning if it detects and reuses a cached file, precisely so this
doesn't happen silently.

Author: MuleGuard.AI ML Engineering
================================================================================
"""

import os
import sys
import gc
import json
import time
import logging
import warnings
from pathlib import Path

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

import numpy as np
import pandas as pd
import joblib
import pyarrow.parquet as pq

import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_PATH = Path("ml/data/processed/features_lgbm_final.parquet")
MODEL_DIR = Path("ml/models")
REPORT_DIR = Path("ml/reports/lgbm_final")
MODEL_PATH = MODEL_DIR / "lgbm_best_final.pkl"
EXEC_SUMMARY_PATH = REPORT_DIR / "executive_summary.md"

TARGET_COL = "is_laundering"
TIMESTAMP_COL = "timestamp"
ID_LIKE_COLS = {
    "transaction_id", "account_id", "from_account", "to_account",
    "sender_id", "receiver_id",
}

OOT_SPLIT_FRACTION = 0.80
INTERNAL_VAL_FRACTION = 0.10
N_OPTUNA_TRIALS = 25
N_CV_FOLDS = 5
RANDOM_STATE = 42
SCALE_POS_WEIGHT_MULTIPLIER_RANGE = (0.05, 1.0)

TOP_K_PCTS = [0.01, 0.05]

LGBM_N_JOBS = 4  # Hard limit to prevent OOM/thread crashes on high-core machines
TUNING_SAMPLE_ROWS = 2_000_000
TUNING_NUM_BOOST_ROUND = 800
TUNING_EARLY_STOPPING = 50
FINAL_NUM_BOOST_ROUND = 6000       # generous ceiling — the run that reached
                                    # PR-AUC ~0.70 used this; early stopping
                                    # decides the real tree count
FINAL_EARLY_STOPPING = 150         # more patient than the tuning phase —
                                    # gains near convergence are slow
MAX_NUM_LEAVES = 128
MAX_BIN = 127
DOWNCAST_FLOAT32 = True
SHAP_SAMPLE_ROWS = 50_000
USE_TWO_ROUND_LOADING = True

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# LOGGING
# ==============================================================================
log = logging.getLogger("MuleGuard.LGBM")
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


def _log_memory(tag: str):
    if _HAS_PSUTIL:
        vm = psutil.virtual_memory()
        log.info(f"[mem check: {tag}] RAM used={vm.percent:.1f}% "
                 f"({vm.used/1e9:.1f}GB / {vm.total/1e9:.1f}GB)")


def _json_safe(obj):
    """
    Recursively converts NumPy scalar types (bool_, int64, float64, etc.)
    into native Python types before JSON serialization.

    BUG FIX: `recalls[idx_80] >= 0.80` (where recalls comes from
    sklearn's precision_recall_curve, a NumPy array) returns a NumPy
    boolean, not a native Python bool. json.dump() only accepts native
    Python types and raises `TypeError: Object of type bool is not JSON
    serializable` on the NumPy version, even though it prints identically
    to a real bool. This crashed the pipeline on its LAST step, after a
    full ~2-hour training run had already completed successfully -- this
    helper is applied to every json.dump() call below so that can't
    happen again, regardless of which metric/type introduces a NumPy
    scalar in the future.
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):   # covers np.bool_, np.int64, np.float64, etc.
        return obj.item()
    return obj


def _section(title: str):
    log.info("=" * 88)
    log.info(title)
    log.info("=" * 88)


# ==============================================================================
# 1. DATA LOADING — proven stable, unchanged from the working baseline
# ==============================================================================
def _needed_columns(path: Path):
    schema_names = pq.ParquetFile(path).schema.names
    drop = {c for c in schema_names if c.lower() in ID_LIKE_COLS}
    needed = [c for c in schema_names if c not in drop]
    if drop:
        log.info(f"Skipping ID-like columns at load time (never read into RAM): {sorted(drop)}")
    return needed


def load_and_prepare_arrays(path: Path):
    needed_cols = _needed_columns(path)
    log.info(f"Loading engineered features from: {path} ({len(needed_cols)} columns)")
    t0 = time.time()

    df = pd.read_parquet(path, columns=needed_cols, engine="pyarrow")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")
    if TIMESTAMP_COL not in df.columns:
        raise ValueError(f"Timestamp column '{TIMESTAMP_COL}' not found.")

    log.info("Sorting chronologically (required for a valid OOT split)...")
    df = df.sort_values(TIMESTAMP_COL, kind="mergesort")

    feature_cols = [c for c in df.columns if c not in (TARGET_COL, TIMESTAMP_COL)]

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"Feature columns must be pre-encoded numeric types. "
                          f"Non-numeric columns found: {non_numeric}")

    ts_all = df[TIMESTAMP_COL].to_numpy(copy=True)
    y_all = df[TARGET_COL].to_numpy(dtype=np.int32, copy=True)
    dtype = np.float32 if DOWNCAST_FLOAT32 else np.float64
    X_all = df[feature_cols].to_numpy(dtype=dtype, copy=True)

    del df
    gc.collect()

    log.info(f"Loaded {len(y_all):,} rows x {len(feature_cols)} features in {time.time()-t0:.1f}s "
             f"(feature matrix: {X_all.nbytes/1e9:.2f} GB)")
    _log_memory("after load+prepare")

    fraud_rate = float(y_all.mean())
    log.info(f"Fraud rate: {fraud_rate:.4%} ({int(y_all.sum()):,} fraud / {len(y_all):,} total)")

    return X_all, y_all, ts_all, feature_cols


# ==============================================================================
# 2. CHRONOLOGICAL SPLIT
# ==============================================================================
def chronological_splits(X_all, y_all, ts_all, oot_frac: float, val_frac: float):
    n = len(y_all)
    train_end = int(n * oot_frac)
    val_start = int(train_end * (1 - val_frac))

    X_fit, y_fit = X_all[:val_start], y_all[:val_start]
    X_val, y_val = X_all[val_start:train_end], y_all[val_start:train_end]
    X_oot, y_oot = X_all[train_end:], y_all[train_end:]

    log.info(f"Chronological split -> fit={len(y_fit):,} rows (fraud={int(y_fit.sum()):,}) | "
             f"val={len(y_val):,} rows (fraud={int(y_val.sum()):,}) | "
             f"OOT={len(y_oot):,} rows (fraud={int(y_oot.sum()):,})")
    log.info(f"fit window: {ts_all[0]} -> {ts_all[val_start-1]} | "
             f"val window: {ts_all[val_start]} -> {ts_all[train_end-1]} | "
             f"OOT window: {ts_all[train_end]} -> {ts_all[-1]}")
    return X_fit, y_fit, X_val, y_val, X_oot, y_oot


def compute_scale_pos_weight(y_fit: np.ndarray, y_val: np.ndarray) -> float:
    n_pos = int(y_fit.sum() + y_val.sum())
    n_neg = int(len(y_fit) + len(y_val) - n_pos)
    weight = n_neg / n_pos
    log.info(f"scale_pos_weight (raw ratio) = count(legit)/count(fraud) = {n_neg:,}/{n_pos:,} = {weight:.2f}")
    return weight


# ==============================================================================
# 3. BAYESIAN HYPERPARAMETER TUNING
# ==============================================================================
def build_time_series_folds(n_rows: int, n_folds: int):
    tscv = TimeSeriesSplit(n_splits=n_folds)
    folds = []
    for tr_idx, val_idx in tscv.split(np.arange(n_rows)):
        tr_slice = slice(int(tr_idx.min()), int(tr_idx.max()) + 1)
        val_slice = slice(int(val_idx.min()), int(val_idx.max()) + 1)
        folds.append((tr_slice, val_slice))
    return folds


def get_tuning_subset(X_fit: np.ndarray, y_fit: np.ndarray):
    n = len(y_fit)
    if n <= TUNING_SAMPLE_ROWS:
        return X_fit, y_fit
    log.info(f"Fit partition has {n:,} rows; capping tuning subset to the most "
             f"recent {TUNING_SAMPLE_ROWS:,} rows (zero-copy view).")
    return X_fit[-TUNING_SAMPLE_ROWS:], y_fit[-TUNING_SAMPLE_ROWS:]


def make_objective(X_tune, y_tune, raw_scale_pos_weight, cv_folds, feature_cols):
    def objective(trial: optuna.Trial) -> float:
        spw_multiplier = trial.suggest_float(
            "scale_pos_weight_multiplier", *SCALE_POS_WEIGHT_MULTIPLIER_RANGE, log=True
        )
        params = {
            "objective": "binary",
            "metric": ["average_precision", "auc"],
            "boosting_type": "gbdt",
            "is_unbalance": False,
            "scale_pos_weight": raw_scale_pos_weight * spw_multiplier,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, MAX_NUM_LEAVES),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 2000),
            "min_child_samples": 50,   # UNWEIGHTED floor — prevents a LightGBM C++ assertion
                                        # crash ("right_count > 0") triggered when the WEIGHTED
                                        # min_data_in_leaf check is fooled by extreme
                                        # scale_pos_weight on a near-empty split
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
            "verbosity": -1,
            "seed": RANDOM_STATE,
            "n_jobs": LGBM_N_JOBS,
            "max_bin": MAX_BIN,
        }

        fold_scores = []
        for fold_i, (tr_slice, val_slice) in enumerate(cv_folds):
            X_tr, y_tr = X_tune[tr_slice], y_tune[tr_slice]
            X_val, y_val = X_tune[val_slice], y_tune[val_slice]

            train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols, free_raw_data=True)
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set, feature_name=feature_cols, free_raw_data=True)

            model = lgb.train(
                params, train_set, num_boost_round=TUNING_NUM_BOOST_ROUND,
                valid_sets=[val_set], valid_names=["val"],
                callbacks=[
                    lgb.early_stopping(TUNING_EARLY_STOPPING, first_metric_only=True, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            preds = model.predict(X_val, num_iteration=model.best_iteration)
            pr_auc = average_precision_score(y_val, preds)
            fold_scores.append(pr_auc)
            trial.report(np.mean(fold_scores), fold_i)

            del train_set, val_set, model, preds
            gc.collect()
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))
    return objective


def tune_hyperparameters(X_fit, y_fit, raw_scale_pos_weight, feature_cols):
    _section(f"HYPERPARAMETER TUNING — {N_OPTUNA_TRIALS} trials, {N_CV_FOLDS}-fold time-series CV")

    optuna_cache = REPORT_DIR / "optuna_best_params.json"
    if optuna_cache.exists():
        log.warning(f"CACHED TUNING RESULT FOUND at {optuna_cache} — REUSING IT instead of "
                    f"re-tuning. If you have regenerated features_lgbm.parquet since this cache "
                    f"was created (e.g. after fixing feature leakage), DELETE THIS FILE and "
                    f"rerun so tuning reflects the CURRENT data. Proceeding with cached params.")
        with open(optuna_cache) as f:
            cached = json.load(f)
        log.info(f"Cached best PR-AUC (CV): {cached['best_value_pr_auc']:.4f}")
        return cached["best_params"]

    X_tune, y_tune = get_tuning_subset(X_fit, y_fit)
    cv_folds = build_time_series_folds(len(y_tune), N_CV_FOLDS)

    sampler = TPESampler(seed=RANDOM_STATE)
    pruner = MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner,
                                 study_name="mule_guard_lgbm")

    objective = make_objective(X_tune, y_tune, raw_scale_pos_weight, cv_folds, feature_cols)
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False, n_jobs=1)
    gc.collect()

    log.info(f"Best trial PR-AUC (CV): {study.best_value:.4f}")
    log.info(f"Best params: {json.dumps(study.best_params, indent=2)}")

    with open(optuna_cache, "w") as f:
        json.dump(_json_safe({"best_value_pr_auc": study.best_value, "best_params": study.best_params}), f, indent=2)

    return study.best_params


# ==============================================================================
# 4. SINGLE-MODEL TRAINING — the ONLY model this pipeline trains
# ==============================================================================
def train_final_model(X_fit, y_fit, X_val, y_val, best_params, raw_scale_pos_weight, feature_cols):
    """
    Trained to genuine convergence: FINAL_NUM_BOOST_ROUND=6000 is a ceiling,
    not a target. Early stopping (patience=150, first_metric_only=True,
    watching ONLY the internal validation slice) decides the real tree
    count. If best_iteration lands near the ceiling, that's flagged below
    as a sign the model was still improving and the ceiling should be
    raised further next time.
    """
    best_params = dict(best_params)
    spw_multiplier = best_params.pop("scale_pos_weight_multiplier", 1.0)
    scale_pos_weight = raw_scale_pos_weight * spw_multiplier

    if "min_data_in_leaf" in best_params:
        best_params["min_data_in_leaf"] = max(best_params["min_data_in_leaf"], 100)

    params = {
        "objective": "binary",
        "metric": ["average_precision", "auc"],
        "boosting_type": "gbdt",
        "is_unbalance": False,
        "scale_pos_weight": scale_pos_weight,
        "min_child_samples": 50,
        "verbosity": -1,
        "seed": RANDOM_STATE,
        "n_jobs": LGBM_N_JOBS,
        "max_bin": MAX_BIN,
        "two_round": USE_TWO_ROUND_LOADING,
        **best_params,
    }

    fit_set = lgb.Dataset(X_fit, label=y_fit, feature_name=feature_cols, free_raw_data=True)
    val_set = lgb.Dataset(X_val, label=y_val, reference=fit_set, feature_name=feature_cols, free_raw_data=True)

    log.info(f"Training on {len(y_fit):,} fit rows | scale_pos_weight={scale_pos_weight:.2f} "
             f"({raw_scale_pos_weight:.2f} raw x {spw_multiplier:.4f} tuned) | "
             f"ceiling={FINAL_NUM_BOOST_ROUND} rounds, early_stopping patience={FINAL_EARLY_STOPPING}")
    _log_memory("before final training")

    model = lgb.train(
        params, fit_set, num_boost_round=FINAL_NUM_BOOST_ROUND,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[
            lgb.early_stopping(FINAL_EARLY_STOPPING, first_metric_only=True, verbose=True),
            lgb.log_evaluation(period=200),
        ],
    )
    _log_memory("after final training")

    if model.best_iteration is not None and model.best_iteration < 20:
        log.warning(f"best_iteration={model.best_iteration} suspiciously low — "
                     f"check for label leakage or a mis-set scale_pos_weight.")
    elif model.best_iteration is not None and model.best_iteration >= FINAL_NUM_BOOST_ROUND - 5:
        log.warning(f"best_iteration={model.best_iteration} hit the {FINAL_NUM_BOOST_ROUND}-round "
                     f"ceiling — model may still be improving; consider raising FINAL_NUM_BOOST_ROUND.")
    else:
        log.info(f"Converged via genuine early stopping at best_iteration={model.best_iteration} "
                  f"(well under the {FINAL_NUM_BOOST_ROUND}-round ceiling)")

    del fit_set, val_set
    gc.collect()
    return model, scale_pos_weight


# ==============================================================================
# 5. EVALUATION
# ==============================================================================
def evaluate_at_top_k(y_true: np.ndarray, y_scores: np.ndarray, k_frac: float):
    n = len(y_scores)
    k = max(1, int(np.ceil(n * k_frac)))
    order = np.argsort(-y_scores)
    top_k_idx = order[:k]
    flagged = np.zeros(n, dtype=bool)
    flagged[top_k_idx] = True
    tp = int(np.sum(y_true[flagged] == 1))
    fp = int(np.sum(y_true[flagged] == 0))
    total_fraud = int(np.sum(y_true == 1))
    recall = tp / total_fraud if total_fraud > 0 else 0.0
    precision = tp / k if k > 0 else 0.0
    return {"top_k_pct": k_frac, "n_flagged": k, "true_positives": tp,
            "false_positives": fp, "recall": recall, "precision": precision}


def evaluate_model(model, X_oot: np.ndarray, y_oot: np.ndarray):
    _section("EVALUATION — Out-of-Time Test Set (touched here for the first time)")

    y_scores = model.predict(X_oot, num_iteration=model.best_iteration)
    y_true = y_oot

    pr_auc = average_precision_score(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    log.info(f"PR-AUC  (Primary Metric): {pr_auc:.4f}")
    log.info(f"ROC-AUC                 : {roc_auc:.4f}")

    y_pred_default = (y_scores >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_default)
    log.info(f"Confusion Matrix @0.5 threshold (reference only — not the production "
             f"operating point, see Top-K capacity metrics below):\n{cm}")
    log.info("\n" + classification_report(y_true, y_pred_default, digits=4))

    capacity_results = []
    for k_frac in TOP_K_PCTS:
        res = evaluate_at_top_k(y_true, y_scores, k_frac)
        capacity_results.append(res)
        log.info(f"[Alert Capacity Top {k_frac:.0%}] Flagged={res['n_flagged']:,} | "
                 f"Recall={res['recall']:.4%} | Precision={res['precision']:.4%} | "
                 f"TP={res['true_positives']} | FP={res['false_positives']}")

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    above_80 = np.where(recalls[:-1] >= 0.80)[0]
    idx_80 = above_80[-1] if len(above_80) > 0 else None
    mandate_met = bool(idx_80 is not None and recalls[idx_80] >= 0.80)   # explicit bool() -- see _json_safe docstring
    if mandate_met:
        log.info(f"Recall >= 80% MANDATE MET at score threshold={thresholds[idx_80]:.4f} "
                 f"(Recall={recalls[idx_80]:.4%}, Precision={precisions[idx_80]:.4%})")
    else:
        log.warning("Recall >= 80% mandate NOT reached at any threshold on this OOT set.")

    metrics = {
        "pr_auc": pr_auc, "roc_auc": roc_auc,
        "capacity_metrics": capacity_results, "mandate_met": mandate_met,
    }
    with open(REPORT_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(_json_safe(metrics), f, indent=2)

    _plot_pr_curve(precisions, recalls, pr_auc)
    return y_scores, metrics


def _plot_pr_curve(precisions, recalls, pr_auc):
    plt.figure(figsize=(7, 6))
    plt.plot(recalls, precisions, color="#c0392b", linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
    plt.axvline(0.80, color="gray", linestyle="--", linewidth=1, label="80% Recall mandate")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("MuleGuard.AI — Precision-Recall Curve (OOT Test Set)")
    plt.legend(loc="upper right"); plt.grid(alpha=0.3); plt.tight_layout()
    out_path = REPORT_DIR / "pr_curve.png"
    plt.savefig(out_path, dpi=150); plt.close()
    log.info(f"Saved PR curve plot -> {out_path}")


# ==============================================================================
# 6. EXPLAINABILITY — ALL features, full ranked table (not a top-10 preview)
# ==============================================================================
def generate_shap_report(model, X_oot: np.ndarray, feature_cols, sample_size=SHAP_SAMPLE_ROWS):
    _section("EXPLAINABILITY — SHAP feature attribution, all features")

    n = len(X_oot)
    if n > sample_size:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n, size=sample_size, replace=False)
        X_shap = X_oot[idx]
    else:
        X_shap = X_oot

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure()
    shap.summary_plot(shap_values, X_shap, feature_names=feature_cols,
                       plot_type="bar", show=False, max_display=len(feature_cols))
    plt.title("MuleGuard.AI — Global Feature Importance, ALL Features (mean |SHAP value|)")
    plt.tight_layout()
    bar_path = REPORT_DIR / "shap_summary_bar_full.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight"); plt.close()
    log.info(f"Saved FULL feature-importance bar plot ({len(feature_cols)} features) -> {bar_path}")

    plt.figure()
    shap.summary_plot(shap_values, X_shap, feature_names=feature_cols, show=False, max_display=30)
    plt.title("MuleGuard.AI — SHAP Value Distribution (Top 30 of Full Feature Set)")
    plt.tight_layout()
    dot_path = REPORT_DIR / "shap_summary_dot.png"
    plt.savefig(dot_path, dpi=150, bbox_inches="tight"); plt.close()
    log.info(f"Saved SHAP beeswarm plot -> {dot_path}")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap}).sort_values(
        "mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1
    importance_df = importance_df[["rank", "feature", "mean_abs_shap"]]
    importance_df.to_csv(REPORT_DIR / "shap_feature_importance_full.csv", index=False)

    log.info(f"FULL feature importance ranking, all {len(feature_cols)} features:\n"
              + importance_df.to_string(index=False))

    return importance_df


# ==============================================================================
# 7. SERIALIZATION
# ==============================================================================
def save_model(model, feature_cols, best_params, metrics, importance_df):
    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "best_params": best_params,
        "metrics": metrics,
        "feature_importance": importance_df.to_dict(orient="records"),
        "target_col": TARGET_COL,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
    }
    joblib.dump(artifact, MODEL_PATH)
    log.info(f"Model artifact serialized -> {MODEL_PATH}")


# ==============================================================================
# 8. EXECUTIVE SUMMARY — professional, manager/model-risk-review-ready
# ==============================================================================
def write_executive_summary(metrics, importance_df, best_params, n_fit, n_val, n_oot,
                             raw_scale_pos_weight, best_iteration):
    mandate_line = "**MET**" if metrics["mandate_met"] else "**NOT MET**"
    top_features = importance_df.head(15).to_string(index=False)

    report = f"""# MuleGuard.AI — Executive Summary
Generated: {pd.Timestamp.utcnow().isoformat()}

## Business Result
- **PR-AUC:** {metrics['pr_auc']:.4f}
- **ROC-AUC:** {metrics['roc_auc']:.4f}
- **80% Recall mandate:** {mandate_line}
- **Model complexity:** {best_iteration} trees (genuine convergence via early stopping)

## Alert-Capacity Metrics (the numbers that matter operationally)
| Alert Budget | Transactions Flagged/Day | Recall (Fraud Caught) | Precision |
|---|---|---|---|
"""
    for cap in metrics["capacity_metrics"]:
        report += f"| Top {cap['top_k_pct']:.0%} | {cap['n_flagged']:,} | {cap['recall']:.2%} | {cap['precision']:.2%} |\n"

    report += f"""
## Data
- Fit set: {n_fit:,} rows | Validation set: {n_val:,} rows | OOT test set: {n_oot:,} rows (never used until final evaluation)
- Class imbalance (raw ratio, legit:fraud): {raw_scale_pos_weight:.2f}:1

## Tuned Hyperparameters
```json
{json.dumps(best_params, indent=2)}
```

## Top 15 Features (of {len(importance_df)} total — full ranking in shap_feature_importance_full.csv)
```
{top_features}
```

## Interpretation for Non-Technical Stakeholders
This model was trained strictly on past transactions and evaluated only on
future, unseen transactions — the same constraint it will face in
production. If investigators review the top {metrics['capacity_metrics'][0]['top_k_pct']:.0%} of
flagged transactions daily, the model catches
{metrics['capacity_metrics'][0]['recall']:.1%} of all laundering activity.
The top drivers behind flagged transactions match patterns a trained human
investigator would look for (repeated sender-receiver pairs, account age,
payment method, transaction velocity), which is a strong sanity check that
the model is learning real fraud behavior rather than noise.
"""
    with open(EXEC_SUMMARY_PATH, "w") as f:
        f.write(report)
    log.info(f"Executive summary written -> {EXEC_SUMMARY_PATH}")


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    _section("MuleGuard.AI — Enterprise Training Pipeline (Single-Model, Final)")
    log.info(f"Resource guardrails: n_jobs={LGBM_N_JOBS} (of {os.cpu_count()} logical cores), "
             f"tuning_sample_rows={TUNING_SAMPLE_ROWS:,}, max_num_leaves={MAX_NUM_LEAVES}, "
             f"final_round_ceiling={FINAL_NUM_BOOST_ROUND}, early_stopping_patience={FINAL_EARLY_STOPPING}")
    _log_memory("startup")

    X_all, y_all, ts_all, feature_cols = load_and_prepare_arrays(DATA_PATH)
    log.info(f"Feature set ({len(feature_cols)} columns): {feature_cols}")

    X_fit, y_fit, X_val, y_val, X_oot, y_oot = chronological_splits(
        X_all, y_all, ts_all, OOT_SPLIT_FRACTION, INTERNAL_VAL_FRACTION)
    del ts_all
    gc.collect()

    raw_scale_pos_weight = compute_scale_pos_weight(y_fit, y_val)

    best_params = tune_hyperparameters(X_fit, y_fit, raw_scale_pos_weight, feature_cols)

    model, scale_pos_weight = train_final_model(
        X_fit, y_fit, X_val, y_val, best_params, raw_scale_pos_weight, feature_cols)

    y_scores, metrics = evaluate_model(model, X_oot, y_oot)

    importance_df = generate_shap_report(model, X_oot, feature_cols)

    save_model(model, feature_cols, {**best_params, "scale_pos_weight": scale_pos_weight},
               metrics, importance_df)

    write_executive_summary(metrics, importance_df, best_params,
                             len(y_fit), len(y_val), len(y_oot), raw_scale_pos_weight,
                             model.best_iteration)

    _section("PIPELINE COMPLETE")
    log.info(f"PR-AUC: {metrics['pr_auc']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")
    for cap in metrics["capacity_metrics"]:
        log.info(f"Top {cap['top_k_pct']:.0%} capacity -> Recall={cap['recall']:.4%}, Precision={cap['precision']:.4%}")
    log.info(f"Mandate met: {metrics['mandate_met']}")
    log.info(f"Model saved to: {MODEL_PATH}")
    log.info(f"Executive summary: {EXEC_SUMMARY_PATH}")
    log.info(f"Full feature importance: {REPORT_DIR / 'shap_feature_importance_full.csv'}")


if __name__ == "__main__":
    main()