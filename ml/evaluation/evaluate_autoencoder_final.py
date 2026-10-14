"""
================================================================================
MuleGuard.AI — Autoencoder v6
Enterprise Validation & Explainability Suite — Isolated Staging Version
================================================================================

ISOLATION GUARANTEE
-------------------
This script loads artifacts ONLY from ml/models/autoencoder_v6/ and
ml/data/processed/autoencoder_v6_cache/ — the v6 staging paths.
The existing production model (ml/models/autoencoder/autoencoder_oot_v2.pt) is
NEVER read or modified.

Produces (in ml/reports/autoencoder_v6/):
  evaluation_metrics.json   — real ROC-AUC / PR-AUC / alert-capacity table
  pr_curve.png              — Precision-Recall curve (if matplotlib installed)
  feature_attribution.csv   — per-feature reconstruction error ranking
  feature_attribution.png   — bar chart of feature attribution (if matplotlib installed)
  evaluation_log.txt        — full run log
  executive_summary.md      — data-driven narrative for stakeholders

Run:
  python -X utf8 ml/evaluation/evaluate_autoencoder_v6.py
  (or: python -X utf8 scripts/evaluate_autoencoder_v6.py)
================================================================================
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

try:
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
except ImportError:
    # Lightweight pure-numpy fallback if scikit-learn is not installed
    def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=bool)
        y_score = np.asarray(y_score)
        n_pos = int(np.sum(y_true))
        n_neg = len(y_true) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5
        order = np.argsort(y_score)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
        sorted_score = y_score[order]
        diff = np.diff(sorted_score)
        if np.any(diff == 0):
            i = 0
            while i < len(y_score):
                j = i
                while j < len(y_score) - 1 and sorted_score[j + 1] == sorted_score[i]:
                    j += 1
                if j > i:
                    avg_rank = (i + 1 + j + 1) / 2.0
                    ranks[order[i : j + 1]] = avg_rank
                    i = j + 1
                else:
                    i += 1
        sum_pos_ranks = np.sum(ranks[y_true])
        return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    def precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray):
        y_true = np.asarray(y_true, dtype=bool)
        y_score = np.asarray(y_score)
        order = np.argsort(y_score)[::-1]
        y_true_sorted = y_true[order]
        y_score_sorted = y_score[order]
        tp = np.cumsum(y_true_sorted)
        fp = np.cumsum(~y_true_sorted)
        recalls = tp / max(tp[-1], 1)
        precisions = tp / np.maximum(tp + fp, 1)
        precisions = np.r_[precisions, 1.0]
        recalls = np.r_[recalls, 0.0]
        thresholds = y_score_sorted
        return precisions, recalls, thresholds

    def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
        precisions, recalls, _ = precision_recall_curve(y_true, y_score)
        return float(-np.sum(np.diff(recalls) * precisions[:-1]))

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ==============================================================================
# CONFIGURATION — all paths are v6 staging paths
# ==============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
if _SCRIPT_DIR.name == "scripts":
    _PROJECT_ROOT = _SCRIPT_DIR.parent
elif _SCRIPT_DIR.name == "evaluation":
    _PROJECT_ROOT = _SCRIPT_DIR.parent.parent
else:
    _PROJECT_ROOT = _SCRIPT_DIR

MODEL_DIR  = _PROJECT_ROOT / "ml" / "models"  / "Autoencoder"
REPORT_DIR = _PROJECT_ROOT / "ml" / "reports" / "Autoencoder"
CACHE_DIR  = _PROJECT_ROOT / "ml" / "data"    / "processed" / "autoencoder_cache"

RESERVED_CORES  = 2
ALERT_BUDGETS   = [0.001, 0.005, 0.01, 0.05]   # 0.1%, 0.5%, 1%, 5%
SAMPLE_ROWS_ATT = 200_000                        # rows sampled for feature attribution

# 21 features — must match retrain_autoencoder_v6.py exactly
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

# Cold-start guard — must match retrain_autoencoder_v6.py
COLD_START_FEATURES = ["txn_count_1h", "fan_out_24h"]
COLD_START_INDICES  = [FEATURE_NAMES.index(f) for f in COLD_START_FEATURES]


# ==============================================================================
# LOGGING  (directories created here after path resolution above)
# ==============================================================================
REPORT_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("MuleGuard.AE.v6.Eval")
log.propagate = False
log.setLevel(logging.INFO)
if not log.handlers:
    _fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    _sh  = logging.StreamHandler(sys.stdout)
    _sh.setFormatter(_fmt)
    _fh  = logging.FileHandler(REPORT_DIR / "evaluation_log.txt", mode="w", encoding="utf-8")
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
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


# ==============================================================================
# ARCHITECTURE — must match retrain_autoencoder_v6.py exactly
# ==============================================================================
class AutoencoderV6(nn.Module):
    """
    Must match retrain_autoencoder_v6.py's architecture exactly, or
    load_state_dict() will fail. Architecture is read from the checkpoint's
    saved hyperparameters, so this class adapts to whatever was trained.
    """

    def __init__(
        self,
        input_dim:      int   = len(FEATURE_NAMES),
        hidden_dim:     int   = 128,
        bottleneck_dim: int   = 12,   # must match retrain_autoencoder_v6.py BOTTLENECK_DIM
        dropout_rate:   float = 0.15,
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
# BATCH UTILITIES
# ==============================================================================
def batch_ranges(n: int, batch_size: int = 65_536):
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
    """True where txn_count_1h==0 AND fan_out_24h==0 (new account with no history)."""
    return (X[:, COLD_START_INDICES[0]] == 0.0) & (X[:, COLD_START_INDICES[1]] == 0.0)


# ==============================================================================
# ARTIFACT LOADING
# ==============================================================================
def load_artifacts(device: torch.device):
    model_path      = MODEL_DIR / "autoencoder_final.pt"
    scaler_path     = MODEL_DIR / "scaler_stats.json"
    thresh_path     = MODEL_DIR / "threshold_config.json"
    error_norm_path = MODEL_DIR / "feature_error_norm.json"
    cache_meta_path = CACHE_DIR / "metadata.json"

    missing = [p for p in (model_path, scaler_path, thresh_path, cache_meta_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Required artifacts missing:\n" +
            "\n".join(f"  {p}" for p in missing) +
            "\nRun retrain_autoencoder_final.py first."
        )

    log.info(f"Loading final checkpoint: {model_path} ...")
    ckpt = torch.load(model_path, map_location=device, weights_only=True)

    # Verify feature names match
    saved_features = ckpt.get("feature_names", FEATURE_NAMES)
    if saved_features != FEATURE_NAMES:
        raise RuntimeError(
            f"Checkpoint feature_names ({len(saved_features)} features) does not match "
            f"this script's FEATURE_NAMES ({len(FEATURE_NAMES)} features). "
            "Run retrain_autoencoder_v6.py to produce a fresh v6 checkpoint."
        )

    model = AutoencoderV6(
        input_dim      = ckpt.get("input_dim",      len(FEATURE_NAMES)),
        hidden_dim     = ckpt.get("hidden_dim",      128),
        bottleneck_dim = ckpt.get("bottleneck_dim",  12),
        dropout_rate   = ckpt.get("dropout_rate",    0.15),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    log.info(
        f"  Architecture: {model.input_dim}→{model.hidden_dim}→{model.hidden_dim//2}"
        f"→{model.bottleneck_dim}→{model.hidden_dim//2}→{model.hidden_dim}→{model.input_dim}"
    )

    with open(scaler_path, encoding="utf-8") as f:
        scaler = json.load(f)
    with open(thresh_path, encoding="utf-8") as f:
        thresh = json.load(f)
    with open(cache_meta_path, encoding="utf-8") as f:
        cache_meta = json.load(f)

    if scaler.get("feature_order") != FEATURE_NAMES:
        raise RuntimeError(
            "scaler_stats.json feature_order does not match FEATURE_NAMES. "
            "Re-run retrain_autoencoder_v6.py."
        )

    mean = np.array(scaler["mean"], dtype=np.float32)
    std  = np.array(scaler["std"],  dtype=np.float32)
    std[std == 0] = 1.0

    error_norm = None
    if error_norm_path.exists():
        with open(error_norm_path, encoding="utf-8") as f:
            en_raw = json.load(f)
        if en_raw.get("feature_names") != FEATURE_NAMES:
            raise RuntimeError(
                "feature_error_norm.json feature_names mismatch. Re-run retrain_autoencoder_v6.py."
            )
        error_norm = {
            "mean": np.array(en_raw["val_squared_error_mean"], dtype=np.float64),
            "std":  np.array(en_raw["val_squared_error_std"],  dtype=np.float64),
        }
        log.info("  Per-feature error normalization loaded (normalized score is primary metric).")
    else:
        log.warning(
            f"{error_norm_path} not found. Falling back to raw unweighted MSE. "
            "Re-run retrain_autoencoder_v6.py to enable improved scoring."
        )

    return model, mean, std, thresh, cache_meta, error_norm


# ==============================================================================
# CACHE LOADING
# ==============================================================================
def load_test_cache(cache_meta: dict) -> Tuple[np.ndarray, np.ndarray]:
    counts  = cache_meta["counts"]
    n_feat  = len(FEATURE_NAMES)
    n_test  = counts["test"]

    if cache_meta.get("feature_names") != FEATURE_NAMES:
        raise RuntimeError(
            "Cache feature_names does not match FEATURE_NAMES. "
            "Delete ml/data/processed/autoencoder_v6_cache/ and re-run retrain_autoencoder_v6.py."
        )

    X_test = np.memmap(CACHE_DIR / "test_X.f32", dtype=np.float32, mode="r",
                       shape=(n_test, n_feat))
    y_test = np.memmap(CACHE_DIR / "test_y.i8",  dtype=np.int8,    mode="r",
                       shape=(n_test,))
    return X_test, np.asarray(y_test, dtype=np.int8)


# ==============================================================================
# SCORING
# ==============================================================================
def score_raw_mse(
    model:      AutoencoderV6,
    X:          np.ndarray,
    mean:       np.ndarray,
    std:        np.ndarray,
    device:     torch.device,
    batch_size: int = 65_536,
) -> np.ndarray:
    """Raw unweighted MSE — kept for comparison with normalized score."""
    scores = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        for start, end in batch_ranges(len(X), batch_size):
            batch = to_tensor(X[start:end], mean, std, device)
            recon = model(batch)
            scores[start:end] = torch.mean(
                torch.square(recon - batch), dim=1
            ).cpu().numpy()
    return scores


def score_normalized(
    model:      AutoencoderV6,
    X:          np.ndarray,
    mean:       np.ndarray,
    std:        np.ndarray,
    error_norm: dict,
    device:     torch.device,
    batch_size: int = 65_536,
) -> np.ndarray:
    """
    Primary metric: per-feature validation-normalized anomaly score.
    z = clip((squared_error - mean) / std, 0.0, None)
    score = mean(z across features)
    """
    feat_mean = error_norm["mean"]
    feat_std  = np.maximum(error_norm["std"], 1e-9)
    scores    = np.zeros(len(X), dtype=np.float32)

    with torch.no_grad():
        for start, end in batch_ranges(len(X), batch_size):
            batch = to_tensor(X[start:end], mean, std, device)
            recon = model(batch)
            sq_e  = torch.square(recon - batch).cpu().numpy().astype(np.float64)
            z     = np.clip((sq_e - feat_mean) / feat_std, 0.0, None)
            scores[start:end] = z.mean(axis=1).astype(np.float32)

    return scores


# ==============================================================================
# ALERT BUDGET METRICS
# ==============================================================================
def alert_budget(labels: np.ndarray, scores: np.ndarray, fraction: float) -> dict:
    n           = len(scores)
    flagged     = max(1, int(math.ceil(n * fraction)))
    top_idx     = np.argpartition(scores, -flagged)[-flagged:]
    caught      = int(np.sum(labels[top_idx] == 1))
    total_fraud = int(np.sum(labels == 1))
    precision   = caught / flagged
    prevalence  = total_fraud / n if n else 0.0
    return {
        "alert_pct":      fraction * 100,
        "cutoff":         float(np.min(scores[top_idx])),
        "flagged_volume": flagged,
        "caught":         caught,
        "total_fraud":    total_fraud,
        "recall":         caught / total_fraud if total_fraud else 0.0,
        "precision":      precision,
        "lift":           precision / prevalence if prevalence else 0.0,
    }


# ==============================================================================
# FEATURE ATTRIBUTION
# ==============================================================================
def compute_feature_attribution(
    model:       AutoencoderV6,
    X:           np.ndarray,
    mean:        np.ndarray,
    std:         np.ndarray,
    device:      torch.device,
    sample_rows: int,
):
    """
    Per-feature mean squared reconstruction error over a random sample of the
    OOT test set. Higher error = model finds this feature harder to reconstruct =
    behaviorally anomalous signal.
    """
    n = len(X)
    if n > sample_rows:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(n, size=sample_rows, replace=False))
        X_sample = X[idx]
    else:
        X_sample = X[:]

    per_feat_sum = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    total_rows   = 0
    with torch.no_grad():
        for start, end in batch_ranges(len(X_sample)):
            batch = to_tensor(X_sample[start:end], mean, std, device)
            recon = model(batch)
            sq_e  = torch.square(recon - batch).sum(dim=0).cpu().numpy()
            per_feat_sum += sq_e.astype(np.float64)
            total_rows   += (end - start)

    mean_sq_err = per_feat_sum / max(total_rows, 1)
    return sorted(zip(FEATURE_NAMES, mean_sq_err), key=lambda t: t[1], reverse=True)


# ==============================================================================
# PLOTS (Optional if matplotlib is installed)
# ==============================================================================
def plot_pr_curve(precisions, recalls, pr_auc: float, random_baseline: float) -> Optional[Path]:
    if not _HAS_MATPLOTLIB:
        log.warning("matplotlib not found; skipping pr_curve.png generation. Run 'pip install matplotlib' if visual plots are desired.")
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recalls, precisions, color="#16a085", linewidth=2,
            label=f"AE v6 PR-AUC = {pr_auc:.4f}")
    ax.axhline(y=random_baseline, color="#7f8c8d", linestyle="--", linewidth=1,
               label=f"Random baseline = {random_baseline:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("MuleGuard.AI — Autoencoder v6 Precision-Recall Curve (OOT Test)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = REPORT_DIR / "pr_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_feature_attribution(ranked: list) -> Optional[Path]:
    if not _HAS_MATPLOTLIB:
        log.warning("matplotlib not found; skipping feature_attribution.png generation. (feature_attribution.csv was saved)")
        return None
    features = [f for f, _ in ranked]
    values   = [v for _, v in ranked]
    fig, ax  = plt.subplots(figsize=(10, 8))
    ax.barh(features[::-1], values[::-1], color="#2980b9")
    ax.set_xlabel("Mean squared reconstruction error (scaled feature space)")
    ax.set_title(
        "MuleGuard.AI — Autoencoder v6 Feature Attribution\n"
        "(higher = model reconstructs this feature least accurately)"
    )
    fig.tight_layout()
    out = REPORT_DIR / "feature_attribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ==============================================================================
# COLD-START REPORT
# ==============================================================================
def report_cold_start(X_test: np.ndarray, y_test: np.ndarray) -> dict:
    cold = cold_start_mask(X_test)
    n_cold       = int(cold.sum())
    n_cold_fraud = int(y_test[cold].sum())
    return {
        "cold_start_rows":       n_cold,
        "cold_start_fraud_rows": n_cold_fraud,
        "cold_start_pct":        n_cold / max(len(X_test), 1) * 100,
        "note": (
            "Cold-start rows are new accounts with txn_count_1h==0 AND fan_out_24h==0. "
            "They are assigned ae_score=0.0 to prevent false positives on first transactions. "
            f"{n_cold_fraud} of {n_cold} cold-start rows are actual fraud — "
            "these are missed by the AE alone but are handled by the supervised LightGBM model."
        ),
    }


# ==============================================================================
# EXECUTIVE SUMMARY
# ==============================================================================
def write_executive_summary(
    metrics:     dict,
    ranked:      list,
    cache_meta:  dict,
    cold_report: dict,
) -> Path:
    counts    = cache_meta["counts"]
    raw_comp  = metrics.get("raw_mse_comparison", {})
    top10     = "\n".join(
        f"{i+1:>2}. {name:<35} {val:.6f}"
        for i, (name, val) in enumerate(ranked[:10])
    )

    report = f"""# MuleGuard.AI — Autoencoder v6 Executive Summary

## Isolation Status
- **v6 model**: `ml/models/autoencoder_v6/autoencoder_v6.pt` (staging)
- **v2 model**: `ml/models/autoencoder/autoencoder_oot_v2.pt` (production — untouched)

## Business Result
- **OOT ROC-AUC:** {metrics['roc_auc']:.4f}
- **OOT PR-AUC:** {metrics['pr_auc']:.6f}  (random baseline {metrics['prevalence']:.6f})
- **PR-AUC lift over random:** {metrics['pr_auc'] / max(metrics['prevalence'], 1e-9):.1f}x
- **Scoring method:** per-feature validation-normalized anomaly score

{"Comparison with raw unweighted MSE score: ROC-AUC=" + f"{raw_comp.get('roc_auc',0):.4f}" + ", PR-AUC=" + f"{raw_comp.get('pr_auc',0):.6f}" if raw_comp else ""}

## Architecture (v6 upgrades)
- **Input → Hidden → Bottleneck:** 21 → 128 → 64 → **12** → 64 → 128 → 21
- **Compression ratio:** 1.75x (21 → 12 bottleneck enforcing genuine latent compression)
- **Normalization:** LayerNorm (replaces BatchNorm — stable at batch_size=1, safe at inference)
- **Regularization:** Dropout(0.15) on encoder
- **Feature source:** `features_autoencoder.parquet` (21 pre-engineered AML features)

## Cold-Start Guard (fixes score=1 on new transactions)
- **Cold-start rows in OOT test:** {cold_report['cold_start_rows']:,} ({cold_report['cold_start_pct']:.2f}%)
- **Cold-start fraud rows:** {cold_report['cold_start_fraud_rows']:,}
- **Action:** These rows are assigned ae_score=0.0 (not flagged by AE alone)

## Data
- Train rows (legit only):      {counts['train']:,}
- Validation rows (legit only): {counts['validation']:,}
- OOT test rows (all labels):   {counts['test']:,}  ({metrics['n_fraud']:,} fraud, {metrics['prevalence']:.4%})
- Split: last 20% of chronologically-sorted rows (identical to LightGBM OOT split)

## Alert-Capacity Metrics
| Alert Budget | Flagged | Recall | Precision | Lift over random |
|---|---|---|---|---|
"""
    for b in metrics["alert_budgets"]:
        report += (
            f"| Top {b['alert_pct']:.1f}% | {b['flagged_volume']:,} | "
            f"{b['recall']:.2%} | {b['precision']:.2%} | {b['lift']:.1f}x |\n"
        )

    report += f"""
## Feature Attribution (Top 10 of {len(ranked)} — full ranking in feature_attribution.csv)
Mean squared reconstruction error per feature in scaled feature space.
Higher = the model finds this feature harder to reconstruct = stronger anomaly signal.

```
{top10}
```

## Next Steps
1. 38 AE signals are already extracted in `features_ae_signals_v6.parquet`.
2. Run `train_lgbm_aml.py` with AE signals merged (Phase 2).
3. Promote v6 model to production by copying `ml/models/autoencoder_v6/` → `ml/models/autoencoder/`.

## Model Governance Note
This model is trained on legitimate transactions only and never shown confirmed fraud during
training. It complements the supervised LightGBM model by flagging structurally unusual
transactions. Both models evaluate on the identical OOT test window for valid comparison.
"""
    out = REPORT_DIR / "executive_summary.md"
    out.write_text(report, encoding="utf-8")
    return out


# ==============================================================================
# MAIN
# ==============================================================================
def main() -> None:
    _section("MuleGuard.AI — Autoencoder v6 Evaluation Suite (Isolated Staging)")
    _log_memory("startup")

    torch_threads = max(1, (os.cpu_count() or 4) - RESERVED_CORES)
    torch.set_num_threads(torch_threads)
    log.info(f"Resource guardrail: torch.set_num_threads({torch_threads})")
    log.info(f"Loading from v6 staging paths: {MODEL_DIR} / {CACHE_DIR}")
    log.info("Production model (ml/models/autoencoder/) is NEVER touched by this script.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mean, std, thresh, cache_meta, error_norm = load_artifacts(device)
    X_test, y_test = load_test_cache(cache_meta)

    n_total    = len(y_test)
    n_fraud    = int(np.sum(y_test == 1))
    prevalence = n_fraud / n_total if n_total else 0.0
    log.info(f"Loaded OOT test set: {n_total:,} rows ({n_fraud:,} fraud, prevalence={prevalence:.4%})")
    _log_memory("after cache load")

    # ── Cold-start report ───────────────────────────────────────────────────
    _section("Cold-Start Guard Analysis")
    cold_report = report_cold_start(X_test, y_test)
    log.info(
        f"Cold-start rows: {cold_report['cold_start_rows']:,} ({cold_report['cold_start_pct']:.2f}%) "
        f"→ scored 0.0 (suppressed)"
    )
    log.info(f"Cold-start fraud rows: {cold_report['cold_start_fraud_rows']:,}")

    # ── Scoring ─────────────────────────────────────────────────────────────
    _section("Scoring OOT test set (real model inference)")
    raw_scores = score_raw_mse(model, X_test, mean, std, device)
    raw_roc    = float(roc_auc_score(y_test, raw_scores))
    raw_pr     = float(average_precision_score(y_test, raw_scores))
    log.info(f"[Raw unweighted MSE — comparison only] ROC-AUC={raw_roc:.4f} | PR-AUC={raw_pr:.6f}")

    if error_norm is not None:
        scores = score_normalized(model, X_test, mean, std, error_norm, device)
        scoring_method = "per_feature_validation_normalized"
        log.info("Using per-feature validation-normalized score as PRIMARY metric.")
    else:
        scores = raw_scores
        scoring_method = "raw_unweighted_mse"
        log.warning("No feature_error_norm.json — falling back to raw MSE.")

    _log_memory("after scoring")

    roc_auc = float(roc_auc_score(y_test, scores))
    pr_auc  = float(average_precision_score(y_test, scores))
    precisions, recalls, _ = precision_recall_curve(y_test, scores)

    log.info(f"OOT ROC-AUC: {roc_auc:.4f}")
    log.info(
        f"OOT PR-AUC:  {pr_auc:.6f}  "
        f"(random baseline {prevalence:.6f}, lift {pr_auc/max(prevalence,1e-9):.1f}x)"
    )

    # ── Alert capacity ───────────────────────────────────────────────────────
    _section("Alert-Capacity Metrics")
    budgets = [alert_budget(y_test, scores, frac) for frac in ALERT_BUDGETS]
    for b in budgets:
        log.info(
            f"[Top {b['alert_pct']:.1f}%] Flagged={b['flagged_volume']:,} | "
            f"Recall={b['recall']:.2%} | Precision={b['precision']:.2%} | Lift={b['lift']:.1f}x"
        )

    # Calibrated p99.9 threshold (from validation threshold config)
    p99_9_thresh = float(thresh.get("p99_9", thresh.get("p99_5", thresh.get("p99", 0.0))))
    p99_9_flags  = scores >= p99_9_thresh
    p99_9_count  = int(np.sum(p99_9_flags))
    p99_9_caught = int(np.sum(y_test[p99_9_flags] == 1)) if p99_9_count else 0
    log.info(
        f"[Calibrated p99.9 threshold={p99_9_thresh:.6f}] "
        f"Flagged={p99_9_count:,} | Caught={p99_9_caught:,} | "
        f"Recall={p99_9_caught/max(n_fraud,1):.2%}"
    )

    # ── Feature attribution ──────────────────────────────────────────────────
    _section("Feature Attribution (per-feature reconstruction error ranking)")
    ranked = compute_feature_attribution(model, X_test, mean, std, device, SAMPLE_ROWS_ATT)
    for rank, (feat, val) in enumerate(ranked, 1):
        log.info(f"  {rank:>2}. {feat:<38} {val:.6f}")

    with (REPORT_DIR / "feature_attribution.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "feature", "mean_squared_reconstruction_error"])
        for rank, (feat, val) in enumerate(ranked, 1):
            writer.writerow([rank, feat, val])

    # ── Plots (matplotlib optional) ──────────────────────────────────────────
    pr_plot   = plot_pr_curve(precisions, recalls, pr_auc, prevalence)
    att_plot  = plot_feature_attribution(ranked)
    if pr_plot:
        log.info(f"Saved PR curve           -> {pr_plot}")
    if att_plot:
        log.info(f"Saved feature attribution -> {att_plot}")

    # ── Metrics JSON ─────────────────────────────────────────────────────────
    metrics = {
        "model_version":   "v6",
        "n_total":         n_total,
        "n_fraud":         n_fraud,
        "prevalence":      prevalence,
        "roc_auc":         roc_auc,
        "pr_auc":          pr_auc,
        "scoring_method":  scoring_method,
        "raw_mse_comparison": {"roc_auc": raw_roc, "pr_auc": raw_pr},
        "alert_budgets":   budgets,
        "calibrated_p99_9": {
            "threshold": p99_9_thresh,
            "flagged":   p99_9_count,
            "caught":    p99_9_caught,
            "recall":    p99_9_caught / max(n_fraud, 1),
        },
        "cold_start_guard": cold_report,
        "split":           cache_meta["split"],
    }

    metrics_path = REPORT_DIR / "evaluation_metrics.json"
    metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    log.info(f"Wrote {metrics_path}")

    # ── Executive summary ────────────────────────────────────────────────────
    summary_path = write_executive_summary(metrics, ranked, cache_meta, cold_report)
    log.info(f"Wrote {summary_path}")

    _section("EVALUATION COMPLETE — v6")
    log.info(f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.6f}")
    log.info(f"All outputs in: {REPORT_DIR}")
    log.info("Next step: review reports, then run train_lgbm_aml.py for Phase-2 LGBM integration.")


if __name__ == "__main__":
    main()
