"""
ml/training/tgn_evaluate.py
===========================
MuleGuard.AI — Sprint 5/9 — TGN Enterprise Validation Suite

Produces the SAME class of deliverables that already exist for LightGBM
(train_lgbm_aml.py) and the Autoencoder (evaluate_autoencoder_v2.py), so all
three models in this system are governed to the same standard:

    ml/reports/tgn/evaluation_metrics.json   real ROC-AUC/PR-AUC/alert-capacity
    ml/reports/tgn/pr_curve.png              real PR curve, OOT test set
    ml/reports/tgn/threshold_config.json     p95/p97/p99/p99_5 calibrated on
                                              VALIDATION scores only, never test
    ml/reports/tgn/edge_feature_ablation.json  contribution of the 13 Step-5.0
                                              edge features (if an ablation
                                              checkpoint is supplied)
    ml/reports/tgn/executive_summary.md      plain-language report, every
                                              number traceable to a source file
    ml/reports/tgn/evaluation_log.txt        dual console+file log

WHY THIS IS A SEPARATE SCRIPT FROM tgn_train.py
--------------------------------------------------------------------------
tgn_train.py's own end-of-run block computes test_auc/test_pr_auc from a
RESERVOIR SAMPLE (capped at 500,000 events, chosen for memory safety during
a live training run where the process is also holding gradients, optimizer
state, and the full train/val arrays). That's the right tradeoff mid-training,
but it means the only historical TGN numbers on disk came from a training
run's own console log, not from a governed, independently re-runnable
evaluation artifact -- unlike LightGBM and the Autoencoder, which both have
a dedicated evaluate script producing versioned report files. This script
closes that gap: it loads an ALREADY-TRAINED checkpoint fresh, rebuilds the
identical split and features tgn_train.py used, and scores the OOT test set
in FULL (not sampled -- the post-row-fraction-fix test set is ~6.4M rows,
small enough to hold complete score/label arrays in memory, ~50MB total).

CRITICAL: uses the exact same load_and_join / assign_temporal_split /
prepare_arrays functions as tgn_train.py (imported, not reimplemented) --
this guarantees the test population evaluated here is BYTE-IDENTICAL to
what the checkpoint was actually tested against during training, with zero
risk of the two scripts' split logic silently drifting apart over time.

Execution:
    python ml/training/tgn_evaluate.py \\
        --model-path ml/models/tgn_v2_best.pt \\
        --transactions-path ml/data/processed/transactions_clean.parquet \\
        --edge-features-path ml/features/tgn_edge_features.parquet

    # Include the edge-feature ablation comparison (requires having already
    # trained a second checkpoint with --ablate-edge-features):
    python ml/training/tgn_evaluate.py ... --ablation-model-path ml/models/tgn_v2_ablation.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from tgn_model_v2 import TemporalGraphNetwork

# Reuse tgn_train.py's own data-loading and split logic verbatim -- see the
# module docstring above for why this matters. Never reimplement these.
from tgn_train_v3 import (
    EDGE_FEATURE_NAMES, SplitConfig, TrainConfig,
    load_and_join, assign_temporal_split, build_node_index,
    prepare_arrays, iter_event_batches,
)


# ==============================================================================
# CONFIGURATION -- mirrors train_lgbm_aml.py / evaluate_autoencoder_v2.py
# ==============================================================================
log = logging.getLogger("MuleGuard.TGN.Eval")
log.propagate = False
log.setLevel(logging.INFO)
if not log.handlers:
    _formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    _stream_handler = logging.StreamHandler(sys.stdout)
    _stream_handler.setFormatter(_formatter)
    log.addHandler(_stream_handler)


def _log_memory(tag: str) -> None:
    if _HAS_PSUTIL:
        vm = psutil.virtual_memory()
        log.info(f"[mem check: {tag}] RAM used={vm.percent:.1f}% ({vm.used/1e9:.1f}GB / {vm.total/1e9:.1f}GB)")


def _json_safe(value):
    """Same helper, same reason, as every other pipeline in this project --
    see train_lgbm_aml.py's _json_safe() docstring for the numpy-bool/json
    crash this prevents."""
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transactions-path", type=Path, default=Path("ml/data/processed/transactions_clean.parquet"))
    p.add_argument("--edge-features-path", type=Path, default=Path("ml/features/tgn_edge_features.parquet"))
    p.add_argument("--model-path", type=Path, required=True,
                   help="Trained TGN checkpoint to evaluate (e.g. ml/models/tgn_v2_best.pt).")
    p.add_argument("--ablation-model-path", type=Path, default=None,
                   help="Optional: a checkpoint trained with --ablate-edge-features, "
                        "to report the real contribution of the 13 Step-5.0 edge features.")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--report-dir", type=Path, default=Path("ml/reports/tgn"),
                   help="Directory to save evaluation reports.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ==============================================================================
# Full-array scoring (not the training script's memory-safe reservoir sample --
# see module docstring for why the full array is affordable here)
# ==============================================================================
def score_split_full(model: TemporalGraphNetwork, arrays: tuple, batch_size: int, device: torch.device,
                      record_scores: bool) -> dict:
    """Replays one split through the model in strict chronological order,
    advancing memory exactly as run_split() does in tgn_train.py. If
    record_scores is True, returns the FULL per-event score and label
    arrays (used for the val and test splits); if False, only advances
    memory and discards scores (used for the train split, where holding
    ~23M scores would cost real memory for no purpose -- we only need
    memory to end up in the correct chronological state before val/test).
    """
    model.eval()
    n = len(arrays[0])
    scores = np.empty(n, dtype=np.float32) if record_scores else None
    labels_out = np.empty(n, dtype=np.float32) if record_scores else None
    pos = 0
    _n = model.memory.num_nodes - 1

    with torch.no_grad():
        for src, dst, x_cont, fmt_id, evt_time, labels in iter_event_batches(arrays, batch_size):
            src = src.to(device).clamp_(0, _n)
            dst = dst.to(device).clamp_(0, _n)
            x_cont, fmt_id = x_cont.to(device), fmt_id.to(device)
            evt_time = evt_time.to(device)
            risk_logits = model.step(src, dst, x_cont, fmt_id, evt_time)
            if record_scores:
                b = len(risk_logits)
                scores[pos:pos + b] = risk_logits.detach().cpu().numpy()
                labels_out[pos:pos + b] = labels.numpy()
                pos += b

    if not record_scores:
        return {}

    scores = np.nan_to_num(scores, nan=0.0, posinf=20.0, neginf=-20.0)
    with np.errstate(over="ignore"):
        probs = 1.0 / (1.0 + np.exp(-scores.astype(np.float64)))
    return {"probs": probs, "labels": labels_out}


def alert_budget(labels: np.ndarray, scores: np.ndarray, fraction: float) -> dict:
    """Identical methodology to train_lgbm_aml.py's evaluate_at_top_k() --
    same metric, same budgets, so the two models' alert-capacity tables are
    directly comparable line for line."""
    n = len(scores)
    flagged = max(1, int(math.ceil(n * fraction)))
    top_idx = np.argpartition(scores, -flagged)[-flagged:]
    caught = int(np.sum(labels[top_idx] == 1))
    total_fraud = int(np.sum(labels == 1))
    precision = caught / flagged
    prevalence = total_fraud / n if n else 0.0
    return {
        "alert_pct": fraction * 100,
        "flagged_volume": flagged,
        "caught": caught,
        "total_fraud": total_fraud,
        "recall": caught / total_fraud if total_fraud else 0.0,
        "precision": precision,
        "lift": precision / prevalence if prevalence else 0.0,
    }


def plot_pr_curve(precisions, recalls, pr_auc: float, report_dir: Path) -> Path:
    plt.figure(figsize=(7, 6))
    plt.plot(recalls, precisions, color="#8e44ad", linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("MuleGuard.AI — TGN Precision-Recall Curve (OOT Test Set)")
    plt.legend(loc="upper right"); plt.grid(alpha=0.3); plt.tight_layout()
    out_path = report_dir / "pr_curve.png"
    plt.savefig(out_path, dpi=150); plt.close()
    return out_path


def plot_f1_curve(precisions, recalls, thresholds, report_dir: Path) -> Path:
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
    plt.figure(figsize=(7, 6))
    plt.plot(thresholds, f1_scores, color="#27ae60", linewidth=2)
    plt.xlabel("Threshold"); plt.ylabel("F1 Score")
    plt.title("MuleGuard.AI — TGN F1 Score Curve (OOT Test Set)")
    plt.grid(alpha=0.3); plt.tight_layout()
    out_path = report_dir / "f1_curve.png"
    plt.savefig(out_path, dpi=150); plt.close()
    return out_path


def load_checkpoint(model_path: Path, device: torch.device) -> tuple[TemporalGraphNetwork, dict]:
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    log.info(f"Loading TGN checkpoint from {model_path} ...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if "node_index" not in checkpoint:
        raise ValueError(
            f"{model_path} has no 'node_index' -- this looks like a per-epoch "
            "checkpoint, not the final saved model. Re-run training to completion "
            "(the final save includes node_index; per-epoch saves omit it to stay small)."
        )
    model = TemporalGraphNetwork(
        num_nodes=checkpoint["num_nodes"], memory_dim=checkpoint["memory_dim"],
        edge_dim=checkpoint["edge_dim"], time_dim=checkpoint["time_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log.info(f"Checkpoint metadata: val_auc_best={checkpoint.get('val_auc_best', 'n/a')} | "
              f"test_auc (from training run)={checkpoint.get('test_auc', 'n/a')} | "
              f"epochs_trained={checkpoint.get('epochs_trained', 'n/a')}")
    return model, checkpoint


def rebuild_data(cfg_paths, batch_size: int, ablate: bool = False):
    """Rebuilds transactions+features, split, node index, and per-split
    arrays -- IDENTICAL call sequence to tgn_train.py's main(), guaranteeing
    the same population this checkpoint was actually trained/tested on."""
    fake_cfg = TrainConfig(
        transactions_path=cfg_paths.transactions_path, edge_features_path=cfg_paths.edge_features_path,
        neo4j_uri="", neo4j_user="", neo4j_password="", neo4j_database="",
        epochs=0, warmup_epochs=0, patience=0, batch_size=batch_size, lr=0.0, lr_min=0.0,
        grad_clip_norm=0.0, memory_dim=172, edge_dim=64, time_dim=64, dropout=0.0,
        focal_alpha=0.85, focal_gamma=2.0, seed=42, ablate_edge_features=ablate,
        model_out=Path("."), ablation_out=Path("."), baseline_gnn_metrics_path=Path("."),
        device="cpu",
    )
    df = load_and_join(fake_cfg)
    split = SplitConfig()
    split_labels = assign_temporal_split(df, split)
    node_index = build_node_index(df)

    train_df = df[split_labels == "train"].reset_index(drop=True)
    val_df = df[split_labels == "validation"].reset_index(drop=True)
    test_df = df[split_labels == "test"].reset_index(drop=True)
    log.info(f"Rebuilt split -> train={len(train_df):,} val={len(val_df):,} test={len(test_df):,} events "
              f"(must match the training run's own split log line, since this reuses the identical function)")

    num_nodes = len(node_index)
    train_arrays = prepare_arrays(train_df, node_index, num_nodes, ablate)
    val_arrays = prepare_arrays(val_df, node_index, num_nodes, ablate)
    test_arrays = prepare_arrays(test_df, node_index, num_nodes, ablate)
    return train_arrays, val_arrays, test_arrays, num_nodes


def evaluate_checkpoint(model_path: Path, args, label: str) -> dict:
    """Full evaluation pipeline for one checkpoint. Returns the metrics dict.
    Shared by the primary model and the optional ablation comparison."""
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(model_path, device)
    ablate = "ablation" in label.lower()

    _section(f"Rebuilding data for: {label}")
    train_arrays, val_arrays, test_arrays, num_nodes = rebuild_data(args, args.batch_size, ablate=ablate)
    if num_nodes != checkpoint["num_nodes"]:
        raise ValueError(
            f"Node count mismatch: checkpoint was trained on {checkpoint['num_nodes']:,} nodes, "
            f"but rebuilding from {args.transactions_path} now gives {num_nodes:,}. "
            "The source data has changed since this checkpoint was trained -- "
            "evaluating against a different population than the model saw would be meaningless."
        )

    _section(f"Replaying train+val to build realistic memory state: {label}")
    model.reset_memory()
    t0 = time.time()
    score_split_full(model, train_arrays, args.batch_size, device, record_scores=False)
    log.info(f"Train replay complete in {time.time()-t0:.1f}s ({len(train_arrays[0]):,} events)")
    val_result = score_split_full(model, val_arrays, args.batch_size, device, record_scores=True)
    log.info(f"Validation scored: {len(val_arrays[0]):,} events, {int(val_result['labels'].sum()):,} fraud")

    _section(f"Scoring OOT test set (full, not sampled): {label}")
    test_result = score_split_full(model, test_arrays, args.batch_size, device, record_scores=True)
    n_test = len(test_result["labels"])
    n_fraud = int(test_result["labels"].sum())
    prevalence = n_fraud / n_test if n_test else 0.0
    log.info(f"Test scored: {n_test:,} events, {n_fraud:,} fraud ({prevalence:.4%})")

    roc_auc = float(roc_auc_score(test_result["labels"], test_result["probs"]))
    pr_auc = float(average_precision_score(test_result["labels"], test_result["probs"]))
    precisions, recalls, thresholds = precision_recall_curve(test_result["labels"], test_result["probs"])
    log.info(f"[{label}] OOT ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.6f} "
              f"(random baseline {prevalence:.6f}, lift {pr_auc/max(prevalence,1e-9):.1f}x)")

    alert_budgets = [alert_budget(test_result["labels"], test_result["probs"], frac) for frac in ALERT_BUDGETS]
    for b in alert_budgets:
        log.info(f"[Top {b['alert_pct']:.1f}%] Flagged={b['flagged_volume']:,} | "
                  f"Recall={b['recall']:.2%} | Precision={b['precision']:.2%} | Lift={b['lift']:.1f}x")

    # Threshold calibration -- fit on VALIDATION scores only, exactly the
    # same "never touch test" discipline as train_lgbm_aml.py and the
    # Autoencoder pipeline's threshold_config.json.
    threshold_config = {
        "p95": float(np.percentile(val_result["probs"], 95.0)),
        "p97": float(np.percentile(val_result["probs"], 97.0)),
        "p99": float(np.percentile(val_result["probs"], 99.0)),
        "p99_5": float(np.percentile(val_result["probs"], 99.5)),
        "calibrated_on": "validation split only, never test",
    }
    p99_flags = test_result["probs"] >= threshold_config["p99"]
    p99_count = int(np.sum(p99_flags))
    p99_caught = int(np.sum(test_result["labels"][p99_flags] == 1)) if p99_count else 0
    log.info(f"[Calibrated p99 threshold={threshold_config['p99']:.6f}] Flagged={p99_count:,} | "
              f"Caught={p99_caught:,} | Recall={p99_caught/max(n_fraud,1):.2%}")

    return {
        "label": label, "n_total": n_test, "n_fraud": n_fraud, "prevalence": prevalence,
        "roc_auc": roc_auc, "pr_auc": pr_auc, "alert_budgets": alert_budgets,
        "threshold_config": threshold_config,
        "calibrated_p99": {"flagged": p99_count, "caught": p99_caught,
                             "recall": p99_caught / max(n_fraud, 1)},
        "precisions": precisions, "recalls": recalls, "thresholds": thresholds,
        "checkpoint_reported_val_auc_best": checkpoint.get("val_auc_best"),
        "role_accuracy_heuristic": checkpoint.get("role_accuracy"),
    }


def write_executive_summary(metrics: dict, ablation_metrics: dict | None, report_dir: Path) -> Path:
    ablation_note = ""
    if ablation_metrics is not None:
        delta_roc = metrics["roc_auc"] - ablation_metrics["roc_auc"]
        delta_pr = metrics["pr_auc"] - ablation_metrics["pr_auc"]
        ablation_note = f"""
## Edge-Feature Ablation (Step 5.0's 13 features, zeroed out)
| | With edge features | Edge features zeroed | Contribution |
|---|---|---|---|
| ROC-AUC | {metrics['roc_auc']:.4f} | {ablation_metrics['roc_auc']:.4f} | {delta_roc:+.4f} |
| PR-AUC | {metrics['pr_auc']:.6f} | {ablation_metrics['pr_auc']:.6f} | {delta_pr:+.4f} |

This is the documented evidence (per the Forward Plan's Sprint 9 validation report
requirement) for why the architecture includes the Step 5.0 edge-feature layer
alongside the GDS/graph layer -- required for the Sprint 11 regulatory
explainability documentation.
"""
    else:
        ablation_note = """
## Edge-Feature Ablation
Not run for this evaluation. To produce the documented evidence required for
the Sprint 9 validation report and Sprint 11 explainability documentation,
train a second checkpoint with `--ablate-edge-features` and re-run this
script with `--ablation-model-path` pointing to it.
"""

    report = f"""# MuleGuard.AI — Temporal Graph Network Executive Summary

## Business Result
- **OOT ROC-AUC:** {metrics['roc_auc']:.4f}
- **OOT PR-AUC:** {metrics['pr_auc']:.6f} (vs. random baseline {metrics['prevalence']:.6f})
- **PR-AUC lift over random:** {metrics['pr_auc'] / max(metrics['prevalence'], 1e-9):.1f}x
- **Reported validation AUC at training time:** {metrics['checkpoint_reported_val_auc_best']}

## Data
- OOT test rows (untouched, full population -- not a reservoir sample): {metrics['n_total']:,} ({metrics['n_fraud']:,} fraud, {metrics['prevalence']:.4%})
- Split methodology: last 20% of chronologically-sorted rows, row-fraction split (matches train_lgbm_aml.py's OOT_SPLIT_FRACTION exactly, so this model's test population is the SAME population LightGBM and the Autoencoder are evaluated on)

## Alert-Capacity Metrics
| Alert Budget | Transactions Flagged | Recall | Precision | Lift over random |
|---|---|---|---|---|
"""
    for b in metrics["alert_budgets"]:
        report += (f"| Top {b['alert_pct']:.1f}% | {b['flagged_volume']:,} | "
                    f"{b['recall']:.2%} | {b['precision']:.2%} | {b['lift']:.1f}x |\n")

    report += ablation_note

    report += """
## Interpretation for Non-Technical Stakeholders
This model reasons about relationships between accounts over time, not just
a single transaction's own attributes -- it maintains a running memory for
every account and updates it as new transactions arrive, then scores a new
transaction using both accounts' current memory plus the graph structure
around them. Because it is evaluated on the identical held-out time window
as LightGBM and the Autoencoder, its output can be safely combined with
theirs (e.g. flag a transaction if any model scores it highly) without
double-counting or mismatched test populations.

**Caveat on the role-classification numbers:** this model also produces a
per-account role label (mule / bridge / hub / none), but that head is not
part of the trained loss and is currently checked only against a heuristic
pseudo-label, not investigator-validated ground truth. Do not use the
role-accuracy number as a production quality gate until real labels are
available.
"""
    out_path = report_dir / "executive_summary.md"
    out_path.write_text(report, encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    
    # Configure file logger using the dynamic report dir
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if not any(isinstance(h, logging.FileHandler) for h in log.handlers):
        _file_handler = logging.FileHandler(args.report_dir / "evaluation_log.txt", mode="w")
        _file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
        log.addHandler(_file_handler)

    _section("MuleGuard.AI — TGN Enterprise Validation Suite")
    _log_memory("startup")

    torch_threads = max(1, (os.cpu_count() or 4) - RESERVED_CORES)
    torch.set_num_threads(torch_threads)
    log.info(f"Resource guardrail: torch.set_num_threads({torch_threads})")

    metrics = evaluate_checkpoint(args.model_path, args, label="primary")
    ablation_metrics = None
    if args.ablation_model_path is not None:
        _section("Evaluating ablation checkpoint (edge features zeroed)")
        ablation_metrics = evaluate_checkpoint(args.ablation_model_path, args, label="ablation")

    _section("Writing reports")
    pr_curve_path = plot_pr_curve(metrics["precisions"], metrics["recalls"], metrics["pr_auc"], args.report_dir)
    log.info(f"Saved PR curve -> {pr_curve_path}")
    
    f1_curve_path = plot_f1_curve(metrics["precisions"], metrics["recalls"], metrics["thresholds"], args.report_dir)
    log.info(f"Saved F1 curve -> {f1_curve_path}")

    (args.report_dir / "threshold_config.json").write_text(
        json.dumps(_json_safe(metrics["threshold_config"]), indent=2), encoding="utf-8"
    )
    log.info(f"Wrote {args.report_dir / 'threshold_config.json'}")

    metrics_out = {k: v for k, v in metrics.items() if k not in ("precisions", "recalls", "thresholds")}
    if ablation_metrics is not None:
        ablation_out = {k: v for k, v in ablation_metrics.items() if k not in ("precisions", "recalls", "thresholds")}
        metrics_out["ablation"] = ablation_out
        (args.report_dir / "edge_feature_ablation.json").write_text(
            json.dumps(_json_safe({
                "with_edge_features": {"roc_auc": metrics["roc_auc"], "pr_auc": metrics["pr_auc"]},
                "edge_features_zeroed": {"roc_auc": ablation_metrics["roc_auc"], "pr_auc": ablation_metrics["pr_auc"]},
                "roc_auc_contribution": metrics["roc_auc"] - ablation_metrics["roc_auc"],
                "pr_auc_contribution": metrics["pr_auc"] - ablation_metrics["pr_auc"],
            }), indent=2), encoding="utf-8"
        )
        log.info(f"Wrote {args.report_dir / 'edge_feature_ablation.json'}")

    (args.report_dir / "evaluation_metrics.json").write_text(
        json.dumps(_json_safe(metrics_out), indent=2), encoding="utf-8"
    )
    log.info(f"Wrote {args.report_dir / 'evaluation_metrics.json'}")

    summary_path = write_executive_summary(metrics, ablation_metrics, args.report_dir)
    log.info(f"Wrote {summary_path}")

    _section("VALIDATION COMPLETE")
    log.info(f"ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.6f}")


if __name__ == "__main__":
    main()