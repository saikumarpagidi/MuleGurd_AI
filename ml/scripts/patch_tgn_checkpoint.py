"""
ml/scripts/patch_tgn_checkpoint.py
====================================
MuleGuard.AI — Sprint 7 Fix

Problem:
    tgn_v1_best.pt was saved with num_nodes=0 and empty node_index.
    The training run completed successfully (val_auc=0.9873, test_auc=0.9706)
    but the final _save_checkpoint call somehow wrote node_index as {} (empty dict).
    
    This causes IndexError in TGNMemory.compute_delta_t:
        self._last_update = np.zeros(0, dtype=np.float32)
        last = self._last_update[ids]   # IndexError: out of bounds for size 0

Fix:
    1. Reload the transactions CSV/parquet to reconstruct the exact node_index
       (same deterministic enumeration used in tgn_train.py::build_node_index)
    2. Patch num_nodes and node_index back into the checkpoint
    3. Re-save the checkpoint to the same path

Usage:
    python ml/scripts/patch_tgn_checkpoint.py

Run from the project root (c:\\Users\\INP\\Documents\\MuleGurd_AI).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT_PATH = PROJECT_ROOT / "ml" / "models" / "tgn" / "tgn_v1_best.pt"
DATA_PATH = PROJECT_ROOT / "ml" / "data" / "processed" / "transactions_clean.parquet"

# Fallback: raw CSV path (in case parquet isn't present)
DATA_PATH_CSV = PROJECT_ROOT / "ml" / "data" / "raw" / "HI-Small_Trans.csv"


def build_node_index(df: pd.DataFrame) -> dict[str, int]:
    """Same logic as tgn_train.py::build_node_index — must be identical."""
    accounts = pd.unique(pd.concat([df["from_account"], df["to_account"]]))
    return {str(acc): i for i, acc in enumerate(accounts)}


def main() -> int:
    print(f"[patch] Checkpoint: {CKPT_PATH}")
    if not CKPT_PATH.exists():
        print(f"[ERROR] Checkpoint not found at {CKPT_PATH}")
        return 1

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print("[patch] Loading checkpoint ...")
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    print(f"[patch] Keys: {list(ckpt.keys())}")
    print(f"[patch] Current num_nodes: {ckpt.get('num_nodes', '(missing)')}")
    print(f"[patch] Current node_index len: {len(ckpt.get('node_index', {}))}")

    if ckpt.get("num_nodes", 0) > 0 and len(ckpt.get("node_index", {})) > 0:
        print("[patch] Checkpoint already has a valid node_index — nothing to do.")
        return 0

    # ── Load dataset ──────────────────────────────────────────────────────────
    if DATA_PATH.exists():
        print(f"[patch] Loading dataset from {DATA_PATH} ...")
        df = pd.read_parquet(DATA_PATH)
        # Normalise column names
        col_map = {}
        for col in df.columns:
            lc = col.lower().replace(" ", "_")
            if "from" in lc and "account" in lc:
                col_map[col] = "from_account"
            elif "to" in lc and "account" in lc:
                col_map[col] = "to_account"
        df = df.rename(columns=col_map)
    elif DATA_PATH_CSV.exists():
        print(f"[patch] Loading dataset from CSV {DATA_PATH_CSV} ...")
        df = pd.read_csv(DATA_PATH_CSV)
        # HI-Small column names: "Account", "Account.1" — remap
        rename = {}
        cols = list(df.columns)
        # Detect account columns by common HI-Small header patterns
        for c in cols:
            cl = c.strip().lower()
            if cl in ("account", "from account", "from_account"):
                rename[c] = "from_account"
            elif cl in ("account.1", "to account", "to_account"):
                rename[c] = "to_account"
        df = df.rename(columns=rename)
    else:
        print(f"[ERROR] Neither {DATA_PATH} nor {DATA_PATH_CSV} found.")
        print("  Run the feature pipeline first to produce transactions_clean.parquet")
        return 1

    if "from_account" not in df.columns or "to_account" not in df.columns:
        print(f"[ERROR] Could not find from_account/to_account columns. Found: {list(df.columns[:10])}")
        return 1

    print(f"[patch] Dataset loaded: {len(df):,} rows")

    # ── Build node index ──────────────────────────────────────────────────────
    print("[patch] Building node index ...")
    node_index = build_node_index(df)
    num_nodes = len(node_index)
    print(f"[patch] Built node_index: {num_nodes:,} unique accounts")

    # ── Validate against model weight shape ──────────────────────────────────
    # TGNMemory has a GRUCell whose hidden_size = memory_dim.
    # The model weights don't contain the memory tensors (they're numpy, excluded
    # from state_dict), so we can't cross-check num_nodes from weights alone.
    # But we can check memory_dim is consistent.
    memory_dim = ckpt.get("memory_dim", 172)
    print(f"[patch] memory_dim from checkpoint: {memory_dim}")

    # ── Patch and save ────────────────────────────────────────────────────────
    ckpt["num_nodes"] = num_nodes
    ckpt["node_index"] = node_index

    backup_path = CKPT_PATH.with_suffix(".pre_patch.pt")
    if not backup_path.exists():
        import shutil
        shutil.copy2(str(CKPT_PATH), str(backup_path))
        print(f"[patch] Backup saved to {backup_path}")

    torch.save(ckpt, str(CKPT_PATH))
    print(f"[patch] ✅ Patched checkpoint saved to {CKPT_PATH}")
    print(f"[patch] num_nodes={num_nodes:,} | node_index entries={len(node_index):,}")

    # ── Quick verification ────────────────────────────────────────────────────
    verify = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    assert verify["num_nodes"] == num_nodes, "Verification failed: num_nodes mismatch"
    assert len(verify["node_index"]) == num_nodes, "Verification failed: node_index length mismatch"
    print("[patch] ✅ Verification passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
