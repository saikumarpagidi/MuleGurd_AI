"""
services/graph-service/app/core/tgn_scorer.py
===============================================
MuleGuard.AI — Sprint 5

Loads the trained TGN (tgn_v1_best.pt) and exposes a single entry point:

    score_transaction(from_account, to_account, edge_features, event_time_s)
        -> TGNInferenceResult

Serving-time architecture (matches the spec's z_uv formula):
  1. Fetch src/dst node IDs from the checkpoint's node_index dict
  2. Build edge features (x_continuous + payment_format_id)
  3. Call model.step() → pre-update memory risk logit (correct for real-time
     fraud prevention — see TemporalGraphNetwork.step() docstring)
  4. Advance memory state (so subsequent requests see this transaction's effect)

Memory management:
  - The TGN memory bank is SHARED STATE across all concurrent requests.
    An asyncio.Lock guards every model.step() call so memory updates are
    applied in arrival order (FIFO), matching the model's chronological
    training assumption. This is correct for a single-process server.
  - For multi-process deployments (k8s horizontal scaling), a shared
    Redis-backed memory store would be needed — that is a Sprint 10 item.

Graceful degradation:
  - If tgn_v1_best.pt is missing or incompatible → GDS heuristic mode.
  - If a requested account_id is not in the training node_index (OOV) →
    use a dummy node_id (0) with a warning. This is intentional: unknown
    accounts have no history, which is itself a risk signal (new account
    immediately transacting large amounts). The model's zero-initialized
    memory for node 0 will produce a conservative mid-range score.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("graph-service.tgn_scorer")

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# Payment format ID mapping — must match tgn_edge_feature_spec.json
PAYMENT_FORMAT_MAP: dict[str, int] = {
    "ACH": 0, "Cash": 1, "Cheque": 2, "Credit Card": 3,
    "Bitcoin": 4, "Reinvestment": 5, "Wire": 6, "OTHER": 7,
}
PAYMENT_FORMAT_UNKNOWN = 7


@dataclass
class TGNInferenceResult:
    tgn_score: float          # Risk probability in [0, 1]
    is_mule_probability: float# Probability src account is a mule (v4 node head)
    ring_role: str            # HUB | BRIDGE | MULE | NONE
    ring_probability: float   # Confidence of ring_role prediction
    in_scc_ring: bool         # Whether src account is in a topological SCC ring
    degraded_mode: bool       # True if heuristic fallback was used
    processing_time_ms: float
    model_version: str        # Checkpoint filename used


class TGNScorer:
    """Lifecycle-managed TGN inference wrapper. One instance per process."""

    def __init__(self, model_path: str, num_nodes_expected: int,
                 memory_dim: int = 172, edge_dim: int = 64, time_dim: int = 64):
        self._model_path = Path(model_path)
        self._num_nodes_expected = num_nodes_expected
        self._memory_dim = memory_dim
        self._edge_dim = edge_dim
        self._time_dim = time_dim

        self._model = None        # TemporalGraphNetwork instance
        self._node_index: dict[str, int] = {}  # account_id → int tensor index
        self._device = None
        self._loaded = False
        self._lock = asyncio.Lock()  # guard memory updates
        self._model_version = "not_loaded"

    def load(self) -> None:
        """Load TGN checkpoint at service startup (synchronous — called in lifespan)."""
        if not _HAS_TORCH:
            log.warning("PyTorch not installed. TGN scorer will run in heuristic-only mode.")
            return
        if not self._model_path.exists():
            log.warning("TGN checkpoint not found at %s. Heuristic fallback mode active.", self._model_path)
            return

        try:
            from .tgn_model import TemporalGraphNetwork

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(self._model_path, map_location=self._device, weights_only=False)

            num_nodes = checkpoint.get("num_nodes", self._num_nodes_expected)
            memory_dim = checkpoint.get("memory_dim", self._memory_dim)
            edge_dim = checkpoint.get("edge_dim", self._edge_dim)
            time_dim = checkpoint.get("time_dim", self._time_dim)
            node_index = checkpoint.get("node_index", {})

            # Guard: a checkpoint with num_nodes=0 or empty node_index is unusable.
            # This happens when the training script saves an early/interrupted checkpoint
            # without the final node_index. Patch with ml/scripts/patch_tgn_checkpoint.py.
            if num_nodes == 0 or len(node_index) == 0:
                log.error(
                    "TGN checkpoint %s has num_nodes=%d and node_index with %d entries. "
                    "This checkpoint is MISSING its node_index. "
                    "Fix: run `python ml/scripts/patch_tgn_checkpoint.py` to patch it. "
                    "Falling back to heuristic mode.",
                    self._model_path.name, num_nodes, len(node_index),
                )
                self._loaded = False
                return

            model = TemporalGraphNetwork(
                num_nodes=num_nodes,
                memory_dim=memory_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
            ).to(self._device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            self._model = model
            self._node_index = node_index
            self._memory_dim = memory_dim
            self._model_version = self._model_path.name
            self._loaded = True
            log.info(
                "TGN model loaded: %s | num_nodes=%d | node_index=%d accounts | device=%s",
                self._model_path.name, num_nodes, len(self._node_index), self._device,
            )
        except Exception as exc:
            log.error("Failed to load TGN model from %s: %s — heuristic fallback active.", self._model_path, exc)
            self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None

    def _resolve_node_id(self, account_id: str) -> int:
        """Map account_id string → integer node index.

        OOV accounts (not seen during training) are mapped to node 0, which
        has zero-initialized memory — a valid 'no history' representation.
        The caller should treat OOV accounts as a mild risk uplift signal
        (new account transacting immediately with high amounts is itself
        suspicious), but we do NOT raise an exception on OOV.
        """
        idx = self._node_index.get(account_id)
        if idx is None:
            log.debug("OOV account_id '%s' — mapped to node 0 (zero memory).", account_id)
            return 0
        return idx

    def _build_edge_tensor(
        self,
        amount: float,
        payment_format: str,
        timestamp_s: float,
        from_account: str,
        to_account: str,
    ) -> tuple:
        """Build the 12 continuous edge features + payment_format_id tensor.

        Feature order matches CONTINUOUS_COLS in tgn_train.py and
        tgn_edge_feature_spec.json exactly:
            log_amount_paid, is_cross_bank, is_cash_payment, is_bitcoin_transaction,
            structuring_threshold_flag, amount_ratio, amount_difference,
            hour_sin, hour_cos, weekday_sin, weekday_cos, is_cross_currency
        """
        import datetime

        dt = datetime.datetime.fromtimestamp(timestamp_s, tz=datetime.timezone.utc)
        hour = dt.hour
        weekday = dt.weekday()
        fmt_id = PAYMENT_FORMAT_MAP.get(payment_format, PAYMENT_FORMAT_UNKNOWN)

        # Compute from-bank != to-bank (best effort: prefix match on account_id)
        from_prefix = str(from_account)[:4] if from_account else ""
        to_prefix = str(to_account)[:4] if to_account else ""
        is_cross_bank = 1.0 if from_prefix != to_prefix else 0.0

        x_continuous = np.array([
            math.log1p(amount),                                  # log_amount_paid
            is_cross_bank,                                       # is_cross_bank
            1.0 if payment_format == "Cash" else 0.0,           # is_cash_payment
            1.0 if payment_format == "Bitcoin" else 0.0,        # is_bitcoin_transaction
            1.0 if 8_000 <= amount <= 9_999.99 else 0.0,        # structuring_threshold_flag
            min(amount / (amount + 1e-6), 5.0),                 # amount_ratio (conservative)
            0.0,                                                 # amount_difference (unknown at serving)
            math.sin(2 * math.pi * hour / 24),                  # hour_sin
            math.cos(2 * math.pi * hour / 24),                  # hour_cos
            math.sin(2 * math.pi * weekday / 7),                # weekday_sin
            math.cos(2 * math.pi * weekday / 7),                # weekday_cos
            0.0,                                                 # is_cross_currency (unknown)
        ], dtype=np.float32)

        return x_continuous, fmt_id

    async def score(
        self,
        from_account: str,
        to_account: str,
        amount: float,
        payment_format: str,
        timestamp_s: float,
        neo4j_src_props: dict | None = None,
    ) -> TGNInferenceResult:
        """Score a single transaction using the TGN or heuristic fallback."""
        t0 = time.perf_counter()

        if self.is_loaded:
            result = await self._score_tgn(from_account, to_account, amount, payment_format,
                                           timestamp_s, neo4j_src_props)
        else:
            result = self._score_heuristic(neo4j_src_props)

        result.processing_time_ms = (time.perf_counter() - t0) * 1000
        return result

    async def _score_tgn(
        self,
        from_account: str,
        to_account: str,
        amount: float,
        payment_format: str,
        timestamp_s: float,
        neo4j_src_props: dict | None,
    ) -> TGNInferenceResult:
        """Real TGN inference path — calls model.step() under asyncio.Lock."""
        src_id = self._resolve_node_id(from_account)
        dst_id = self._resolve_node_id(to_account)
        x_continuous, fmt_id = self._build_edge_tensor(amount, payment_format, timestamp_s,
                                                        from_account, to_account)

        src_t = torch.tensor([src_id], dtype=torch.long, device=self._device)
        dst_t = torch.tensor([dst_id], dtype=torch.long, device=self._device)
        x_cont_t = torch.from_numpy(x_continuous).unsqueeze(0).to(self._device)
        fmt_t = torch.tensor([fmt_id], dtype=torch.long, device=self._device)
        evt_t = torch.tensor([timestamp_s], dtype=torch.float32, device=self._device)

        # Memory bank is shared state — protect all reads+writes with the lock
        async with self._lock:
            with torch.no_grad():
                risk_logit = self._model.step(src_t, dst_t, x_cont_t, fmt_t, evt_t)
                gnn_score = float(torch.sigmoid(risk_logit).item())

                # Node-level role prediction directly from the src account's memory vector.
                # NOTE: propagate_embeddings() concatenates ALL 2M node memories into a
                # single tensor for TransformerConv propagation — that is correct for
                # offline evaluation (batch), but at serving time it would:
                #   (a) allocate a 2M×172 = ~1.4 GB tensor on every request, and
                #   (b) run attention over the full graph instead of the local neighborhood
                #       (the mini_edge of 2 nodes is not a valid 2-hop neighborhood).
                # Instead, read the src account's post-update memory directly and pass it
                # to predict_node. This gives the role head the same information it uses
                # after training (m_u(t) for the source account).
                # v4: Extract both mule prediction and role prediction
                src_memory = self._model.memory.get_memory(src_t)  # [1, memory_dim]
                is_mule_logit, role_logits = self._model.predict_node(src_memory)
                
                is_mule_probability = float(torch.sigmoid(is_mule_logit).item())
                
                role_probs = torch.softmax(role_logits, dim=-1).squeeze(0)
                role_idx = int(role_probs.argmax().item())
                ring_probability = float(role_probs[role_idx].item())

        role_names = ["NONE", "MULE", "BRIDGE", "HUB"]
        ring_role = role_names[role_idx]
        in_scc_ring = bool(neo4j_src_props.get("in_scc_ring", False)) if neo4j_src_props else False

        return TGNInferenceResult(
            tgn_score=gnn_score,
            is_mule_probability=is_mule_probability,
            ring_role=ring_role,
            ring_probability=ring_probability,
            in_scc_ring=in_scc_ring,
            degraded_mode=False,
            processing_time_ms=0.0,  # filled by caller
            model_version=self._model_version,
        )

    def _score_heuristic(self, neo4j_props: dict | None) -> TGNInferenceResult:
        """Fallback when TGN model is not loaded. Derives a risk score purely
        from GDS graph properties (PageRank + SCC ring membership).

        Heuristic formula (from settings defaults):
            score = 0.6 * normalized_pagerank + 0.3 * in_scc_ring + 0.1 * community_signal

        This gives a meaningful structural risk signal even before TGN training
        completes, making Sprint 6 validation possible without trained weights.
        """
        if not neo4j_props:
            # Completely unknown account — use conservative mid-range
            return TGNInferenceResult(
                tgn_score=0.5,
                is_mule_probability=0.0,
                ring_role="NONE",
                ring_probability=0.5,
                in_scc_ring=False,
                degraded_mode=True,
                processing_time_ms=0.0,
                model_version="heuristic_no_neo4j",
            )

        pagerank = float(neo4j_props.get("pagerank", 0.0))
        in_scc_ring = bool(neo4j_props.get("in_scc_ring", False))
        community_id = int(neo4j_props.get("community_id", -1))

        # Normalize pagerank: typical values range 0–0.01 for individual accounts
        # Clamp and scale to [0, 1]
        normalized_pr = min(pagerank / 0.01, 1.0)

        scc_signal = 1.0 if in_scc_ring else 0.0
        community_signal = 0.5 if community_id > 0 else 0.0  # non-singleton community

        heuristic_score = (0.6 * normalized_pr + 0.3 * scc_signal + 0.1 * community_signal)
        heuristic_score = float(min(max(heuristic_score, 0.0), 1.0))

        ring_role = "MULE" if in_scc_ring else "NONE"

        return TGNInferenceResult(
            tgn_score=heuristic_score,
            is_mule_probability=0.0,
            ring_role=ring_role,
            ring_probability=0.6 if in_scc_ring else 0.4,
            in_scc_ring=in_scc_ring,
            degraded_mode=True,
            processing_time_ms=0.0,
            model_version="heuristic_gds_only",
        )
