"""
ml/training/tgn_model.py
==========================
MuleGuard.AI — Sprint 5

Temporal Graph Network architecture, implementing exactly the formulation
specified in the Forward Plan / graph-architecture reference doc:

    Dynamic Edge Representation:
        h_channel = Embedding(payment_format_encoded) in R^8
        e_uv = Linear2(ReLU(Linear1([x_continuous_12, h_channel]))) in R^edge_dim

    Continuous Fourier Harmonic Time Vector:
        phi_k(dt) = cos(omega_k * dt + psi_k),  omega_k = 10^(-4k/64),  k=1..time_dim

    Dynamic Node Memory Update (GRU):
        m_u(t) = GRUCell([m_u(t-), e_uv, phi(dt_u)], m_u(t-))  in R^memory_dim

    Graph Attention Layers: 2x TransformerConv, propagate over 2-hop neighborhood

    Interaction Risk Scoring (edge-level, used at serving time):
        z_uv = [m_u(t), m_v(t), e_uv, phi(dt)]
        risk = sigmoid(Linear(Dropout(ReLU(Linear(z_uv)))))

    Output Heads (node-level, used for offline evaluation):
        Head 1: P(is_mule) -- binary
        Head 2: P(role)    -- softmax over [HUB, BRIDGE, MULE, NONE]

Why a custom implementation rather than PyG's built-in TGNMemory module:
    PyG ships a TGNMemory class, but it hardcodes a specific message
    function shape and doesn't expose the exact Fourier time encoding or
    dual-head design this project's spec calls for. Building it explicitly
    here keeps every formula traceable back to the plan's equations (each
    method below is annotated with which line of the spec it implements),
    which matters for the Sprint 11 model-governance/explainability
    documentation this project ultimately needs.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

# Must match ml/features/build_tgn_edge_features.py's PAYMENT_FORMAT_ID.
NUM_PAYMENT_FORMATS = 7 + 1  # +1 for an "unknown/unseen" bucket
CONTINUOUS_EDGE_DIM = 12  # the 13 edge features minus payment_format_encoded (embedded separately)


class FourierTimeEncoding(nn.Module):
    """phi_k(dt) = cos(omega_k * dt + psi_k), omega_k = 10^(-4k/64).

    omega_k is FIXED (not learned) per the spec's formula — this gives a
    fixed bank of "clocks" running at exponentially spaced frequencies (from
    very fast to very slow), a standard technique (used in the original TGN
    and Time2Vec papers) for letting a GRU-based memory reason about
    elapsed time without needing to see raw, unbounded delta-t values
    directly. psi_k is a learned per-frequency phase offset.
    """

    def __init__(self, time_dim: int = 64):
        super().__init__()
        self.time_dim = time_dim
        k = torch.arange(1, time_dim + 1, dtype=torch.float32)
        omega = torch.pow(10.0, -4.0 * k / 64.0)
        self.register_buffer("omega", omega)  # fixed, not a learned parameter
        self.psi = nn.Parameter(torch.zeros(time_dim))  # learned phase offsets

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """delta_t: [batch] elapsed time (seconds) since the node's last event.
        Returns: [batch, time_dim]
        """
        delta_t = delta_t.unsqueeze(-1)  # [batch, 1]
        return torch.cos(self.omega.unsqueeze(0) * delta_t + self.psi.unsqueeze(0))


class EdgeEncoder(nn.Module):
    """e_uv = Linear2(ReLU(Linear1([x_continuous_12, h_channel])))

    Fuses the 12 continuous Step-5.0 edge features with a learned payment-
    channel embedding into a single dense edge representation.
    """

    def __init__(self, continuous_dim: int = CONTINUOUS_EDGE_DIM, channel_embed_dim: int = 8, edge_dim: int = 64):
        super().__init__()
        self.channel_embedding = nn.Embedding(NUM_PAYMENT_FORMATS, channel_embed_dim)
        in_dim = continuous_dim + channel_embed_dim
        self.linear1 = nn.Linear(in_dim, in_dim * 2)
        self.linear2 = nn.Linear(in_dim * 2, edge_dim)

    def forward(self, x_continuous: torch.Tensor, payment_format_id: torch.Tensor) -> torch.Tensor:
        """x_continuous: [batch, 12], payment_format_id: [batch] (long). Returns [batch, edge_dim]."""
        h_channel = self.channel_embedding(payment_format_id)
        fused = torch.cat([x_continuous, h_channel], dim=-1)
        return self.linear2(F.relu(self.linear1(fused)))


class TGNMemory(nn.Module):
    """m_u(t) = GRUCell([m_u(t-), e_uv, phi(dt_u)], m_u(t-))

    Maintains a persistent memory vector per account, updated every time
    that account participates in a transaction (as sender OR receiver).

    BSOD FIX (0x0000001A / 0x41792 PFN Page-Transition Corruption):
    The original implementation used register_buffer() to store the memory
    as a single 1.43 GB PyTorch tensor (2,076,999 x 172 x float32).
    During epoch 2+ training, Windows attempted to page-swap portions of
    this large virtual memory region to disk (pagefile) while the training
    loop simultaneously indexed random offsets into it (different account IDs
    per batch). This race between the Windows Memory Manager's page-out path
    and the Python/PyTorch access path corrupted the PFN (Page Frame Number)
    list, triggering the BSOD.

    Fix: store memory in N_SHARDS separate numpy arrays, each ~179 MB.
    No single allocation triggers the large-page / section-object path.
    Windows can independently manage each shard without PFN corruption.
    Tensors are created per-batch as narrow views for GRU computation.
    """
    N_SHARDS = 8  # 8 × 179 MB = safe, well under large-page threshold

    def __init__(self, num_nodes: int, memory_dim: int, edge_dim: int, time_dim: int):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_nodes = num_nodes
        self.gru = nn.GRUCell(input_size=edge_dim + time_dim, hidden_size=memory_dim)

        # Sharded numpy arrays — never a single large contiguous block.
        # Each shard covers a contiguous range of account IDs.
        shard_size = (num_nodes + self.N_SHARDS - 1) // self.N_SHARDS
        self._shard_size = shard_size
        self._memory_shards = [
            np.zeros((min(shard_size, num_nodes - i * shard_size), memory_dim), dtype=np.float32)
            for i in range(self.N_SHARDS)
            if i * shard_size < num_nodes
        ]
        self._last_update = np.zeros(num_nodes, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Public API (same as before — callers unchanged)                      #
    # ------------------------------------------------------------------ #

    def reset_state(self) -> None:
        for shard in self._memory_shards:
            shard[:] = 0.0
        self._last_update[:] = 0.0

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Gather memory rows for the given node IDs → [batch, memory_dim] tensor."""
        ids = node_ids.cpu().numpy()
        rows = self._gather(ids)
        return torch.from_numpy(rows)

    def compute_delta_t(self, node_ids: torch.Tensor, current_time: torch.Tensor) -> torch.Tensor:
        ids = node_ids.cpu().numpy()
        last = self._last_update[ids]
        return current_time - torch.from_numpy(last).to(current_time.device)

    def update(self, node_ids: torch.Tensor, edge_repr: torch.Tensor,
               time_repr: torch.Tensor, current_time: torch.Tensor) -> torch.Tensor:
        """Advance per-account memory via GRUCell for the given node IDs.

        Returns the post-GRU tensor WITH grad_fn so the caller (step()) can
        wire it into the z_uv computation, creating a gradient path from
        loss → risk_head → GRU weights.

        Gradient strategy (standard TGN — Rossi et al. 2020):
          - `prev_memory` is read from numpy storage → NOT in the autograd graph
            (this is the "detach across timesteps" that prevents BPTT over the
            full event history, which would be O(dataset_size) memory).
          - The GRUCell forward for the *current* batch IS in the autograd graph
            when called from the training path (run_split with optimizer).
            This lets the GRU's weight matrices receive gradients from the
            current batch's loss.
          - After the GRU computation, new_memory is detached before being
            written back to numpy storage for the next batch.

        Calling contexts:
          - run_warmup() → decorated @torch.no_grad() → GRU computes but no
            graph is built → correct (warmup should not train).
          - run_split(optimizer=None) → wrapped with torch.no_grad() at call
            site → no graph → correct for val/test.
          - run_split(optimizer=...) → no outer no_grad → GRU trains → correct.
        """
        ids = node_ids.cpu().numpy()
        prev_memory_np = self._gather(ids)
        # Deliberately NOT requires_grad — detached across time (standard TGN)
        prev_memory = torch.from_numpy(prev_memory_np)
        gru_input = torch.cat([edge_repr.detach().cpu(), time_repr.detach().cpu()], dim=-1)
        # GRUCell forward: stays in autograd graph during training path
        new_memory = self.gru(gru_input, prev_memory)
        # Detach ONLY at the numpy storage boundary — not before
        self._scatter(ids, new_memory.detach().cpu().numpy())
        self._last_update[ids] = current_time.cpu().numpy()
        # Return the live tensor (with grad_fn) so step() can use it in z_uv
        return new_memory



    # ------------------------------------------------------------------ #
    # Warmup state save/restore (replaces the 1.43GB .clone() calls)      #
    # ------------------------------------------------------------------ #

    def save_state_to_file(self, path: str) -> None:
        """Persist memory shards + last_update as individual .npy files inside `path`
        (a directory). Using one .npy file per shard avoids the zipfile/CRC-32
        corruption bug in Python 3.10's zipfile module that affects large npz files.
        """
        import os
        os.makedirs(path, exist_ok=True)
        for i, shard in enumerate(self._memory_shards):
            np.save(os.path.join(path, f"shard_{i}.npy"), shard)
        np.save(os.path.join(path, "last_update.npy"), self._last_update)

    def load_state_from_file(self, path: str) -> None:
        """Restore memory shards + last_update from individual .npy files in `path`."""
        import os
        for i in range(len(self._memory_shards)):
            self._memory_shards[i][:] = np.load(os.path.join(path, f"shard_{i}.npy"))
        self._last_update[:] = np.load(os.path.join(path, "last_update.npy"))

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _gather(self, ids: np.ndarray) -> np.ndarray:
        """Read memory rows for `ids` from shards into a new [len(ids), memory_dim] array."""
        out = np.empty((len(ids), self.memory_dim), dtype=np.float32)
        for i, nid in enumerate(ids):
            si = int(nid) // self._shard_size
            local = int(nid) % self._shard_size
            out[i] = self._memory_shards[si][local]
        return out

    def _scatter(self, ids: np.ndarray, values: np.ndarray) -> None:
        """Write memory rows `values` back into shards at positions `ids`."""
        for i, nid in enumerate(ids):
            si = int(nid) // self._shard_size
            local = int(nid) % self._shard_size
            self._memory_shards[si][local] = values[i]


class TemporalGraphNetwork(nn.Module):
    """Full TGN: memory + message function + Fourier time + 2-hop attention
    propagation + dual output heads + edge-level interaction risk scoring.
    """

    def __init__(
        self,
        num_nodes: int,
        memory_dim: int = 172,
        edge_dim: int = 64,
        time_dim: int = 64,
        channel_embed_dim: int = 8,
        attn_heads: int = 2,
        dropout: float = 0.2,
        num_roles: int = 4,  # HUB, BRIDGE, MULE, NONE
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.time_encoding = FourierTimeEncoding(time_dim=time_dim)
        self.edge_encoder = EdgeEncoder(
            continuous_dim=CONTINUOUS_EDGE_DIM, channel_embed_dim=channel_embed_dim, edge_dim=edge_dim
        )
        self.memory = TGNMemory(num_nodes=num_nodes, memory_dim=memory_dim, edge_dim=edge_dim, time_dim=time_dim)

        # 2x TransformerConv layers propagate updated memories through the
        # 2-hop neighborhood, producing node embeddings that encode local
        # ring topology on top of raw per-account memory state.
        self.attn1 = TransformerConv(memory_dim, memory_dim, heads=attn_heads, concat=False, dropout=dropout)
        self.attn2 = TransformerConv(memory_dim, memory_dim, heads=attn_heads, concat=False, dropout=dropout)

        # Output Head 1: P(is_mule)
        self.is_mule_head = nn.Sequential(
            nn.Linear(memory_dim, memory_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(memory_dim // 2, 1),
        )
        # Output Head 2: P(role) over [HUB, BRIDGE, MULE, NONE]
        self.role_head = nn.Sequential(
            nn.Linear(memory_dim, memory_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(memory_dim // 2, num_roles),
        )

        # Edge-level interaction risk scoring (z_uv formula) — this is what
        # /graph-score calls at serving time for a single incoming transaction.
        risk_in_dim = memory_dim * 2 + edge_dim + time_dim
        self.risk_head = nn.Sequential(
            nn.Linear(risk_in_dim, risk_in_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(risk_in_dim // 2, 1),
        )

    def reset_memory(self) -> None:
        self.memory.reset_state()

    def encode_edge(self, x_continuous: torch.Tensor, payment_format_id: torch.Tensor) -> torch.Tensor:
        return self.edge_encoder(x_continuous, payment_format_id)

    def step(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        x_continuous: torch.Tensor,
        payment_format_id: torch.Tensor,
        event_time: torch.Tensor,
    ) -> torch.Tensor:
        """Process one batch of temporally-ordered transaction events.

        Returns the edge-level interaction risk probability (z_uv formula).

        Gradient wiring:
          - `e_uv` and `time_repr` carry gradients from the edge encoder and
            time encoding — always trained.
          - `m_src_post` / `m_dst_post` are the GRU outputs for this batch.
            They carry the GRU weights' gradients into the z_uv → risk_head →
            loss graph, so the GRU trains. The previous-step memory
            (read from numpy storage) is NOT in the graph (detach across time).
          - After computing risk_logit, the post-GRU memory is detached and
            written back to numpy storage for the next batch.
        """
        e_uv = self.encode_edge(x_continuous, payment_format_id)

        src_delta_t = self.memory.compute_delta_t(src, event_time)
        dst_delta_t = self.memory.compute_delta_t(dst, event_time)
        src_time_repr = self.time_encoding(src_delta_t)
        dst_time_repr = self.time_encoding(dst_delta_t)

        # Run GRU update — returns post-GRU tensors WITH grad_fn so the GRU
        # weights appear in the computation graph.
        m_src_post = self.memory.update(src, e_uv, src_time_repr, event_time)
        m_dst_post = self.memory.update(dst, e_uv, dst_time_repr, event_time)

        # z_uv uses POST-update memory (the GRU's output for this batch)
        # so loss gradients flow back through the GRU weights.
        z_uv = torch.cat([m_src_post, m_dst_post, e_uv, src_time_repr], dim=-1)
        risk_logit = self.risk_head(z_uv).squeeze(-1)

        return risk_logit

    def propagate_embeddings(self, node_ids: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Runs the 2x TransformerConv layers over the CURRENT memory state
        for a snapshot of the graph, producing node embeddings for the
        node-level output heads (is_mule / role). `edge_index` should be the
        2-hop neighborhood subgraph around `node_ids` (built by the caller,
        e.g. via PyG's k_hop_subgraph utility), per the "2-hop" spec.
        """
        import numpy as np
        full_memory_np = np.concatenate(self.memory._memory_shards, axis=0)
        full_memory = torch.from_numpy(full_memory_np).to(edge_index.device)
        h = self.attn1(full_memory, edge_index)

        h = F.elu(h)
        h = self.attn2(h, edge_index)
        h = F.elu(h)
        return h[node_ids]

    def predict_node(self, node_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (is_mule_logit [batch], role_logits [batch, num_roles])."""
        is_mule_logit = self.is_mule_head(node_embedding).squeeze(-1)
        role_logits = self.role_head(node_embedding)
        return is_mule_logit, role_logits
