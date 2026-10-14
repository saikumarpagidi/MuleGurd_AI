"""
services/graph-service/app/core/config.py
==========================================
MuleGuard.AI — Sprint 5 (graph-service configuration)

All runtime settings driven by environment variables with sane local defaults
so the service works identically in docker-compose, k8s, and local dev.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


import os
from pathlib import Path

# When running natively from services/graph-service, we need to go up 4 directories to reach workspace root.
# In Docker, WORKDIR is /app and ml/models is mapped directly to /app/ml/models.
_cwd_model_path = Path("ml/models/tgn/tgn_v4_best.pt")
_relative_model_path = Path(__file__).resolve().parent.parent.parent.parent.parent / "ml" / "models" / "tgn" / "tgn_v4_best.pt"
DEFAULT_TGN_PATH = str(_cwd_model_path) if _cwd_model_path.exists() else str(_relative_model_path)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GRAPH_SERVICE_", extra="ignore")

    # ── Neo4j Bolt connection ────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    # Connection pool size for the async Neo4j driver
    neo4j_max_connection_pool_size: int = 10
    # Per-query timeout in seconds (must be < 25ms latency budget)
    neo4j_query_timeout_s: float = 0.020  # 20ms

    # ── TGN model artifact ───────────────────────────────────────────────────
    tgn_model_path: str = DEFAULT_TGN_PATH
    # Number of nodes the model was trained on; used to validate checkpoint compatibility
    tgn_num_nodes_expected: int = 2_076_999
    tgn_memory_dim: int = 172
    tgn_edge_dim: int = 64
    tgn_time_dim: int = 64

    # ── Latency budget ───────────────────────────────────────────────────────
    # Total wall-clock latency allowed for /graph-score response.
    # This matches the Sprint 5 gate requirement: /graph-score < 25ms.
    latency_budget_ms: float = 25.0

    # ── Fallback heuristic weights (when TGN model is not loaded) ────────────
    # These are used to compute a degraded score from GDS graph properties only.
    heuristic_pagerank_weight: float = 0.6
    heuristic_scc_weight: float = 0.3
    heuristic_community_weight: float = 0.1

    # ── Service ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"


settings = Settings()
