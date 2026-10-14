"""
services/ai-engine/app/inference.py
=====================================
Model loading and scoring logic for the /score endpoint (Sprint 4C).

Loads LightGBM + Autoencoder artifacts ONCE at service startup (not per
request - that would blow the <200ms latency budget) and exposes a single
`score_transaction()` call that the FastAPI route delegates to.

SHAP note: uses shap.TreeExplainer, the fast, exact, model-specific explainer
for tree ensembles (as opposed to the slow, sampling-based KernelExplainer).
This is what makes a per-request SHAP computation feasible inside a 200ms
latency budget at all.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import shap
import torch
import torch.nn as nn

from .config import Settings
from .schemas import (
    AeAnomalyDirection,
    AeFeatureContribution,
    DecisionPreliminary,
    FeatureSource,
    ScoreResponse,
    ShapDirection,
    ShapFeatureContribution,
)

log = logging.getLogger("ai-engine.inference")

DEFAULT_TOP_K = 5


# --------------------------------------------------------------------------- #
# Autoencoder architecture - must match the v3 trained checkpoint
# --------------------------------------------------------------------------- #
# Encoder: Linear -> LayerNorm -> GELU -> Dropout (x2), then Linear to the
# bottleneck (default 12). The decoder mirrors the encoder (no dropout).
# Dimensions (input_dim, hidden_dim, bottleneck_dim, dropout_rate) are read
# from the checkpoint when present; the defaults below are the v3 values.


class Autoencoder(nn.Module):
    """Deep autoencoder with LayerNorm and GELU, matching the v3 trained artifact."""

    def __init__(
        self,
        input_dim: int = 21,
        hidden_dim: int = 128,
        bottleneck_dim: int = 12,
        dropout_rate: float = 0.15,
    ):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def _resolve_model_path(relative_path: str | Path) -> Path:
    """Resolve a model path relative to CWD first, then project root.
    Prevents FileNotFoundError when the service is started from a
    non-root working directory (e.g. inside services/ai-engine/).
    """
    p = Path(relative_path)
    if p.exists():
        return p
    root_p = Path(__file__).resolve().parent.parent.parent.parent / relative_path
    if root_p.exists():
        return root_p
    return p  # return original so error messages show the intended path


def _load_torch_checkpoint(path: Path):
    """Prefer the safe loader (no arbitrary pickle execution); fall back only
    if the checkpoint contains non-tensor objects that require full unpickling."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "weights_only=True load failed for %s (%s); falling back to "
            "weights_only=False. Only do this for trusted checkpoints.",
            path, exc,
        )
        return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #


class ModelRegistry:
    """Owns every loaded model artifact + derived objects (SHAP explainer,
    feature order, AE scaler/threshold). One instance is created at FastAPI
    startup and reused across all requests - safe because inference calls
    below do not mutate any shared state.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lgbm_model: Optional[lgb.Booster] = None
        self.lgbm_feature_order: list[str] = []
        self.shap_explainer: Optional[shap.TreeExplainer] = None
        self.ae_model: Optional[Autoencoder] = None
        self.ae_mean: Optional[np.ndarray] = None
        self.ae_std: Optional[np.ndarray] = None
        self.ae_p95: float = 0.0
        self.ae_p99_5: float = 1.0
        self.ae_feat_mean: Optional[np.ndarray] = None
        self.ae_feat_std: Optional[np.ndarray] = None
        self.ae_feature_order: list[str] = []
        self._loaded = False

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def load(self) -> None:
        t0 = time.perf_counter()
        self._load_lgbm()
        self._load_autoencoder()
        self._loaded = True
        log.info("All models loaded in %.2fs", time.perf_counter() - t0)

    # --- LightGBM -----------------------------------------------------------

    def _load_lgbm(self) -> None:
        path = _resolve_model_path(self.settings.lgbm_model_path)
        if not path.exists():
            raise FileNotFoundError(f"LightGBM model not found at {path}")

        # The pkl file is a checkpoint dict:
        # {"model": Booster, "feature_cols": [...], "best_params": {...}, "metrics": {...}}
        lgbm_checkpoint = joblib.load(path)

        if isinstance(lgbm_checkpoint, dict) and "model" in lgbm_checkpoint:
            raw_model = lgbm_checkpoint["model"]
            self.lgbm_feature_order = lgbm_checkpoint.get("feature_cols", [])
            metrics = lgbm_checkpoint.get("metrics", {})

            # CRITICAL FIX FOR WINDOWS:
            # joblib unpickling can corrupt the underlying C++ pointer in the Booster.
            # We force a clean rebuild of the C++ structures by dumping to string
            # and reloading - this guarantees a fresh, non-corrupted Booster.
            model_str = raw_model.model_to_string()
            self.lgbm_model = lgb.Booster(model_str=model_str)

            log.info(
                "Loaded LightGBM Booster (%d features, ROC-AUC=%.4f, PR-AUC=%.4f) from %s",
                len(self.lgbm_feature_order),
                metrics.get("roc_auc", 0),
                metrics.get("pr_auc", 0),
                path,
            )
        else:
            # Fallback: raw model object
            self.lgbm_model = lgbm_checkpoint
            self.lgbm_feature_order = (
                self.lgbm_model.feature_name()
                if hasattr(self.lgbm_model, "feature_name")
                else []
            )
            log.warning("LightGBM loaded as raw object (not a checkpoint dict) from %s", path)

        # Cache the SHAP TreeExplainer at startup - building it takes ~8s
        # for 4000 trees. Doing it per request would destroy the 200ms SLA.
        log.info("Building SHAP TreeExplainer (may take a few seconds)...")
        self.shap_explainer = shap.TreeExplainer(
            self.lgbm_model,
            feature_perturbation="tree_path_dependent",
        )
        log.info("SHAP TreeExplainer initialized for LightGBM model")

    # --- Autoencoder ----------------------------------------------------------

    def _load_autoencoder(self) -> None:
        path = _resolve_model_path(self.settings.autoencoder_model_path)
        if not path.exists():
            raise FileNotFoundError(f"Autoencoder model not found at {path}")

        checkpoint = _load_torch_checkpoint(path)
        is_dict = isinstance(checkpoint, dict)
        state_dict = (
            checkpoint["model_state_dict"]
            if is_dict and "model_state_dict" in checkpoint
            else checkpoint
        )
        input_dim = checkpoint.get("input_dim", 21) if is_dict else 21
        bottleneck_dim = checkpoint.get("bottleneck_dim", 12) if is_dict else 12
        hidden_dim = checkpoint.get("hidden_dim", 128) if is_dict else 128
        dropout_rate = checkpoint.get("dropout_rate", 0.15) if is_dict else 0.15

        model = Autoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            dropout_rate=dropout_rate,
        )
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Autoencoder checkpoint at {path} does not match the Autoencoder "
                f"architecture (input_dim={input_dim}, bottleneck_dim={bottleneck_dim}). "
                f"Underlying error: {exc}"
            ) from exc
        model.eval()
        self.ae_model = model

        # Feature order from checkpoint (falls back to the scaler file below)
        self.ae_feature_order = checkpoint.get("feature_names", []) if is_dict else []
        log.info(
            "Loaded Autoencoder (input_dim=%s, bottleneck=%s) from %s",
            input_dim, bottleneck_dim, path,
        )

        # Scaler stats
        scaler_path = _resolve_model_path(self.settings.autoencoder_scaler_path)
        if scaler_path.exists():
            with open(scaler_path) as f:
                scaler = json.load(f)
            self.ae_mean = np.array(scaler["mean"], dtype=np.float32)
            self.ae_std = np.array(scaler["std"], dtype=np.float32)
            self.ae_std[self.ae_std < 1e-8] = 1.0
            if not self.ae_feature_order:
                self.ae_feature_order = scaler.get("feature_order", [])
            log.info("AE scaler loaded (%d features)", len(self.ae_mean))
        else:
            log.warning(
                "%s not found; Autoencoder will run WITHOUT input standardization.",
                scaler_path,
            )
            self.ae_mean = np.zeros(input_dim, dtype=np.float32)
            self.ae_std = np.ones(input_dim, dtype=np.float32)

        # Threshold calibration
        threshold_path = _resolve_model_path(self.settings.autoencoder_threshold_path)
        if threshold_path.exists():
            with open(threshold_path) as f:
                thresh = json.load(f)
            self.ae_p95 = float(thresh.get("p95", 0.0))
            self.ae_p99_5 = float(thresh.get("p99_5", thresh.get("p99", 1.0)))
            log.info("AE thresholds: p95=%.4f, p99.5=%.4f", self.ae_p95, self.ae_p99_5)
        else:
            log.warning(
                "%s not found; Autoencoder anomaly score will not be calibrated.",
                threshold_path,
            )
            self.ae_p95, self.ae_p99_5 = 0.0, 1.0

        # --------------------------------------------------------------- #
        # Feature error norm calibration (V3)
        #
        # ROOT-CAUSE FIX: earlier this JSON's `val_squared_error_mean` /
        # `val_squared_error_std` arrays were consumed positionally
        # (index i assumed to match self.ae_feature_order[i]) with NO check
        # that the JSON was stored in that order. The arrays were RANK-sorted
        # (by error magnitude), not in model-column order, so every z-score
        # compared a feature's error against a DIFFERENT feature's baseline.
        # This caused unpredictable explosions/suppressions in the top-5
        # attribution and in ae_score itself.
        #
        # Fix: require the JSON to carry its own `feature_order` (or
        # `feature_names`) list and reindex it to match self.ae_feature_order.
        # Fail loudly if the JSON declares no order or the feature sets differ:
        # silently-wrong scoring is worse than a startup crash.
        # --------------------------------------------------------------- #
        feature_norm_path = _resolve_model_path(
            getattr(
                self.settings,
                "autoencoder_feature_norm_path",
                "ml/models/Autoencoder/feature_error_norm.json",
            )
        )
        if not feature_norm_path.exists():
            feature_norm_path = _resolve_model_path("ml/models/Autoencoder/feature_error_norm.json")

        if not feature_norm_path.exists():
            # The p95 / p99.5 thresholds are calibrated on the z-score scale, so
            # running without these norms would silently mis-score every request.
            raise FileNotFoundError(
                f"Autoencoder feature-error norms not found at {feature_norm_path}. "
                f"Refusing to start: p95/p99.5 thresholds are calibrated on the "
                f"z-score scale and would be meaningless without them."
            )

        with open(feature_norm_path) as f:
            en = json.load(f)

        json_feature_order = en.get("feature_order") or en.get("feature_names")
        raw_mean = np.array(en["val_squared_error_mean"], dtype=np.float64)
        raw_std = np.array(en["val_squared_error_std"], dtype=np.float64)

        if not json_feature_order:
            raise ValueError(
                f"{feature_norm_path} has no 'feature_order' / 'feature_names' key. "
                f"Refusing to load feature-error norms positionally - a prior order "
                f"mismatch silently corrupted every AE z-score and top-5 attribution. "
                f"Regenerate the JSON with an explicit feature_order matching the "
                f"model's training column order."
            )

        if set(json_feature_order) != set(self.ae_feature_order):
            missing = set(self.ae_feature_order) - set(json_feature_order)
            extra = set(json_feature_order) - set(self.ae_feature_order)
            raise ValueError(
                f"{feature_norm_path} feature set does not match ae_feature_order. "
                f"Missing from JSON: {missing}. Unexpected in JSON: {extra}."
            )

        if json_feature_order != self.ae_feature_order:
            # Reindex JSON arrays into self.ae_feature_order column order.
            pos = {name: i for i, name in enumerate(json_feature_order)}
            reindex = [pos[name] for name in self.ae_feature_order]
            raw_mean = raw_mean[reindex]
            raw_std = raw_std[reindex]
            log.warning(
                "%s feature order differed from ae_feature_order - reindexed. "
                "Regenerate the JSON in canonical order to remove this dependency.",
                feature_norm_path,
            )

        self.ae_feat_mean = raw_mean

        # Gentle std floor: clamps only near-zero-variance features, without
        # squashing the natural spread of z-scores across the rest of the model.
        median_std = float(np.median(raw_std))
        safe_floor = max(1e-6, 0.05 * median_std)
        self.ae_feat_std = np.maximum(raw_std, safe_floor)

        pct_clipped = float((raw_std < safe_floor).mean() * 100)
        log.info(
            "AE feature norms loaded (%d features, median_std=%.6f, safe_floor=%.6f, %.1f%% clipped)",
            len(self.ae_feat_mean), median_std, safe_floor, pct_clipped,
        )


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def score_transaction(
    registry: ModelRegistry,
    settings: Settings,
    transaction_id: str,
    lgbm_features: dict[str, float],
    ae_features: dict[str, float],
    feature_source: str,
) -> ScoreResponse:
    """Pure scoring function - no HTTP dependency. Takes features and returns
    a ScoreResponse. Easily testable without spinning up a server."""
    if not registry.is_ready:
        raise RuntimeError("Models are not loaded yet")

    t0 = time.perf_counter()

    lgbm_score, shap_top5 = _score_lgbm(registry, lgbm_features, settings.shap_top_k)
    # The AE has its own top-k setting so it is not coupled to SHAP's.
    ae_top_k = getattr(settings, "ae_top_k", DEFAULT_TOP_K)
    
    # Cold-start guard: new/unseen accounts have no velocity history in Redis.
    # Read from lgbm_features since these might not be explicitly passed in ae_features.
    is_cold = (
        lgbm_features.get("txn_count_1h", 0.0) == 0.0 
        and lgbm_features.get("fan_out_24h", 0.0) == 0.0
    )
    
    ae_score, ae_recon_error, ae_top5 = _score_autoencoder(registry, ae_features, top_k=ae_top_k, is_cold=is_cold)

    # Preliminary weighted decision (LightGBM + AE only).
    # The authoritative fusion (0.50*TGN + 0.30*LightGBM + 0.15*AE + 0.05*Device)
    # happens in the Sprint 6 scoring-orchestrator.
    preliminary_score = 0.7 * lgbm_score + 0.3 * ae_score
    if preliminary_score >= settings.block_threshold:
        decision = DecisionPreliminary.BLOCK
    elif preliminary_score >= settings.review_threshold:
        decision = DecisionPreliminary.REVIEW
    else:
        decision = DecisionPreliminary.APPROVE

    latency_ms = (time.perf_counter() - t0) * 1000
    if latency_ms > settings.max_inference_latency_ms:
        log.warning(
            "Inference for %s took %.1fms, exceeding the %sms SLA target",
            transaction_id, latency_ms, settings.max_inference_latency_ms,
        )

    return ScoreResponse(
        transaction_id=transaction_id,
        lgbm_score=lgbm_score,
        ae_score=ae_score,
        ae_reconstruction_error=ae_recon_error,
        shap_top_5=shap_top5,
        ae_top_5=ae_top5,
        decision_preliminary=decision,
        feature_source=FeatureSource(feature_source),
        model_versions={
            "lightgbm": f"{Path(settings.lgbm_model_path).parent.name}/{Path(settings.lgbm_model_path).name}",
            "autoencoder": Path(settings.autoencoder_model_path).name,
        },
        inference_latency_ms=round(latency_ms, 3),
    )


def _score_lgbm(
    registry: ModelRegistry,
    features: dict[str, float],
    top_k: int,
) -> tuple[float, list[ShapFeatureContribution]]:
    ordered_values = np.array(
        [[features.get(name, 0.0) for name in registry.lgbm_feature_order]],
        dtype=np.float64,
    )

    raw_score = registry.lgbm_model.predict(ordered_values)[0]
    lgbm_score = float(np.clip(raw_score, 0.0, 1.0))

    shap_values = registry.shap_explainer.shap_values(ordered_values)
    # LightGBM binary classifiers can return either a single array or a
    # [class0, class1] list depending on version/objective - normalize both.
    if isinstance(shap_values, list):
        shap_row = np.asarray(shap_values[-1])[0]
    else:
        shap_row = np.asarray(shap_values)[0]

    order = np.argsort(-np.abs(shap_row))[:top_k]
    top_features = [
        ShapFeatureContribution(
            feature=registry.lgbm_feature_order[i],
            value=float(ordered_values[0, i]),
            shap_contribution=float(shap_row[i]),
            direction=ShapDirection.RAISES_RISK if shap_row[i] > 0 else ShapDirection.LOWERS_RISK,
        )
        for i in order
    ]
    return lgbm_score, top_features


def _build_ae_input(registry: ModelRegistry, features: dict[str, float]) -> tuple[np.ndarray, int]:
    """Build the AE input vector in model column order.

    Missing features are imputed with the training mean (so they standardize
    to 0 = neutral) instead of raw 0.0, which would standardize to
    (0 - mean) / std and create a large false reconstruction error.
    Returns (values, number_of_missing_features).
    """
    import math
    if not registry.ae_feature_order or registry.ae_feature_order[0].startswith("ae_feature_"):
        n = len(registry.ae_mean)
        names = [f"ae_feature_{i}" for i in range(n)]
    else:
        names = registry.ae_feature_order

    values = np.empty(len(names), dtype=np.float32)
    missing = 0
    for i, name in enumerate(names):
        v = features.get(name)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            v = float(registry.ae_mean[i]) if i < len(registry.ae_mean) else 0.0
            missing += 1
        values[i] = v
    return values, missing


def _score_autoencoder(
    registry: ModelRegistry,
    features: dict[str, float],
    top_k: int = DEFAULT_TOP_K,
    is_cold: bool = False,
) -> tuple[float, float, list[AeFeatureContribution]]:
    """Run autoencoder inference.

    Returns (normalized_score, raw_reconstruction_error, ae_top_k_features).

    V3 normalization: computes feature-level z-scores of squared errors
    (mean/std aligned to ae_feature_order - see load-time validation in
    _load_autoencoder), averages them, and scales via p95/p99.5 bounds with a
    cold-start guard.

    ae_top_k_features is sorted by reconstruction_z_score descending, giving
    the same explainability interface as the SHAP top-K for LightGBM.
    Cold-start transactions return an empty list.
    """
    values, n_missing = _build_ae_input(registry, features)
    if n_missing:
        log.debug("AE: %d feature(s) missing, mean-imputed", n_missing)

    standardized = (values - registry.ae_mean) / registry.ae_std
    x = torch.from_numpy(standardized).float().unsqueeze(0)

    with torch.no_grad():
        reconstruction = registry.ae_model(x)
        sq_err = torch.square(reconstruction - x).numpy().astype(np.float64)
        raw_mse = float(np.mean(sq_err))

        # Feature-level z-score normalization (V3)
        if registry.ae_feat_mean is not None and registry.ae_feat_std is not None:
            z_scores = np.clip((sq_err[0] - registry.ae_feat_mean) / registry.ae_feat_std, 0.0, None)
            norm_score = float(np.mean(z_scores))
            err_vector = z_scores
        else:
            norm_score = raw_mse
            err_vector = sq_err[0]

    if is_cold:
        log.debug("AE cold-start guard: score forced to 0.0")
        return 0.0, raw_mse, []

    p95 = registry.ae_p95
    p99_5 = registry.ae_p99_5

    if norm_score <= p95:
        ae_score = 0.5 * (norm_score / max(p95, 1e-6))
    else:
        ae_score = 0.5 + 0.5 * min(1.0, (norm_score - p95) / max(p99_5 - p95, 1e-6))

    log.debug(
        "AE | raw_mse=%.6f | norm=%.6f | p95=%.6f | p99.5=%.6f | score=%.6f",
        raw_mse, norm_score, p95, p99_5, ae_score,
    )

    # Build top-K AE feature explanations, sorted by z-score descending
    feature_names = (
        registry.ae_feature_order
        if registry.ae_feature_order
        else [f"ae_feature_{i}" for i in range(len(err_vector))]
    )
    top_indices = np.argsort(-err_vector)[:top_k]

    ae_top_features: list[AeFeatureContribution] = []
    for idx in top_indices:
        fname = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        fval = float(values[idx]) if idx < len(values) else 0.0
        z_val = float(err_vector[idx])

        # Severity classification based on training-calibrated thresholds
        if z_val <= 0.05:
            direction = AeAnomalyDirection.NO_ANOMALY_CONTRIBUTION
        elif z_val < 1.0:
            direction = AeAnomalyDirection.SMALL_ANOMALY_CONTRIBUTION
        else:
            direction = AeAnomalyDirection.RAISES_ANOMALY

        ae_top_features.append(
            AeFeatureContribution(
                feature=fname,
                value=fval,
                reconstruction_z_score=round(z_val, 4),
                direction=direction,
            )
        )

    return ae_score, raw_mse, ae_top_features
