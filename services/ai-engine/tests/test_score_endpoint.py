"""
services/ai-engine/tests/test_score_endpoint.py
=================================================
Automated validation suite for Sprint 4C. Run with:

    cd services/ai-engine
    pytest tests/ -v

Requires real model artifacts at the configured paths — these tests
intentionally do not mock the models, since the point is to catch real
load-order / shape-mismatch / schema bugs.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "transaction_id": "TXN-TEST-0001",
    "from_account": "ACC-88213",
    "to_account": "ACC-44107",
    "amount": 9500.00,
    "timestamp": "2026-08-24T09:15:00Z",
    "payment_format": "ACH",
    "currency": "USD",
}


def test_health_reports_all_models_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["lgbm_model_loaded"] is True
    assert body["autoencoder_model_loaded"] is True
    assert body["shap_explainer_ready"] is True


def test_score_returns_five_shap_features_with_direction(client):
    r = client.post("/score", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert len(body["shap_top_5"]) == 5
    for feat in body["shap_top_5"]:
        assert feat["direction"] in ("RAISES_RISK", "LOWERS_RISK")
        assert isinstance(feat["shap_contribution"], float)


def test_score_response_fields_are_well_formed(client):
    r = client.post("/score", json=VALID_PAYLOAD)
    body = r.json()
    assert 0.0 <= body["lgbm_score"] <= 1.0
    assert 0.0 <= body["ae_score"] <= 1.0
    assert body["decision_preliminary"] in ("APPROVE", "REVIEW", "BLOCK")
    assert body["feature_source"] in ("redis", "on_the_fly")


def test_score_latency_within_sla(client):
    """Plan requirement: 'Endpoint responds in < 200ms for a single transaction'."""
    r = client.post("/score", json=VALID_PAYLOAD)
    body = r.json()
    assert body["inference_latency_ms"] < 200.0


def test_shap_contributions_sum_to_model_margin(client):
    """Verify SHAP additivity: contributions should sum to approximately
    the raw model margin (pre-sigmoid), which is the space SHAP additivity
    actually holds in for a binary LightGBM classifier.
    """
    from app.inference import ModelRegistry
    from app.config import settings

    registry = ModelRegistry(settings)
    registry.load()

    feats = {name: 0.3 for name in registry.lgbm_feature_order}
    ordered = np.array([[feats[n] for n in registry.lgbm_feature_order]])

    raw_margin = registry.lgbm_model.predict(ordered, raw_score=True)[0]
    shap_vals = registry.shap_explainer.shap_values(ordered)
    shap_row = np.asarray(shap_vals[-1] if isinstance(shap_vals, list) else shap_vals)[0]
    expected_value = registry.shap_explainer.expected_value
    expected_value = expected_value[-1] if isinstance(expected_value, (list, np.ndarray)) else expected_value

    assert abs(raw_margin - (expected_value + shap_row.sum())) < 1e-4


def test_invalid_payload_returns_422(client):
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["amount"] = -50.0  # amount must be > 0
    r = client.post("/score", json=bad_payload)
    assert r.status_code == 422


def test_missing_required_field_returns_422(client):
    bad_payload = dict(VALID_PAYLOAD)
    del bad_payload["from_account"]
    r = client.post("/score", json=bad_payload)
    assert r.status_code == 422


