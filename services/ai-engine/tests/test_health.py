"""
AI Engine Service — Health Endpoint Tests
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Verifies the service health endpoint contract."""

    def test_health_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_health_status_is_ok(self):
        assert client.get("/health").json()["status"] == "ok"

    def test_health_identifies_service(self):
        assert client.get("/health").json()["service"] == "ai-engine"

    def test_health_returns_json(self):
        assert "application/json" in client.get("/health").headers["content-type"]

    def test_health_reports_models_loaded_field(self):
        """Health must expose models_loaded so monitoring can track readiness."""
        data = client.get("/health").json()
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"], list)
