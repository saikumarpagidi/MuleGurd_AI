"""
Feature Service — Health Endpoint Tests
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
        assert client.get("/health").json()["service"] == "feature-service"

    def test_health_returns_json(self):
        assert "application/json" in client.get("/health").headers["content-type"]

    def test_health_reports_feast_connectivity_field(self):
        """Health must expose feast_connected for feature store monitoring."""
        data = client.get("/health").json()
        assert "feast_connected" in data
        assert isinstance(data["feast_connected"], bool)

    def test_health_reports_feature_count_field(self):
        """Health must expose feature_count so monitoring can detect registration failures."""
        data = client.get("/health").json()
        assert "feature_count" in data
        assert isinstance(data["feature_count"], int)
