"""
Graph Service — Health Endpoint Tests
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
        assert client.get("/health").json()["service"] == "graph-service"

    def test_health_returns_json(self):
        assert "application/json" in client.get("/health").headers["content-type"]

    def test_health_reports_neo4j_connectivity_field(self):
        """Health must expose neo4j_connected for infrastructure monitoring."""
        data = client.get("/health").json()
        assert "neo4j_connected" in data
        assert isinstance(data["neo4j_connected"], bool)

    def test_health_reports_tgn_model_field(self):
        """Health must expose tgn_model_loaded for model readiness checks."""
        data = client.get("/health").json()
        assert "tgn_model_loaded" in data
        assert isinstance(data["tgn_model_loaded"], bool)
