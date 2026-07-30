"""
Transaction Ingest Service — Health Endpoint Tests
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Verifies the service health endpoint contract."""

    def test_health_returns_200(self):
        """Health endpoint must return HTTP 200."""
        assert client.get("/health").status_code == 200

    def test_health_status_is_ok(self):
        """Health response must report status: ok."""
        assert client.get("/health").json()["status"] == "ok"

    def test_health_identifies_service(self):
        """Health response must identify this service by name."""
        assert client.get("/health").json()["service"] == "transaction-ingest"

    def test_health_returns_json(self):
        """Health endpoint must return application/json content type."""
        assert "application/json" in client.get("/health").headers["content-type"]
