"""Unit tests for API-004 Health Check endpoint (/health)."""

from fastapi.testclient import TestClient

from src.api import HealthCheckResponse, app
from src.api.schemas import HealthCheckResponse as DirectHealthCheckResponse


def test_health_check_endpoint():
    """Verify that GET /health returns 200 OK and valid health response payload."""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["embedder_loaded"] is True


def test_health_check_response_model_validation():
    """Verify HealthCheckResponse model validation and defaults."""
    res = HealthCheckResponse()
    assert res.status == "healthy"
    assert res.version == "1.0.0"
    assert res.embedder_loaded is True
    assert HealthCheckResponse is DirectHealthCheckResponse
