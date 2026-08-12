"""Unit tests for API-013 OpenAPI documentation and schema verification."""

import pytest
from fastapi.testclient import TestClient

from src.api import app, global_rate_limiter, request_counter

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_environment(monkeypatch):
    """Set server API_KEY and reset rate limiter before each test."""
    monkeypatch.setenv("API_KEY", "test-key-123")
    global_rate_limiter.reset()
    request_counter.reset(start=1)
    yield
    global_rate_limiter.reset()
    request_counter.reset(start=1)


def test_swagger_ui_docs_endpoint_accessible():
    """Verify GET /docs returns HTTP 200 and renders HTML Swagger UI."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>PECS Evaluation API - Swagger UI</title>" in response.text or "swagger-ui" in response.text.lower()


def test_openapi_json_endpoint_accessible_and_valid():
    """Verify GET /openapi.json returns HTTP 200 and valid OpenAPI JSON specification."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()

    assert spec["info"]["title"] == "PECS Evaluation API"
    assert spec["info"]["version"] == "1.0.0"
    assert "paths" in spec
    assert "components" in spec
    assert "schemas" in spec["components"]


def test_openapi_health_endpoint_schema():
    """Verify /health endpoint details and response schema appear in OpenAPI spec."""
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]

    assert "/health" in paths
    health_op = paths["/health"]["get"]
    assert health_op["summary"] == "Health Check"
    assert "Health" in health_op["tags"]
    assert "200" in health_op["responses"]


def test_openapi_evaluate_endpoint_schema():
    """Verify /evaluate endpoint request, responses, and status codes appear in OpenAPI spec."""
    spec = client.get("/openapi.json").json()
    paths = spec["paths"]

    assert "/evaluate" in paths
    eval_op = paths["/evaluate"]["post"]
    assert eval_op["summary"] == "Evaluate Prompt Submissions"
    assert "Evaluation" in eval_op["tags"]

    # Verify documented HTTP response codes
    responses = eval_op["responses"]
    for code in ["200", "403", "413", "422", "429", "500", "502", "504"]:
        assert code in responses, f"Response code {code} missing from /evaluate OpenAPI schema"


def test_openapi_components_schemas():
    """Verify key Pydantic request and response schemas are registered in OpenAPI components."""
    spec = client.get("/openapi.json").json()
    schemas = spec["components"]["schemas"]

    required_schemas = [
        "EvaluationRequest",
        "SubmissionRequest",
        "EvaluationResponse",
        "SubmissionEvaluationResponse",
        "HealthCheckResponse",
        "EvaluationErrorResponse",
        "ErrorDetailResponse",
    ]

    for schema_name in required_schemas:
        assert schema_name in schemas, f"Schema {schema_name} missing from OpenAPI components"
