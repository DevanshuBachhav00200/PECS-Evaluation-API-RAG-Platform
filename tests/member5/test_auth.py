"""Unit tests for API-007 x-api-key authentication dependency."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api import app, global_rate_limiter, verify_api_key

client = TestClient(app)

SAMPLE_EVAL_PAYLOAD = {
    "submissions": [
        {"task": "Task description", "student_prompt": "Prompt text"}
    ]
}


@pytest.fixture(autouse=True)
def reset_rate_limit():
    """Reset rate limiter before and after each auth test."""
    global_rate_limiter.reset()
    yield
    global_rate_limiter.reset()


def test_auth_valid_api_key(monkeypatch):
    """Verify that a valid x-api-key matching server API_KEY allows request to proceed."""
    monkeypatch.setenv("API_KEY", "secret-test-key-123")
    
    with patch("src.api.main.evaluate_batch_safe") as mock_engine:
        mock_engine.return_value = {
            "evaluations": [],
            "overall_summary": {"strength": "", "weakness": "", "improvement": "", "source": "derived", "truncated": [], "criterion_averages": {}},
            "usage": {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "attempts": 1, "estimated_input_tokens": 0, "cost_usd": 0.0},
            "rubric_version": "pecs-rubric-v1.0",
            "rubric_tag": "tag",
            "model_version": "model",
            "meta": {"submissions": 1, "calibrated_submissions": 0, "weights": {}, "config": {}, "caching": {}},
        }
        
        response = client.post(
            "/evaluate",
            json=SAMPLE_EVAL_PAYLOAD,
            headers={"x-api-key": "secret-test-key-123"},
        )
        assert response.status_code == 200


def test_auth_missing_api_key_header(monkeypatch):
    """Verify that missing x-api-key header returns HTTP 403 Forbidden."""
    monkeypatch.setenv("API_KEY", "secret-test-key-123")
    
    response = client.post("/evaluate", json=SAMPLE_EVAL_PAYLOAD)
    assert response.status_code == 403
    assert response.json()["detail"] == "API key header is missing."


def test_auth_invalid_api_key_header(monkeypatch):
    """Verify that invalid x-api-key header returns HTTP 403 Forbidden."""
    monkeypatch.setenv("API_KEY", "secret-test-key-123")
    
    response = client.post(
        "/evaluate",
        json=SAMPLE_EVAL_PAYLOAD,
        headers={"x-api-key": "wrong-key-456"},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid API key."
    # Ensure expected secret key is not exposed in error response
    assert "secret-test-key-123" not in response.text


def test_auth_server_api_key_unset(monkeypatch):
    """Verify that unset API_KEY on server returns HTTP 500 Internal Server Error."""
    monkeypatch.delenv("API_KEY", raising=False)
    
    response = client.post(
        "/evaluate",
        json=SAMPLE_EVAL_PAYLOAD,
        headers={"x-api-key": "some-key"},
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "API key authentication is not configured on the server."


def test_health_endpoint_bypasses_auth(monkeypatch):
    """Verify GET /health remains unauthenticated and returns 200 OK without x-api-key."""
    monkeypatch.setenv("API_KEY", "secret-test-key-123")
    
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
