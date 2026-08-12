"""Unit tests for API-010 structured error-to-HTTP-code mapping."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api import app, global_rate_limiter

client = TestClient(app)

TEST_HEADERS = {"x-api-key": "test-key-123"}
SAMPLE_PAYLOAD = {
    "submissions": [
        {"task": "Sample Task", "student_prompt": "Sample Prompt"}
    ]
}


@pytest.fixture(autouse=True)
def setup_environment(monkeypatch):
    """Set server API_KEY and reset rate limiter before each test."""
    monkeypatch.setenv("API_KEY", "test-key-123")
    global_rate_limiter.reset()
    yield
    global_rate_limiter.reset()


@pytest.mark.parametrize(
    "error_code,expected_status",
    [
        # Safety abort -> 413
        ("RAG_TOKEN_SAFETY_ABORT", 413),
        # Rate limit -> 429
        ("RAG_LLM_RATE_LIMITED", 429),
        # Validation -> 422
        ("EVAL_INVALID_INPUT", 422),
        # Engine / internal error -> 500
        ("RAG_RETRIEVAL_FAILED", 500),
        ("EVAL_RUBRIC_UNAVAILABLE", 500),
        ("EVAL_ETHICS_GATE_UNCONFIRMED", 500),
        # Timeout -> 504
        ("RAG_LLM_TIMEOUT", 504),
        # Upstream LLM / evaluation errors -> 502
        ("RAG_LLM_AUTH_ERROR", 502),
        ("RAG_LLM_CONNECTION_ERROR", 502),
        ("RAG_LLM_BAD_REQUEST", 502),
        ("RAG_LLM_SERVER_ERROR", 502),
        ("RAG_LLM_REFUSAL", 502),
        ("RAG_LLM_EMPTY_RESPONSE", 502),
        ("RAG_LLM_UNKNOWN_ERROR", 502),
        ("RAG_RESPONSE_NOT_JSON", 502),
        ("RAG_RESPONSE_SCHEMA_INVALID", 502),
        ("RAG_RETRY_BUDGET_EXHAUSTED", 502),
        ("EVAL_SCORE_OUT_OF_RANGE", 502),
        ("EVAL_MISSING_CRITERION", 502),
        ("EVAL_SUBMISSION_COUNT_MISMATCH", 502),
    ],
)
@patch("src.api.main.evaluate_batch_safe")
def test_error_code_http_status_mapping(mock_engine, error_code, expected_status):
    """Verify each Member 4 error code maps to the correct HTTP status code."""
    mock_engine.return_value = {
        "error": {
            "code": error_code,
            "layer": "rag" if error_code.startswith("RAG_") else "evaluation",
            "message": f"Error occurred: {error_code}",
            "details": {"test_detail": "info"},
            "retryable": False,
        }
    }

    response = client.post("/evaluate", json=SAMPLE_PAYLOAD, headers=TEST_HEADERS)
    assert response.status_code == expected_status
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == error_code
    # Verify response body is structured JSON and never exposes raw traceback
    assert "Traceback (most recent call last)" not in response.text
    assert 'File "' not in response.text


@patch("src.api.main.evaluate_batch_safe")
def test_unmapped_unknown_error_code_fallback_500(mock_engine):
    """Verify unknown/unmapped engine error codes safely fall back to HTTP 500."""
    mock_engine.return_value = {
        "error": {
            "code": "UNKNOWN_FUTURE_ERROR_CODE",
            "layer": "unknown",
            "message": "An unexpected error occurred",
            "details": {},
            "retryable": False,
        }
    }

    response = client.post("/evaluate", json=SAMPLE_PAYLOAD, headers=TEST_HEADERS)
    assert response.status_code == 500
    data = response.json()
    assert data["error"]["code"] == "UNKNOWN_FUTURE_ERROR_CODE"
    assert "Traceback" not in response.text
