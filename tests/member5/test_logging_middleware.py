"""Unit tests for API-011 request/response logging middleware."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api import LOG_FILE_PATH, app, global_rate_limiter, request_counter

client = TestClient(app)

TEST_HEADERS = {"x-api-key": "test-secret-key"}
SAMPLE_PAYLOAD = {
    "submissions": [
        {"task": "Task description", "student_prompt": "Prompt text"}
    ]
}


@pytest.fixture(autouse=True)
def setup_logging_environment(monkeypatch):
    """Set server API_KEY, reset rate limiter, and reset request counter before each test."""
    monkeypatch.setenv("API_KEY", "test-secret-key")
    global_rate_limiter.reset()
    request_counter.reset(start=1)
    yield
    global_rate_limiter.reset()
    request_counter.reset(start=1)


def test_logging_middleware_auto_incrementing_id():
    """Verify each request receives a sequential auto-incrementing log ID in X-Log-ID header."""
    res1 = client.get("/health")
    assert res1.status_code == 200
    assert res1.headers["X-Log-ID"] == "1"

    res2 = client.get("/health")
    assert res2.status_code == 200
    assert res2.headers["X-Log-ID"] == "2"

    res3 = client.get("/health")
    assert res3.status_code == 200
    assert res3.headers["X-Log-ID"] == "3"


@patch("src.api.main.evaluate_batch_safe")
def test_logging_middleware_writes_to_dedicated_log_file(mock_engine):
    """Verify each request writes exactly one JSON-formatted log entry to logs/api_requests.log."""
    mock_engine.return_value = {
        "evaluations": [],
        "overall_summary": {"strength": "", "weakness": "", "improvement": "", "source": "derived", "truncated": [], "criterion_averages": {}},
        "usage": {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "attempts": 1, "estimated_input_tokens": 0, "cost_usd": 0.0},
        "rubric_version": "v1",
        "rubric_tag": "tag",
        "model_version": "m",
        "meta": {"submissions": 1, "calibrated_submissions": 0, "weights": {}, "config": {}, "caching": {}},
    }

    # Ensure log file exists or read current line count
    initial_lines = 0
    if LOG_FILE_PATH.exists():
        initial_lines = len(LOG_FILE_PATH.read_text(encoding="utf-8").strip().splitlines())

    res = client.post("/evaluate", json=SAMPLE_PAYLOAD, headers=TEST_HEADERS)
    assert res.status_code == 200

    assert LOG_FILE_PATH.exists()
    lines = LOG_FILE_PATH.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == initial_lines + 1, "Exactly one log line must be produced per request"

    last_line = lines[-1]
    assert "log_id" in last_line
    assert '"path": "/evaluate"' in last_line
    assert '"status_code": 200' in last_line


def test_logging_middleware_counter_reset():
    """Verify calling request_counter.reset() resets log_id sequence to specified start value."""
    res1 = client.get("/health")
    assert res1.headers["X-Log-ID"] == "1"

    request_counter.reset(start=100)
    res2 = client.get("/health")
    assert res2.headers["X-Log-ID"] == "100"
