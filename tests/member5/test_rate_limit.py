"""Unit tests for API-009 global in-memory rate limiting (30 req / 60 sec)."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api import app, global_rate_limiter
from src.api.rate_limit import SlidingWindowRateLimiter

client = TestClient(app)

TEST_HEADERS = {"x-api-key": "test-key-123"}
SAMPLE_PAYLOAD = {
    "submissions": [
        {"task": "Task text", "student_prompt": "Prompt text"}
    ]
}


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Reset rate limiter state and set valid server API_KEY before each test."""
    monkeypatch.setenv("API_KEY", "test-key-123")
    global_rate_limiter.reset()
    yield
    global_rate_limiter.reset()


@patch("src.api.main.evaluate_batch_safe")
def test_rate_limit_requests_within_limit_accepted(mock_engine):
    """Verify that 30 requests sent within window are all accepted (HTTP 200)."""
    mock_engine.return_value = {
        "evaluations": [],
        "overall_summary": {"strength": "", "weakness": "", "improvement": "", "source": "derived", "truncated": [], "criterion_averages": {}},
        "usage": {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "attempts": 1, "estimated_input_tokens": 0, "cost_usd": 0.0},
        "rubric_version": "v1",
        "rubric_tag": "tag",
        "model_version": "m",
        "meta": {"submissions": 1, "calibrated_submissions": 0, "weights": {}, "config": {}, "caching": {}},
    }

    for i in range(30):
        res = client.post("/evaluate", json=SAMPLE_PAYLOAD, headers=TEST_HEADERS)
        assert res.status_code == 200, f"Request {i+1} failed with status {res.status_code}"


@patch("src.api.main.evaluate_batch_safe")
def test_rate_limit_31st_request_rejected_429(mock_engine):
    """Verify that the 31st request within the rolling window returns HTTP 429."""
    mock_engine.return_value = {
        "evaluations": [],
        "overall_summary": {"strength": "", "weakness": "", "improvement": "", "source": "derived", "truncated": [], "criterion_averages": {}},
        "usage": {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "attempts": 1, "estimated_input_tokens": 0, "cost_usd": 0.0},
        "rubric_version": "v1",
        "rubric_tag": "tag",
        "model_version": "m",
        "meta": {"submissions": 1, "calibrated_submissions": 0, "weights": {}, "config": {}, "caching": {}},
    }

    # Send 30 requests
    for i in range(30):
        res = client.post("/evaluate", json=SAMPLE_PAYLOAD, headers=TEST_HEADERS)
        assert res.status_code == 200

    # 31st request must return HTTP 429
    res_31 = client.post("/evaluate", json=SAMPLE_PAYLOAD, headers=TEST_HEADERS)
    assert res_31.status_code == 429
    assert "Global rate limit exceeded" in res_31.json()["detail"]


def test_rate_limit_sliding_window_expiration():
    """Verify that requests older than 60 seconds expire from sliding window and allow new requests."""
    limiter = SlidingWindowRateLimiter(max_requests=30, window_seconds=60.0)

    t0 = 1000.0
    # Simulate 30 requests at t0
    for _ in range(30):
        limiter.check_rate_limit(current_time=t0)

    # 31st request at t0 must raise 429
    with pytest.raises(Exception) as exc_info:
        limiter.check_rate_limit(current_time=t0)
    assert getattr(exc_info.value, "status_code", None) == 429

    # At t0 + 60.5s, the initial 30 requests have expired
    t1 = t0 + 60.5
    limiter.check_rate_limit(current_time=t1)  # Should not raise!
    assert limiter.get_current_count(current_time=t1) == 1


def test_rate_limit_state_reset_isolation():
    """Verify that calling reset() clears state for test isolation."""
    limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=60.0)
    for _ in range(5):
        limiter.check_rate_limit()

    assert limiter.get_current_count() == 5
    limiter.reset()
    assert limiter.get_current_count() == 0

