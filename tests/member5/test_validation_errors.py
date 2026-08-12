"""Unit tests for API-008 request validation error mapping (HTTP 422)."""

import pytest
from fastapi.testclient import TestClient

from src.api import app, global_rate_limiter

client = TestClient(app)

TEST_HEADERS = {"x-api-key": "test-secret-key"}


@pytest.fixture(autouse=True)
def setup_auth(monkeypatch):
    """Set server API_KEY environment variable and reset rate limiter for authentication."""
    monkeypatch.setenv("API_KEY", "test-secret-key")
    global_rate_limiter.reset()
    yield
    global_rate_limiter.reset()



def test_validation_missing_required_task():
    """Verify HTTP 422 when submission is missing required 'task' field."""
    payload = {
        "submissions": [
            {"student_prompt": "def sort(): pass"}
        ]
    }
    response = client.post("/evaluate", json=payload, headers=TEST_HEADERS)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    # Field location should point to the missing task field
    errors = data["detail"]
    assert any("task" in err.get("loc", []) for err in errors)


def test_validation_missing_required_student_prompt():
    """Verify HTTP 422 when submission is missing required 'student_prompt' field."""
    payload = {
        "submissions": [
            {"task": "Write a sorting function"}
        ]
    }
    response = client.post("/evaluate", json=payload, headers=TEST_HEADERS)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    errors = data["detail"]
    assert any("student_prompt" in err.get("loc", []) for err in errors)


def test_validation_wrong_field_type_submissions():
    """Verify HTTP 422 when 'submissions' field has wrong type (e.g. string or dict instead of list)."""
    # Submissions as string
    response1 = client.post("/evaluate", json={"submissions": "invalid_string"}, headers=TEST_HEADERS)
    assert response1.status_code == 422

    # Submissions as dict
    response2 = client.post("/evaluate", json={"submissions": {"task": "t", "student_prompt": "p"}}, headers=TEST_HEADERS)
    assert response2.status_code == 422


def test_validation_empty_submissions_list():
    """Verify HTTP 422 when 'submissions' list is empty (violating min_length=1)."""
    payload = {"submissions": []}
    response = client.post("/evaluate", json=payload, headers=TEST_HEADERS)
    assert response.status_code == 422
    data = response.json()
    errors = data["detail"]
    assert any("submissions" in err.get("loc", []) for err in errors)


def test_validation_blank_whitespace_submission_fields():
    """Verify HTTP 422 when task or student_prompt is empty or whitespace only."""
    # Whitespace task
    res1 = client.post("/evaluate", json={"submissions": [{"task": "   ", "student_prompt": "valid"}]}, headers=TEST_HEADERS)
    assert res1.status_code == 422
    assert "task" in str(res1.json()["detail"])

    # Empty student_prompt
    res2 = client.post("/evaluate", json={"submissions": [{"task": "valid", "student_prompt": ""}]}, headers=TEST_HEADERS)
    assert res2.status_code == 422
    assert "student_prompt" in str(res2.json()["detail"])


def test_validation_malformed_json_body():
    """Verify HTTP 422 when raw request body is malformed JSON."""
    response = client.post(
        "/evaluate",
        content="invalid json body {{{",
        headers={"Content-Type": "application/json", "x-api-key": "test-secret-key"},
    )
    assert response.status_code == 422
