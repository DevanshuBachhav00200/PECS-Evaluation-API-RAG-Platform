"""API integration tests for PECS Evaluation API (API-014)."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api import app, global_rate_limiter, request_counter

client = TestClient(app)

TEST_HEADERS = {"x-api-key": "integration-test-secret-key"}

VALID_SUBMISSION_PAYLOAD = {
    "submissions": [
        {
            "task": "Write a Python function to compute factorial of a number.",
            "student_prompt": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
            "domain": "CODE",
            "level": "strong",
        }
    ]
}

SUCCESSFUL_ENGINE_RESULT = {
    "evaluations": [
        {
            "submission_index": 1,
            "task": "Write a Python function to compute factorial of a number.",
            "scores": {"c1": 9.0, "c2": 8.0, "c3": 8.0, "c4": 9.0, "c5": 10.0, "c6": 7.0},
            "composite_score": 8.55,
            "skill_level": "strong",
            "ethics_gate": {
                "mode": "weighted_only",
                "triggered": False,
                "c5_score": 10.0,
                "threshold": 3.0,
                "capped_score": None,
                "pre_gate_composite": 8.55,
                "policy_confirmation": "pending_alliance_compliance_confirmation",
            },
            "justifications": {
                "c1": "Clear task scope",
                "c2": "Correct implementation",
                "c3": "Sufficient context",
                "c4": "Clean formatting",
                "c5": "Ethically sound",
                "c6": "Reasonable complexity",
            },
            "justification_truncated": [],
            "prompt_truncated": False,
            "calibration_used": True,
            "calibration_record_id": "rec-factorial-001",
            "domain": "CODE",
            "level": "strong",
            "confidence": None,
            "confidence_label": None,
            "confidence_status": "not_implemented_in_mvp",
            "model_version": "claude-haiku-4-5-20251001",
            "rubric_version": "pecs-rubric-v1.0",
            "rubric_tag": "pecs-rubric-v1.0+abcdef123456",
        }
    ],
    "overall_summary": {
        "strength": "Clean recursive implementation.",
        "weakness": "Lacks stack overflow handling for large n.",
        "improvement": "Add iterative alternative for large inputs.",
        "source": "evaluator",
        "truncated": [],
        "criterion_averages": {"c1": 9.0, "c2": 8.0, "c3": 8.0, "c4": 9.0, "c5": 10.0, "c6": 7.0},
    },
    "usage": {
        "input_tokens": 1100,
        "output_tokens": 250,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "attempts": 1,
        "estimated_input_tokens": 850,
        "cost_usd": 0.00235,
    },
    "rubric_version": "pecs-rubric-v1.0",
    "rubric_tag": "pecs-rubric-v1.0+abcdef123456",
    "model_version": "claude-haiku-4-5-20251001",
    "meta": {
        "submissions": 1,
        "calibrated_submissions": 1,
        "weights": {"c1": 0.15, "c2": 0.20, "c3": 0.20, "c4": 0.20, "c5": 0.15, "c6": 0.10},
        "config": {"PECS_EVALUATOR_MODEL": "claude-haiku-4-5-20251001"},
        "caching": {"requested": True, "prefix_tokens": 850, "expected_effective": False},
    },
}


@pytest.fixture(autouse=True)
def setup_api_test_env(monkeypatch):
    """Set server API_KEY and reset rate limiter and request counter."""
    monkeypatch.setenv("API_KEY", "integration-test-secret-key")
    global_rate_limiter.reset()
    request_counter.reset(start=1)
    yield
    global_rate_limiter.reset()
    request_counter.reset(start=1)


@patch("src.api.main.evaluate_batch_safe")
def test_evaluate_endpoint_valid_request_success(mock_engine):
    """Integration test: Valid POST /evaluate request returns HTTP 200 and schema-valid response."""
    mock_engine.return_value = SUCCESSFUL_ENGINE_RESULT

    response = client.post("/evaluate", json=VALID_SUBMISSION_PAYLOAD, headers=TEST_HEADERS)
    assert response.status_code == 200
    data = response.json()

    assert len(data["evaluations"]) == 1
    assert data["evaluations"][0]["composite_score"] == 8.55
    assert data["overall_summary"]["source"] == "evaluator"
    assert data["rubric_version"] == "pecs-rubric-v1.0"
    mock_engine.assert_called_once()


def test_evaluate_endpoint_missing_required_fields():
    """Integration test: Request missing 'task' or 'student_prompt' returns HTTP 422."""
    # Missing task
    res1 = client.post(
        "/evaluate",
        json={"submissions": [{"student_prompt": "prompt only"}]},
        headers=TEST_HEADERS,
    )
    assert res1.status_code == 422

    # Missing student_prompt
    res2 = client.post(
        "/evaluate",
        json={"submissions": [{"task": "task only"}]},
        headers=TEST_HEADERS,
    )
    assert res2.status_code == 422


def test_evaluate_endpoint_malformed_wrong_type_payload():
    """Integration test: Wrong data types or malformed JSON returns HTTP 422."""
    # Submissions as string
    res1 = client.post(
        "/evaluate",
        json={"submissions": "not-a-list"},
        headers=TEST_HEADERS,
    )
    assert res1.status_code == 422

    # Empty submissions list
    res2 = client.post(
        "/evaluate",
        json={"submissions": []},
        headers=TEST_HEADERS,
    )
    assert res2.status_code == 422

    # Malformed JSON body
    res3 = client.post(
        "/evaluate",
        content="raw invalid json {{{",
        headers={"Content-Type": "application/json", "x-api-key": "integration-test-secret-key"},
    )
    assert res3.status_code == 422


@patch("src.api.main.evaluate_batch_safe")
def test_evaluate_endpoint_engine_failure_mapping(mock_engine):
    """Integration test: Evaluator engine failures return mapped HTTP status codes."""
    # Anthropic Auth Error -> HTTP 502 Bad Gateway
    mock_engine.return_value = {
        "error": {
            "code": "RAG_LLM_AUTH_ERROR",
            "layer": "rag",
            "message": "Invalid Anthropic API key credential",
            "details": {},
            "retryable": False,
        }
    }
    res_auth = client.post("/evaluate", json=VALID_SUBMISSION_PAYLOAD, headers=TEST_HEADERS)
    assert res_auth.status_code == 502
    assert res_auth.json()["error"]["code"] == "RAG_LLM_AUTH_ERROR"

    # Rubric Unavailable -> HTTP 500 Internal Server Error
    mock_engine.return_value = {
        "error": {
            "code": "EVAL_RUBRIC_UNAVAILABLE",
            "layer": "evaluation",
            "message": "Rubric definition missing",
            "details": {},
            "retryable": False,
        }
    }
    res_rubric = client.post("/evaluate", json=VALID_SUBMISSION_PAYLOAD, headers=TEST_HEADERS)
    assert res_rubric.status_code == 500
    assert res_rubric.json()["error"]["code"] == "EVAL_RUBRIC_UNAVAILABLE"


@patch("src.api.main.evaluate_batch_safe")
def test_evaluate_endpoint_empty_retrieval_scenario(mock_engine):
    """Integration test: Empty retrieval/evaluation scenario returns structured result or error."""
    # When retrieval fails due to database issue -> HTTP 500 structured error
    mock_engine.return_value = {
        "error": {
            "code": "RAG_RETRIEVAL_FAILED",
            "layer": "rag",
            "message": "No vectorstore collection matches found",
            "details": {"query_domain": "UNMATCHED"},
            "retryable": False,
        }
    }
    response = client.post("/evaluate", json=VALID_SUBMISSION_PAYLOAD, headers=TEST_HEADERS)
    assert response.status_code == 500
    data = response.json()
    assert data["error"]["code"] == "RAG_RETRIEVAL_FAILED"
    assert data["error"]["layer"] == "rag"
