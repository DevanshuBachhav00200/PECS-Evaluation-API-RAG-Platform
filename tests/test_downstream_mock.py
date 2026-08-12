"""Integration test simulating Question Engine and Assessment Engine calls (INT-008)."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api import (
    EvaluationErrorResponse,
    EvaluationResponse,
    app,
    global_rate_limiter,
    request_counter,
)

client = TestClient(app)

TEST_HEADERS = {"x-api-key": "mock-downstream-secret-key"}

# ---------------------------------------------------------------------------
# Mock Downstream Payloads
# ---------------------------------------------------------------------------

QUESTION_ENGINE_REQUEST = {
    "submissions": [
        {
            "task": "Write a Python function to check if a string is a palindrome.",
            "student_prompt": "def is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c.isalnum())\n    return cleaned == cleaned[::-1]",
            "domain": "CODE",
            "level": "strong",
        }
    ]
}

ASSESSMENT_ENGINE_REQUEST = {
    "submissions": [
        {
            "task": "Write a Python function to calculate the factorial of n.",
            "student_prompt": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
            "domain": "CODE",
            "level": "strong",
        },
        {
            "task": "Write a Python function to calculate the factorial of n.",
            "student_prompt": "def factorial(n):\n    res = 1\n    for i in range(1, n + 1):\n        res *= i\n    return res",
            "domain": "CODE",
            "level": "developing",
        },
    ]
}

QE_MOCK_ENGINE_RESPONSE = {
    "evaluations": [
        {
            "submission_index": 1,
            "task": "Write a Python function to check if a string is a palindrome.",
            "scores": {"c1": 9.5, "c2": 9.0, "c3": 8.5, "c4": 9.0, "c5": 10.0, "c6": 8.0},
            "composite_score": 9.00,
            "skill_level": "strong",
            "ethics_gate": {
                "mode": "weighted_only",
                "triggered": False,
                "c5_score": 10.0,
                "threshold": 3.0,
                "capped_score": None,
                "pre_gate_composite": 9.00,
                "policy_confirmation": "pending_alliance_compliance_confirmation",
            },
            "justifications": {
                "c1": "Clear task scope",
                "c2": "Correct logic",
                "c3": "Sufficient context",
                "c4": "Clean format",
                "c5": "Ethically sound",
                "c6": "Good complexity",
            },
            "justification_truncated": [],
            "prompt_truncated": False,
            "calibration_used": True,
            "calibration_record_id": "rec-palindrome-001",
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
        "strength": "Clean alphanumeric filtering and string comparison.",
        "weakness": "Does not handle large inputs in memory efficiently.",
        "improvement": "Consider two-pointer approach.",
        "source": "evaluator",
        "truncated": [],
        "criterion_averages": {"c1": 9.5, "c2": 9.0, "c3": 8.5, "c4": 9.0, "c5": 10.0, "c6": 8.0},
    },
    "usage": {
        "input_tokens": 1050,
        "output_tokens": 230,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "attempts": 1,
        "estimated_input_tokens": 800,
        "cost_usd": 0.00215,
    },
    "rubric_version": "pecs-rubric-v1.0",
    "rubric_tag": "pecs-rubric-v1.0+abcdef123456",
    "model_version": "claude-haiku-4-5-20251001",
    "meta": {
        "submissions": 1,
        "calibrated_submissions": 1,
        "weights": {"c1": 0.15, "c2": 0.20, "c3": 0.20, "c4": 0.20, "c5": 0.15, "c6": 0.10},
        "config": {"PECS_EVALUATOR_MODEL": "claude-haiku-4-5-20251001"},
        "caching": {"requested": True, "prefix_tokens": 800, "expected_effective": False},
    },
}

AE_MOCK_ENGINE_RESPONSE = {
    "evaluations": [
        {
            "submission_index": 1,
            "task": "Write a Python function to calculate the factorial of n.",
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
                "c2": "Correct logic",
                "c3": "Sufficient context",
                "c4": "Clean format",
                "c5": "Ethically sound",
                "c6": "Good complexity",
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
        },
        {
            "submission_index": 2,
            "task": "Write a Python function to calculate the factorial of n.",
            "scores": {"c1": 8.0, "c2": 7.5, "c3": 7.0, "c4": 8.0, "c5": 10.0, "c6": 6.0},
            "composite_score": 7.75,
            "skill_level": "developing",
            "ethics_gate": {
                "mode": "weighted_only",
                "triggered": False,
                "c5_score": 10.0,
                "threshold": 3.0,
                "capped_score": None,
                "pre_gate_composite": 7.75,
                "policy_confirmation": "pending_alliance_compliance_confirmation",
            },
            "justifications": {
                "c1": "Clear task scope",
                "c2": "Iterative logic",
                "c3": "Adequate context",
                "c4": "Good format",
                "c5": "Ethically sound",
                "c6": "Moderate complexity",
            },
            "justification_truncated": [],
            "prompt_truncated": False,
            "calibration_used": True,
            "calibration_record_id": "rec-factorial-002",
            "domain": "CODE",
            "level": "developing",
            "confidence": None,
            "confidence_label": None,
            "confidence_status": "not_implemented_in_mvp",
            "model_version": "claude-haiku-4-5-20251001",
            "rubric_version": "pecs-rubric-v1.0",
            "rubric_tag": "pecs-rubric-v1.0+abcdef123456",
        },
    ],
    "overall_summary": {
        "strength": "Good understanding of loop structures and recursion.",
        "weakness": "Lacks boundary condition checks for negative inputs.",
        "improvement": "Add input validation for n < 0.",
        "source": "evaluator",
        "truncated": [],
        "criterion_averages": {"c1": 8.5, "c2": 7.75, "c3": 7.5, "c4": 8.5, "c5": 10.0, "c6": 6.5},
    },
    "usage": {
        "input_tokens": 1800,
        "output_tokens": 420,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "attempts": 1,
        "estimated_input_tokens": 1500,
        "cost_usd": 0.00410,
    },
    "rubric_version": "pecs-rubric-v1.0",
    "rubric_tag": "pecs-rubric-v1.0+abcdef123456",
    "model_version": "claude-haiku-4-5-20251001",
    "meta": {
        "submissions": 2,
        "calibrated_submissions": 2,
        "weights": {"c1": 0.15, "c2": 0.20, "c3": 0.20, "c4": 0.20, "c5": 0.15, "c6": 0.10},
        "config": {"PECS_EVALUATOR_MODEL": "claude-haiku-4-5-20251001"},
        "caching": {"requested": True, "prefix_tokens": 1500, "expected_effective": False},
    },
}


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set server API_KEY and reset rate limit counters before each test."""
    monkeypatch.setenv("API_KEY", "mock-downstream-secret-key")
    global_rate_limiter.reset()
    request_counter.reset(start=1)
    yield
    global_rate_limiter.reset()
    request_counter.reset(start=1)


@patch("src.api.main.evaluate_batch_safe")
def test_question_engine_realistic_request_success(mock_engine):
    """Test realistic Question Engine request -> /evaluate -> structured EvaluationResponse."""
    mock_engine.return_value = QE_MOCK_ENGINE_RESPONSE

    response = client.post("/evaluate", json=QUESTION_ENGINE_REQUEST, headers=TEST_HEADERS)
    assert response.status_code == 200

    # Validate returned payload strictly against EvaluationResponse Pydantic model
    validated_response = EvaluationResponse.model_validate(response.json())
    assert len(validated_response.evaluations) == 1
    assert validated_response.evaluations[0].composite_score == 9.00
    assert validated_response.evaluations[0].skill_level == "strong"
    mock_engine.assert_called_once()


@patch("src.api.main.evaluate_batch_safe")
def test_assessment_engine_realistic_batch_request_success(mock_engine):
    """Test realistic Assessment Engine batch request -> /evaluate -> structured EvaluationResponse."""
    mock_engine.return_value = AE_MOCK_ENGINE_RESPONSE

    response = client.post("/evaluate", json=ASSESSMENT_ENGINE_REQUEST, headers=TEST_HEADERS)
    assert response.status_code == 200

    # Validate returned payload strictly against EvaluationResponse Pydantic model
    validated_response = EvaluationResponse.model_validate(response.json())
    assert len(validated_response.evaluations) == 2
    assert validated_response.evaluations[0].composite_score == 8.55
    assert validated_response.evaluations[1].composite_score == 7.75
    assert validated_response.meta.submissions == 2
    mock_engine.assert_called_once()


@patch("src.api.main.evaluate_batch_safe")
def test_downstream_safety_abort_error_scenario(mock_engine):
    """Test downstream token safety abort error scenario -> HTTP 413 & EvaluationErrorResponse."""
    mock_engine.return_value = {
        "error": {
            "code": "RAG_TOKEN_SAFETY_ABORT",
            "layer": "rag",
            "message": "SAFETY ABORT: estimated input tokens (5200) exceed safety cap (4000)",
            "details": {"estimated_input_tokens": 5200, "safety_cap": 4000},
            "retryable": False,
        }
    }

    response = client.post("/evaluate", json=QUESTION_ENGINE_REQUEST, headers=TEST_HEADERS)
    assert response.status_code == 413

    # Validate returned error payload strictly against EvaluationErrorResponse Pydantic model
    validated_error = EvaluationErrorResponse.model_validate(response.json())
    assert validated_error.error.code == "RAG_TOKEN_SAFETY_ABORT"
    assert validated_error.error.layer == "rag"
    assert validated_error.error.retryable is False
