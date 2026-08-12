"""Unit tests for API-005 Evaluation Endpoint (POST /evaluate)."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)

# Sample valid Member 4 engine success output payload for 1 submission
SUCCESSFUL_ENGINE_OUTPUT_SINGLE = {
    "evaluations": [
        {
            "submission_index": 1,
            "task": "Write a Python sorting function",
            "scores": {"c1": 8.0, "c2": 7.0, "c3": 6.0, "c4": 7.0, "c5": 9.0, "c6": 5.0},
            "composite_score": 7.05,
            "skill_level": "strong",
            "ethics_gate": {
                "mode": "weighted_only",
                "triggered": False,
                "c5_score": 9.0,
                "threshold": 3.0,
                "capped_score": None,
                "pre_gate_composite": 7.05,
                "policy_confirmation": "pending_alliance_compliance_confirmation",
            },
            "justifications": {
                "c1": "Clear task definition",
                "c2": "Proper code structure",
                "c3": "Sufficient context",
                "c4": "Clean formatting",
                "c5": "Ethically compliant",
                "c6": "Reasonable complexity",
            },
            "justification_truncated": [],
            "prompt_truncated": False,
            "calibration_used": True,
            "calibration_record_id": "rec-001",
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
        "strength": "High structural clarity.",
        "weakness": "Minor gaps in efficiency.",
        "improvement": "Add performance benchmarks.",
        "source": "evaluator",
        "truncated": [],
        "criterion_averages": {"c1": 8.0, "c2": 7.0, "c3": 6.0, "c4": 7.0, "c5": 9.0, "c6": 5.0},
    },
    "usage": {
        "input_tokens": 1200,
        "output_tokens": 300,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "attempts": 1,
        "estimated_input_tokens": 900,
        "cost_usd": 0.0027,
    },
    "rubric_version": "pecs-rubric-v1.0",
    "rubric_tag": "pecs-rubric-v1.0+abcdef123456",
    "model_version": "claude-haiku-4-5-20251001",
    "meta": {
        "submissions": 1,
        "calibrated_submissions": 1,
        "weights": {"c1": 0.15, "c2": 0.20, "c3": 0.20, "c4": 0.20, "c5": 0.15, "c6": 0.10},
        "config": {"PECS_EVALUATOR_MODEL": "claude-haiku-4-5-20251001"},
        "caching": {"requested": True, "prefix_tokens": 900, "expected_effective": False},
    },
}

# Sample valid Member 4 engine success output payload for 2 submissions
SUCCESSFUL_ENGINE_OUTPUT_BATCH = {
    "evaluations": [
        {**SUCCESSFUL_ENGINE_OUTPUT_SINGLE["evaluations"][0], "submission_index": 1},
        {
            **SUCCESSFUL_ENGINE_OUTPUT_SINGLE["evaluations"][0],
            "submission_index": 2,
            "task": "Task two",
        },
    ],
    "overall_summary": SUCCESSFUL_ENGINE_OUTPUT_SINGLE["overall_summary"],
    "usage": {**SUCCESSFUL_ENGINE_OUTPUT_SINGLE["usage"], "input_tokens": 2400},
    "rubric_version": "pecs-rubric-v1.0",
    "rubric_tag": "pecs-rubric-v1.0+abcdef123456",
    "model_version": "claude-haiku-4-5-20251001",
    "meta": {**SUCCESSFUL_ENGINE_OUTPUT_SINGLE["meta"], "submissions": 2},
}


@patch("src.api.main.evaluate_batch_safe")
def test_evaluate_successful_single_submission(mock_engine):
    """Verify POST /evaluate returns 200 OK and EvaluationResponse for single submission."""
    mock_engine.return_value = SUCCESSFUL_ENGINE_OUTPUT_SINGLE

    payload = {
        "submissions": [
            {
                "task": "Write a Python sorting function",
                "student_prompt": "def bubble_sort(arr): ...",
                "domain": "CODE",
                "level": "strong",
            }
        ]
    }

    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["evaluations"]) == 1
    assert data["evaluations"][0]["submission_index"] == 1
    assert data["evaluations"][0]["composite_score"] == 7.05
    assert data["rubric_version"] == "pecs-rubric-v1.0"
    mock_engine.assert_called_once_with(
        [
            {
                "task": "Write a Python sorting function",
                "student_prompt": "def bubble_sort(arr): ...",
                "domain": "CODE",
                "level": "strong",
            }
        ]
    )


@patch("src.api.main.evaluate_batch_safe")
def test_evaluate_successful_batch_submissions(mock_engine):
    """Verify POST /evaluate processes multiple submissions in a batch request."""
    mock_engine.return_value = SUCCESSFUL_ENGINE_OUTPUT_BATCH

    payload = {
        "submissions": [
            {"task": "Task 1", "student_prompt": "Prompt 1"},
            {"task": "Task 2", "student_prompt": "Prompt 2"},
        ]
    }

    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["evaluations"]) == 2
    assert data["meta"]["submissions"] == 2
    mock_engine.assert_called_once_with(
        [
            {"task": "Task 1", "student_prompt": "Prompt 1"},
            {"task": "Task 2", "student_prompt": "Prompt 2"},
        ]
    )


def test_evaluate_invalid_request_empty_submissions():
    """Verify POST /evaluate returns HTTP 422 for empty submission list."""
    payload = {"submissions": []}
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 422


def test_evaluate_invalid_request_blank_fields():
    """Verify POST /evaluate returns HTTP 422 when submission has blank whitespace fields."""
    payload = {"submissions": [{"task": "   ", "student_prompt": "Valid prompt"}]}
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 422


@patch("src.api.main.evaluate_batch_safe")
def test_evaluate_error_token_safety_abort(mock_engine):
    """Verify RAG_TOKEN_SAFETY_ABORT error maps to HTTP 413."""
    mock_engine.return_value = {
        "error": {
            "code": "RAG_TOKEN_SAFETY_ABORT",
            "layer": "rag",
            "message": "SAFETY ABORT: estimated input tokens exceed safety cap",
            "details": {"estimated_input_tokens": 5200, "safety_cap": 4000},
            "retryable": False,
        }
    }

    payload = {"submissions": [{"task": "Large task", "student_prompt": "Long prompt"}]}
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 413
    data = response.json()
    assert data["error"]["code"] == "RAG_TOKEN_SAFETY_ABORT"
    assert data["error"]["layer"] == "rag"


@patch("src.api.main.evaluate_batch_safe")
def test_evaluate_error_rate_limited(mock_engine):
    """Verify RAG_LLM_RATE_LIMITED error maps to HTTP 429."""
    mock_engine.return_value = {
        "error": {
            "code": "RAG_LLM_RATE_LIMITED",
            "layer": "rag",
            "message": "Rate limit exceeded on Anthropic API",
            "details": {},
            "retryable": True,
        }
    }

    payload = {"submissions": [{"task": "Task", "student_prompt": "Prompt"}]}
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 429
    assert response.json()["error"]["retryable"] is True


@patch("src.api.main.evaluate_batch_safe")
def test_evaluate_error_timeout(mock_engine):
    """Verify RAG_LLM_TIMEOUT error maps to HTTP 504."""
    mock_engine.return_value = {
        "error": {
            "code": "RAG_LLM_TIMEOUT",
            "layer": "rag",
            "message": "LLM call timed out",
            "details": {},
            "retryable": True,
        }
    }

    payload = {"submissions": [{"task": "Task", "student_prompt": "Prompt"}]}
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "RAG_LLM_TIMEOUT"


@patch("src.api.main.evaluate_batch_safe")
def test_evaluate_error_bad_gateway(mock_engine):
    """Verify LLM server/auth errors map to HTTP 502."""
    mock_engine.return_value = {
        "error": {
            "code": "RAG_LLM_AUTH_ERROR",
            "layer": "rag",
            "message": "Invalid API key credential",
            "details": {},
            "retryable": False,
        }
    }

    payload = {"submissions": [{"task": "Task", "student_prompt": "Prompt"}]}
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "RAG_LLM_AUTH_ERROR"


@patch("src.api.main.evaluate_batch_safe")
def test_evaluate_error_rubric_unavailable(mock_engine):
    """Verify EVAL_RUBRIC_UNAVAILABLE error maps to HTTP 500."""
    mock_engine.return_value = {
        "error": {
            "code": "EVAL_RUBRIC_UNAVAILABLE",
            "layer": "evaluation",
            "message": "Rubric definition file not found",
            "details": {},
            "retryable": False,
        }
    }

    payload = {"submissions": [{"task": "Task", "student_prompt": "Prompt"}]}
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "EVAL_RUBRIC_UNAVAILABLE"
