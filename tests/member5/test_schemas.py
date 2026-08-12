"""Unit tests for API-002 and API-003 Pydantic request and response models."""

import pytest
from pydantic import ValidationError

from src.api import (
    ErrorDetailResponse,
    EthicsGateResponse,
    EvaluationErrorResponse,
    EvaluationMetaResponse,
    EvaluationRequest,
    EvaluationResponse,
    OverallSummaryResponse,
    SubmissionEvaluationResponse,
    SubmissionRequest,
    UsageInfoResponse,
)
from src.api.schemas import (
    ErrorDetailResponse as DirectErrorDetailResponse,
    EthicsGateResponse as DirectEthicsGateResponse,
    EvaluationErrorResponse as DirectEvaluationErrorResponse,
    EvaluationMetaResponse as DirectEvaluationMetaResponse,
    EvaluationRequest as DirectEvaluationRequest,
    EvaluationResponse as DirectEvaluationResponse,
    OverallSummaryResponse as DirectOverallSummaryResponse,
    SubmissionEvaluationResponse as DirectSubmissionEvaluationResponse,
    SubmissionRequest as DirectSubmissionRequest,
    UsageInfoResponse as DirectUsageInfoResponse,
)


def test_package_exports():
    """Verify request and response models can be imported from top-level src.api package."""
    assert SubmissionRequest is DirectSubmissionRequest
    assert EvaluationRequest is DirectEvaluationRequest
    assert EthicsGateResponse is DirectEthicsGateResponse
    assert SubmissionEvaluationResponse is DirectSubmissionEvaluationResponse
    assert OverallSummaryResponse is DirectOverallSummaryResponse
    assert UsageInfoResponse is DirectUsageInfoResponse
    assert EvaluationMetaResponse is DirectEvaluationMetaResponse
    assert EvaluationResponse is DirectEvaluationResponse
    assert ErrorDetailResponse is DirectErrorDetailResponse
    assert EvaluationErrorResponse is DirectEvaluationErrorResponse


def test_valid_submission_request_minimal():
    """Verify creation of a valid SubmissionRequest with minimal required fields."""
    req = SubmissionRequest(
        task="Write a Python function",
        student_prompt="def add(a, b): return a + b",
    )
    assert req.task == "Write a Python function"
    assert req.student_prompt == "def add(a, b): return a + b"
    assert req.domain is None
    assert req.level is None
    assert req.to_qa_pair() == {
        "task": "Write a Python function",
        "student_prompt": "def add(a, b): return a + b",
    }


def test_valid_submission_request_full():
    """Verify creation of SubmissionRequest with optional domain and level fields."""
    req = SubmissionRequest(
        task="Write a binary search function",
        student_prompt="def binary_search(arr, target): ...",
        domain="CODE",
        level="strong",
    )
    assert req.domain == "CODE"
    assert req.level == "strong"
    assert req.to_qa_pair() == {
        "task": "Write a binary search function",
        "student_prompt": "def binary_search(arr, target): ...",
        "domain": "CODE",
        "level": "strong",
    }


def test_submission_request_whitespace_stripping():
    """Verify leading/trailing whitespace is stripped from string fields."""
    req = SubmissionRequest(
        task="   Task text with whitespace   ",
        student_prompt="\tStudent prompt text\n",
    )
    assert req.task == "Task text with whitespace"
    assert req.student_prompt == "Student prompt text"


def test_invalid_submission_request_blank():
    """Verify validation error when task or student_prompt is empty or whitespace only."""
    with pytest.raises(ValidationError) as exc_info:
        SubmissionRequest(task="   ", student_prompt="Valid prompt")
    assert "task" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        SubmissionRequest(task="Valid task", student_prompt="")
    assert "student_prompt" in str(exc_info.value)


def test_invalid_submission_request_missing_fields():
    """Verify validation error when required fields are missing."""
    with pytest.raises(ValidationError):
        SubmissionRequest(task="Only task")  # type: ignore[call-arg]


def test_valid_evaluation_request_batch():
    """Verify creation of a valid EvaluationRequest batch."""
    sub1 = SubmissionRequest(task="Task 1", student_prompt="Prompt 1")
    sub2 = SubmissionRequest(task="Task 2", student_prompt="Prompt 2", domain="CODE", level="developing")

    batch = EvaluationRequest(submissions=[sub1, sub2])
    assert len(batch.submissions) == 2

    qa_pairs = batch.to_qa_pairs()
    assert len(qa_pairs) == 2
    assert qa_pairs[0] == {"task": "Task 1", "student_prompt": "Prompt 1"}
    assert qa_pairs[1] == {
        "task": "Task 2",
        "student_prompt": "Prompt 2",
        "domain": "CODE",
        "level": "developing",
    }


def test_invalid_evaluation_request_empty_batch():
    """Verify validation error when submissions list is empty."""
    with pytest.raises(ValidationError) as exc_info:
        EvaluationRequest(submissions=[])
    assert "submissions" in str(exc_info.value)


# ---------------------------------------------------------------------------
# API-003 Response Model Tests
# ---------------------------------------------------------------------------


def test_ethics_gate_response_model():
    """Verify EthicsGateResponse model validation and defaults."""
    data = {
        "mode": "weighted_only",
        "triggered": False,
        "c5_score": 9.0,
        "threshold": 3.0,
        "capped_score": None,
        "pre_gate_composite": 7.05,
        "policy_confirmation": "pending_alliance_compliance_confirmation",
    }
    gate = EthicsGateResponse(**data)
    assert gate.mode == "weighted_only"
    assert gate.triggered is False
    assert gate.c5_score == 9.0
    assert gate.threshold == 3.0
    assert gate.capped_score is None
    assert gate.pre_gate_composite == 7.05
    assert gate.policy_confirmation == "pending_alliance_compliance_confirmation"


def test_submission_evaluation_response_model():
    """Verify SubmissionEvaluationResponse parses per-submission output from Member 4 engine."""
    data = {
        "submission_index": 1,
        "task": "Task description",
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
            "c1": "Justification 1",
            "c2": "Justification 2",
            "c3": "Justification 3",
            "c4": "Justification 4",
            "c5": "Justification 5",
            "c6": "Justification 6",
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
    sub_eval = SubmissionEvaluationResponse(**data)
    assert sub_eval.submission_index == 1
    assert sub_eval.composite_score == 7.05
    assert sub_eval.scores["c1"] == 8.0
    assert sub_eval.ethics_gate.mode == "weighted_only"
    assert sub_eval.confidence_status == "not_implemented_in_mvp"
    assert sub_eval.rubric_version == "pecs-rubric-v1.0"


def test_submission_evaluation_response_invalid_bounds():
    """Verify validation errors for out-of-bound values in SubmissionEvaluationResponse."""
    base_data = {
        "submission_index": 1,
        "task": "Task description",
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
        "justifications": {"c1": "J1", "c2": "J2", "c3": "J3", "c4": "J4", "c5": "J5", "c6": "J6"},
        "confidence_status": "not_implemented_in_mvp",
        "model_version": "claude-haiku-4-5-20251001",
        "rubric_version": "pecs-rubric-v1.0",
        "rubric_tag": "pecs-rubric-v1.0+tag",
    }

    # Invalid submission_index (< 1)
    with pytest.raises(ValidationError) as exc_info:
        SubmissionEvaluationResponse(**{**base_data, "submission_index": 0})
    assert "submission_index" in str(exc_info.value)

    # Invalid composite_score (> 10.0)
    with pytest.raises(ValidationError) as exc_info:
        SubmissionEvaluationResponse(**{**base_data, "composite_score": 11.5})
    assert "composite_score" in str(exc_info.value)


def test_overall_summary_response_model():
    """Verify OverallSummaryResponse model parsing."""
    summary_data = {
        "strength": "Good clarity across prompt structure.",
        "weakness": "Lacks explicit edge case constraints.",
        "improvement": "Add detailed input/output specifications.",
        "source": "evaluator",
        "truncated": [],
        "criterion_averages": {"c1": 8.0, "c2": 7.0, "c3": 6.0, "c4": 7.0, "c5": 9.0, "c6": 5.0},
    }
    summary = OverallSummaryResponse(**summary_data)
    assert summary.strength == "Good clarity across prompt structure."
    assert summary.source == "evaluator"
    assert summary.criterion_averages["c1"] == 8.0


def test_usage_info_response_model():
    """Verify UsageInfoResponse model parsing and defaults."""
    usage_data = {
        "input_tokens": 1200,
        "output_tokens": 300,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "attempts": 1,
        "estimated_input_tokens": 900,
        "cost_usd": 0.0027,
    }
    usage = UsageInfoResponse(**usage_data)
    assert usage.input_tokens == 1200
    assert usage.output_tokens == 300
    assert usage.cost_usd == 0.0027


def test_evaluation_meta_response_model():
    """Verify EvaluationMetaResponse model parsing."""
    meta_data = {
        "submissions": 1,
        "calibrated_submissions": 1,
        "weights": {"c1": 0.15, "c2": 0.20, "c3": 0.20, "c4": 0.20, "c5": 0.15, "c6": 0.10},
        "config": {"PECS_EVALUATOR_MODEL": "claude-haiku-4-5-20251001"},
        "caching": {"requested": True, "prefix_tokens": 900, "expected_effective": False},
    }
    meta = EvaluationMetaResponse(**meta_data)
    assert meta.submissions == 1
    assert meta.calibrated_submissions == 1
    assert meta.weights["c1"] == 0.15
    assert meta.config["PECS_EVALUATOR_MODEL"] == "claude-haiku-4-5-20251001"


def test_evaluation_response_full_mapping():
    """Verify EvaluationResponse parses a complete Member 4 evaluate_batch output dictionary."""
    full_output = {
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
                    "c1": "Clear objective",
                    "c2": "Good structure",
                    "c3": "Proper context",
                    "c4": "Adequate formatting",
                    "c5": "Ethically sound",
                    "c6": "Fair efficiency",
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

    resp = EvaluationResponse(**full_output)
    assert len(resp.evaluations) == 1
    assert resp.evaluations[0].task == "Write a Python sorting function"
    assert resp.evaluations[0].composite_score == 7.05
    assert resp.overall_summary.source == "evaluator"
    assert resp.usage.input_tokens == 1200
    assert resp.rubric_version == "pecs-rubric-v1.0"
    assert resp.meta.submissions == 1


def test_error_response_models():
    """Verify ErrorDetailResponse and EvaluationErrorResponse parse Member 4 error structures."""
    error_dict = {
        "error": {
            "code": "RAG_TOKEN_SAFETY_ABORT",
            "layer": "rag",
            "message": "SAFETY ABORT: estimated input tokens exceed safety cap",
            "details": {"estimated_input_tokens": 5200, "safety_cap": 4000},
            "retryable": False,
        }
    }

    err_resp = EvaluationErrorResponse(**error_dict)
    assert err_resp.error.code == "RAG_TOKEN_SAFETY_ABORT"
    assert err_resp.error.layer == "rag"
    assert err_resp.error.details["estimated_input_tokens"] == 5200
    assert err_resp.error.retryable is False

