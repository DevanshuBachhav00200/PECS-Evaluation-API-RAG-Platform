"""Unit tests for API-002 Pydantic request models."""

import pytest
from pydantic import ValidationError

from src.api import EvaluationRequest, SubmissionRequest
from src.api.schemas import EvaluationRequest as DirectEvaluationRequest
from src.api.schemas import SubmissionRequest as DirectSubmissionRequest


def test_package_exports():
    """Verify request models can be imported from top-level src.api package."""
    assert SubmissionRequest is DirectSubmissionRequest
    assert EvaluationRequest is DirectEvaluationRequest


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
