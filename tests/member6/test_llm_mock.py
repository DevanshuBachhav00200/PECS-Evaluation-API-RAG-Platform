"""
TEST-006 — Mock-LLM-response tests.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Exercises JSON parsing (RAG-015), fence stripping, schema validation (RAG-016)
and retry logic (RAG-017) with mocked responses.

Acceptance (TEST-006): the suite runs offline with zero real Anthropic API
calls. Enforced, not just intended — ``no_live_api`` below monkeypatches
``llm_call.get_client`` to raise, so any code path that tries to construct a
real client fails the test loudly instead of quietly reaching for a credential
that might exist in a developer's environment. A test that would make a paid
call on someone's laptop and skip on CI is worse than no test.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from m4_fakes import (
    FakeClient,
    FakeResponse,
    FakeUsage,
    evaluator_payload,
    evaluator_response,
)
from src.evaluation.error_types import PecsEvaluationError, RagError
from src.rag import llm_call
from src.rag.build_request import build_request
from src.rag.llm_call import call_llm, extract_text, usage_to_dict
from src.rag.parse_response import parse_llm_response
from src.rag.retry_handler import call_and_validate
from src.rag.validate_llm_output import validate_llm_output

BATCH_CONTENT = (
    "Evaluate the following 1 submission(s).\n\n"
    "### Submission 1\nTask: Review an endpoint.\n"
    "Candidate prompt: List issues by severity.\nCalibration: none"
)


@pytest.fixture(autouse=True)
def no_live_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any attempt to build a real Anthropic client fail the test."""

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "TEST-006 must not construct a live Anthropic client — inject a "
            "FakeClient instead"
        )

    monkeypatch.setattr(llm_call, "get_client", _forbidden)
    monkeypatch.setattr(llm_call, "_client", None)


def _request() -> Dict[str, Any]:
    return build_request(BATCH_CONTENT)


# ---------------------------------------------------------------------------
# RAG-015 — response parsing and fence stripping
# ---------------------------------------------------------------------------


def test_plain_json_response_parses() -> None:
    payload = evaluator_payload(count=1)
    assert parse_llm_response(json.dumps(payload)) == payload


def test_fenced_json_response_still_parses() -> None:
    """```json fences are a routine model habit, not an error."""
    payload = evaluator_payload(count=1)
    fenced = f"```json\n{json.dumps(payload)}\n```"
    assert parse_llm_response(fenced) == payload


def test_unlabelled_fence_still_parses() -> None:
    payload = evaluator_payload(count=1)
    assert parse_llm_response(f"```\n{json.dumps(payload)}\n```") == payload


def test_surrounding_whitespace_is_tolerated() -> None:
    payload = evaluator_payload(count=1)
    assert parse_llm_response(f"\n\n  {json.dumps(payload)}  \n") == payload


def test_non_json_response_raises_a_structured_error() -> None:
    with pytest.raises(PecsEvaluationError) as excinfo:
        parse_llm_response("I'm sorry, I can't help with that.")
    assert excinfo.value.to_dict()["error"]["code"] == "RAG_RESPONSE_NOT_JSON"


def test_empty_response_raises_a_structured_error() -> None:
    with pytest.raises(PecsEvaluationError):
        parse_llm_response("")


# ---------------------------------------------------------------------------
# RAG-016 — schema validation
# ---------------------------------------------------------------------------


def test_well_formed_payload_validates() -> None:
    validated = validate_llm_output(evaluator_payload(count=2), expected_count=2)
    assert len(validated["evaluations"]) == 2


def test_missing_criterion_is_rejected_with_a_clear_error() -> None:
    payload = evaluator_payload(count=1, drop_criterion="c5")
    with pytest.raises(PecsEvaluationError) as excinfo:
        validate_llm_output(payload, expected_count=1)

    error = excinfo.value.to_dict()["error"]
    assert error["code"] == "RAG_RESPONSE_SCHEMA_INVALID"
    assert "c5" in json.dumps(error)


def test_submission_count_mismatch_is_rejected() -> None:
    with pytest.raises(PecsEvaluationError):
        validate_llm_output(evaluator_payload(count=1), expected_count=3)


def test_payload_without_evaluations_key_is_rejected() -> None:
    with pytest.raises(PecsEvaluationError):
        validate_llm_output({"result": []}, expected_count=1)


# ---------------------------------------------------------------------------
# RAG-014 — one call per batch
# ---------------------------------------------------------------------------


def test_a_batch_of_three_is_graded_in_exactly_one_call() -> None:
    client = FakeClient([evaluator_response(count=3)])
    call_llm(_request(), client=client)
    assert client.call_count == 1


def test_usage_is_extracted_from_the_response() -> None:
    response = FakeResponse(
        json.dumps(evaluator_payload(count=1)),
        usage=FakeUsage(input_tokens=1234, output_tokens=567),
    )
    usage = usage_to_dict(response)
    assert usage["input_tokens"] == 1234
    assert usage["output_tokens"] == 567


def test_text_is_extracted_from_the_content_blocks() -> None:
    assert extract_text(FakeResponse("hello")) == "hello"


# ---------------------------------------------------------------------------
# RAG-017 — retry behaviour
# ---------------------------------------------------------------------------


def _sleep(_seconds: float) -> None:
    """No-op sleep so retry backoff does not slow the suite."""


def test_malformed_then_valid_succeeds_within_the_retry_budget() -> None:
    client = FakeClient(
        [FakeResponse("not json at all"), evaluator_response(count=1)]
    )
    result = call_and_validate(
        _request(), expected_count=1, client=client, max_attempts=3, sleep=_sleep
    )

    assert result["attempts"] == 2
    assert len(result["output"]["evaluations"]) == 1


def test_retry_budget_is_bounded() -> None:
    client = FakeClient([FakeResponse("nope")] * 5)
    with pytest.raises(PecsEvaluationError) as excinfo:
        call_and_validate(
            _request(), expected_count=1, client=client, max_attempts=3, sleep=_sleep
        )

    assert client.call_count == 3
    assert excinfo.value.to_dict()["error"]["code"] == "RAG_RETRY_BUDGET_EXHAUSTED"


def test_a_schema_violation_is_retried() -> None:
    client = FakeClient(
        [
            FakeResponse(json.dumps(evaluator_payload(count=1, drop_criterion="c3"))),
            evaluator_response(count=1),
        ]
    )
    result = call_and_validate(
        _request(), expected_count=1, client=client, max_attempts=3, sleep=_sleep
    )
    assert result["attempts"] == 2


def test_a_refusal_is_not_retried() -> None:
    """A refusal is a deterministic policy outcome; retrying just burns budget."""
    client = FakeClient([FakeResponse("", stop_reason="refusal")])
    with pytest.raises(PecsEvaluationError):
        call_and_validate(
            _request(), expected_count=1, client=client, max_attempts=3, sleep=_sleep
        )
    assert client.call_count == 1


def test_every_attempt_records_its_usage() -> None:
    """Cost accounting must include the calls that failed, not only the winner."""
    client = FakeClient([FakeResponse("not json"), evaluator_response(count=1)])
    result = call_and_validate(
        _request(), expected_count=1, client=client, max_attempts=3, sleep=_sleep
    )
    assert len(result["usages"]) == result["attempts"]


# ---------------------------------------------------------------------------
# RAG-018 — transport failures
# ---------------------------------------------------------------------------


def test_a_transport_exception_becomes_a_structured_error_not_a_traceback() -> None:
    client = FakeClient([ConnectionError("connection reset by peer")])
    with pytest.raises(RagError) as excinfo:
        call_llm(_request(), client=client)

    error = excinfo.value.to_dict()["error"]
    assert error["layer"] == "rag"
    assert error["code"].startswith("RAG_LLM_")
    assert "Traceback" not in error["message"]


def test_an_empty_response_body_is_classified_distinctly() -> None:
    client = FakeClient([FakeResponse("")])
    with pytest.raises(PecsEvaluationError) as excinfo:
        call_and_validate(
            _request(), expected_count=1, client=client, max_attempts=1, sleep=_sleep
        )
    assert "EMPTY" in excinfo.value.to_dict()["error"]["code"] or excinfo.value.to_dict()[
        "error"
    ]["code"] == "RAG_RETRY_BUDGET_EXHAUSTED"


# ---------------------------------------------------------------------------
# RAG-019 — the pre-flight abort happens before any call
# ---------------------------------------------------------------------------


def test_oversized_batch_aborts_before_any_call_is_made() -> None:
    from src.rag.system_prompt import PECS_SYSTEM_PROMPT
    from src.rag.token_safety import preflight_check

    with pytest.raises(PecsEvaluationError) as excinfo:
        preflight_check(PECS_SYSTEM_PROMPT, "x" * 200_000)

    assert excinfo.value.to_dict()["error"]["code"] == "RAG_TOKEN_SAFETY_ABORT"
