"""
Tests for RAG-014 … RAG-018 and RAG-020: the single batched call, fence-stripped
response handling, schema validation, bounded retries, API failure handling, and
usage/cost logging.
"""

from __future__ import annotations

import json

import pytest

from src.evaluation.error_types import (
    LLM_AUTH_ERROR,
    LLM_CONNECTION_ERROR,
    LLM_RATE_LIMITED,
    LLM_SERVER_ERROR,
    LLM_TIMEOUT,
    RESPONSE_NOT_JSON,
    RESPONSE_SCHEMA_INVALID,
    RETRY_BUDGET_EXHAUSTED,
    RUBRIC_UNAVAILABLE,
    EvaluatorError,
    RagError,
    ResponseValidationError,
)
from src.rag.api_error_handler import classify_api_error, handle_api_error, to_rag_error
from src.rag.build_request import build_request
from src.rag.llm_call import call_llm, extract_text, usage_to_dict
from src.rag.parse_response import parse_llm_response, strip_code_fences
from src.rag.retry_handler import call_and_validate
from src.rag.usage_logger import aggregate_usages, estimate_cost_usd, log_usage
from src.rag.validate_llm_output import validate_llm_output

from m4_fakes import (
    FakeClient,
    FakeResponse,
    FakeUsage,
    evaluator_payload,
    evaluator_response,
)


# --------------------------------------------------------------------------- #
# RAG-014 — exactly one API call per batch                                     #
# --------------------------------------------------------------------------- #


def test_batch_of_three_is_graded_in_exactly_one_call() -> None:
    """RAG-014 acceptance: a test batch of 3 submissions costs one API call."""
    client = FakeClient([evaluator_response(count=3)])
    request = build_request("three submissions here")

    result = call_and_validate(request, expected_count=3, client=client)

    assert client.call_count == 1
    assert len(result["output"]["evaluations"]) == 3
    assert result["attempts"] == 1


def test_call_llm_forwards_the_payload_verbatim() -> None:
    client = FakeClient([evaluator_response()])
    request = build_request("batch")
    call_llm(request, client=client)

    sent = client.requests[0]
    assert sent["model"] == request["model"]
    assert sent["system"] == request["system"]
    assert sent["messages"] == request["messages"]


def test_extract_text_concatenates_text_blocks() -> None:
    response = FakeResponse("part one")
    response.content.append(type(response.content[0])("part two"))
    assert extract_text(response) == "part onepart two"


def test_extract_text_raises_on_refusal() -> None:
    response = FakeResponse("", stop_reason="refusal")
    with pytest.raises(RagError) as exc_info:
        extract_text(response)
    assert exc_info.value.code == "RAG_LLM_REFUSAL"


# --------------------------------------------------------------------------- #
# RAG-015 — fence stripping                                                    #
# --------------------------------------------------------------------------- #


def test_fenced_json_still_parses() -> None:
    """RAG-015 acceptance: a response wrapped in ```json fences parses."""
    payload = evaluator_payload(count=1)
    fenced = f"```json\n{json.dumps(payload)}\n```"
    assert parse_llm_response(fenced) == payload


def test_bare_fence_and_plain_json_both_parse() -> None:
    payload = {"evaluations": []}
    assert parse_llm_response(f"```\n{json.dumps(payload)}\n```") == payload
    assert parse_llm_response(json.dumps(payload)) == payload


def test_prose_around_the_object_is_tolerated() -> None:
    text = 'Here is the result:\n{"evaluations": [], "note": "a } inside a string"}\nDone.'
    parsed = parse_llm_response(text)
    assert parsed["note"] == "a } inside a string"


def test_strip_code_fences_leaves_unfenced_text_alone() -> None:
    assert strip_code_fences("  {'a': 1}  ") == "{'a': 1}"


def test_unparseable_response_is_structured() -> None:
    with pytest.raises(ResponseValidationError) as exc_info:
        parse_llm_response("this is not json at all")
    assert exc_info.value.code == RESPONSE_NOT_JSON
    assert exc_info.value.to_dict()["error"]["layer"] == "rag"


def test_non_object_json_is_rejected() -> None:
    with pytest.raises(ResponseValidationError):
        parse_llm_response("[1, 2, 3]")


# --------------------------------------------------------------------------- #
# RAG-016 — schema validation of the parsed response                           #
# --------------------------------------------------------------------------- #


def test_missing_c5_is_rejected_with_a_clear_error() -> None:
    """RAG-016 acceptance: a response missing c5 is rejected clearly."""
    payload = evaluator_payload(count=1, drop_criterion="c5")

    with pytest.raises(ResponseValidationError) as exc_info:
        validate_llm_output(payload, expected_count=1)

    error = exc_info.value
    assert error.code == RESPONSE_SCHEMA_INVALID
    assert "C5" in error.message
    assert error.details["missing"] == ["c5"]


def test_valid_payload_is_returned_sorted_by_index() -> None:
    payload = evaluator_payload(count=3)
    payload["evaluations"].reverse()

    output = validate_llm_output(payload, expected_count=3)

    assert [e["submission_index"] for e in output["evaluations"]] == [1, 2, 3]
    assert set(output["overall_summary"]) == {"strength", "weakness", "improvement"}


def test_wrong_evaluation_count_is_rejected() -> None:
    payload = evaluator_payload(count=2)
    with pytest.raises(ResponseValidationError, match="2 evaluations for 3"):
        validate_llm_output(payload, expected_count=3)


def test_duplicate_submission_index_is_rejected() -> None:
    payload = evaluator_payload(count=2)
    payload["evaluations"][1]["submission_index"] = 1
    with pytest.raises(ResponseValidationError, match="duplicate"):
        validate_llm_output(payload, expected_count=2)


def test_missing_overall_summary_is_rejected() -> None:
    payload = evaluator_payload(count=1, include_summary=False)
    with pytest.raises(ResponseValidationError, match="overall_summary"):
        validate_llm_output(payload, expected_count=1)


def test_non_numeric_score_is_rejected() -> None:
    payload = evaluator_payload(count=1)
    payload["evaluations"][0]["scores"]["c3"] = "high"
    with pytest.raises(ResponseValidationError, match="numeric"):
        validate_llm_output(payload, expected_count=1)


def test_empty_justification_is_rejected() -> None:
    payload = evaluator_payload(count=1)
    payload["evaluations"][0]["justifications"]["c2"] = "   "
    with pytest.raises(ResponseValidationError, match="empty"):
        validate_llm_output(payload, expected_count=1)


def test_optional_confidence_is_preserved() -> None:
    payload = evaluator_payload(count=1)
    payload["evaluations"][0]["confidence"] = 0.8
    output = validate_llm_output(payload, expected_count=1)
    assert output["evaluations"][0]["confidence"] == 0.8


# --------------------------------------------------------------------------- #
# RAG-017 — bounded retries on malformed responses                             #
# --------------------------------------------------------------------------- #


def test_malformed_then_valid_succeeds_within_the_budget() -> None:
    """RAG-017 acceptance: malformed-then-valid succeeds inside the budget."""
    client = FakeClient(
        [
            FakeResponse("I'm afraid I can't do that."),
            evaluator_response(count=1),
        ]
    )

    result = call_and_validate(
        request=build_request("batch"),
        expected_count=1,
        client=client,
        max_attempts=3,
        sleep=lambda _seconds: None,
    )

    assert result["attempts"] == 2
    assert client.call_count == 2
    assert len(result["usages"]) == 2  # both attempts are billed and reported


def test_retry_budget_exhaustion_is_structured() -> None:
    client = FakeClient([FakeResponse("nope") for _ in range(3)])

    with pytest.raises(RagError) as exc_info:
        call_and_validate(
            request=build_request("batch"),
            expected_count=1,
            client=client,
            max_attempts=3,
            sleep=lambda _seconds: None,
        )

    error = exc_info.value
    assert error.code == RETRY_BUDGET_EXHAUSTED
    assert error.details["attempts"] == 3
    assert error.details["last_error_code"] == RESPONSE_NOT_JSON
    assert client.call_count == 3


def test_schema_failure_is_also_retried() -> None:
    bad = FakeResponse(json.dumps(evaluator_payload(drop_criterion="c5")))
    client = FakeClient([bad, evaluator_response(count=1)])

    result = call_and_validate(
        request=build_request("batch"),
        expected_count=1,
        client=client,
        max_attempts=2,
        sleep=lambda _seconds: None,
    )
    assert result["attempts"] == 2


def test_transport_errors_are_not_retried_here() -> None:
    """Transport failures surface immediately — the SDK already retries them."""

    class RateLimitError(Exception):
        status_code = 429

    client = FakeClient([RateLimitError("slow down"), evaluator_response()])

    with pytest.raises(RagError) as exc_info:
        call_and_validate(
            request=build_request("batch"),
            expected_count=1,
            client=client,
            max_attempts=3,
            sleep=lambda _seconds: None,
        )

    assert exc_info.value.code == LLM_RATE_LIMITED
    assert client.call_count == 1


def test_a_refusal_is_not_retried() -> None:
    """A refusal is deterministic — re-sending identical input only burns budget.

    RAG-017 retries unusable *response bodies*, not policy outcomes.
    """
    client = FakeClient(
        [
            FakeResponse("", stop_reason="refusal"),
            evaluator_response(count=1),
        ]
    )

    with pytest.raises(RagError) as exc_info:
        call_and_validate(
            request=build_request("batch"),
            expected_count=1,
            client=client,
            max_attempts=3,
            sleep=lambda _seconds: None,
        )

    assert exc_info.value.code == "RAG_LLM_REFUSAL"
    assert client.call_count == 1


def test_an_empty_response_is_retried() -> None:
    """An empty body carries no information — a fresh sample may succeed."""
    empty = FakeResponse("")
    empty.content = []
    client = FakeClient([empty, evaluator_response(count=1)])

    result = call_and_validate(
        request=build_request("batch"),
        expected_count=1,
        client=client,
        max_attempts=3,
        sleep=lambda _seconds: None,
    )

    assert result["attempts"] == 2


def test_empty_response_has_its_own_code() -> None:
    empty = FakeResponse("")
    empty.content = []
    with pytest.raises(RagError) as exc_info:
        extract_text(empty)
    assert exc_info.value.code == "RAG_LLM_EMPTY_RESPONSE"


def test_a_configuration_error_is_not_retried(monkeypatch) -> None:
    """A bad rubric cannot be fixed by another sample; surface it immediately."""

    def broken_validate(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        raise EvaluatorError("rubric weights must sum to 1.0", code=RUBRIC_UNAVAILABLE)

    monkeypatch.setattr(
        "src.rag.retry_handler.validate_llm_output", broken_validate
    )
    client = FakeClient([evaluator_response(count=1)] * 3)

    with pytest.raises(EvaluatorError) as exc_info:
        call_and_validate(
            request=build_request("batch"),
            expected_count=1,
            client=client,
            max_attempts=3,
            sleep=lambda _seconds: None,
        )

    assert exc_info.value.code == RUBRIC_UNAVAILABLE
    assert client.call_count == 1


def test_fenced_response_needs_no_retry() -> None:
    client = FakeClient([evaluator_response(count=1, fenced=True)])
    result = call_and_validate(
        request=build_request("batch"), expected_count=1, client=client
    )
    assert result["attempts"] == 1


# --------------------------------------------------------------------------- #
# RAG-018 — API failure handling                                               #
# --------------------------------------------------------------------------- #


class AuthenticationError(Exception):
    status_code = 401


class APITimeoutError(Exception):
    pass


class APIConnectionError(Exception):
    pass


class InternalServerError(Exception):
    status_code = 503


@pytest.mark.parametrize(
    "exc, expected",
    [
        (AuthenticationError("bad key"), LLM_AUTH_ERROR),
        (APITimeoutError("timed out"), LLM_TIMEOUT),
        (APIConnectionError("no route"), LLM_CONNECTION_ERROR),
        (InternalServerError("overloaded"), LLM_SERVER_ERROR),
    ],
)
def test_api_errors_are_classified(exc: Exception, expected: str) -> None:
    assert classify_api_error(exc) == expected


def test_api_key_error_returns_a_structured_error_not_a_traceback() -> None:
    """RAG-018 acceptance: a simulated API-key error returns a structured error."""
    client = FakeClient([AuthenticationError("invalid x-api-key")])

    with pytest.raises(RagError) as exc_info:
        call_llm(build_request("batch"), client=client)

    payload = exc_info.value.to_dict()
    assert payload["error"]["code"] == LLM_AUTH_ERROR
    assert payload["error"]["layer"] == "rag"
    assert payload["error"]["retryable"] is False
    assert "Traceback" not in payload["error"]["message"]


def test_retryable_flag_is_set_for_transient_failures() -> None:
    payload = handle_api_error(InternalServerError("overloaded"))
    assert payload["error"]["retryable"] is True


def test_to_rag_error_records_status_and_context() -> None:
    error = to_rag_error(InternalServerError("boom"), context={"model": "m"})
    assert error.details["status_code"] == 503
    assert error.details["model"] == "m"
    assert error.details["exception_type"] == "InternalServerError"


def test_unknown_status_maps_to_server_error() -> None:
    class Weird(Exception):
        status_code = 599

    assert classify_api_error(Weird()) == LLM_SERVER_ERROR


# --------------------------------------------------------------------------- #
# RAG-020 — usage and cost logging                                             #
# --------------------------------------------------------------------------- #


def test_usage_to_dict_reads_all_four_counters() -> None:
    response = FakeResponse(
        "{}",
        usage=FakeUsage(
            input_tokens=1000,
            output_tokens=200,
            cache_creation_input_tokens=800,
            cache_read_input_tokens=400,
        ),
    )
    assert usage_to_dict(response) == {
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_creation_input_tokens": 800,
        "cache_read_input_tokens": 400,
    }


def test_usage_to_dict_defaults_missing_counters_to_zero() -> None:
    assert usage_to_dict({"usage": {}})["input_tokens"] == 0
    assert usage_to_dict({})["output_tokens"] == 0


def test_cost_estimate_applies_cache_multipliers() -> None:
    cost = estimate_cost_usd(
        {
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "cache_creation_input_tokens": 1_000_000,
            "cache_read_input_tokens": 1_000_000,
        }
    )
    assert cost["input_usd"] == 1.0
    assert cost["output_usd"] == 5.0
    assert cost["cache_write_usd"] == 1.25
    assert cost["cache_read_usd"] == 0.10
    assert cost["total_usd"] == pytest.approx(7.35)


def test_aggregate_usages_sums_every_attempt() -> None:
    totals = aggregate_usages(
        [
            {"input_tokens": 100, "output_tokens": 10},
            {"input_tokens": 120, "output_tokens": 12},
        ]
    )
    assert totals["input_tokens"] == 220
    assert totals["output_tokens"] == 22


def test_log_usage_emits_one_line_per_call(caplog) -> None:
    """RAG-020 acceptance: every successful call produces one usage log line."""
    with caplog.at_level("INFO", logger="pecs.rag.usage"):
        record = log_usage(
            {"input_tokens": 1500, "output_tokens": 400},
            submissions=3,
            attempts=1,
            rubric_tag="pecs-rubric-v1.0+abc123456789",
        )

    lines = [r for r in caplog.records if r.name == "pecs.rag.usage"]
    assert len(lines) == 1

    logged = json.loads(lines[0].getMessage())
    assert logged["event"] == "evaluation_call"
    assert logged["submissions"] == 3
    assert logged["input_tokens"] == 1500
    assert logged["cost_usd"] == record["cost_usd"] > 0
    assert logged["rubric_tag"] == "pecs-rubric-v1.0+abc123456789"
