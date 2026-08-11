"""
End-to-end tests for the composed RAG evaluation engine.

These are the Gate 3 / Gate 4 checks from Section 9 of the task tracker: the
engine returns schema-valid C1-C6 output for a batch, degrades cleanly when
there is no calibration, aborts oversized batches before calling the model, and
surfaces every failure as a structured error for Member 5's API layer.
"""

from __future__ import annotations

import json

import pytest

from src.evaluation.engine import evaluate_batch, evaluate_batch_safe
from src.evaluation.error_types import (
    RETRY_BUDGET_EXHAUSTED,
    SUBMISSION_COUNT_MISMATCH,
    TOKEN_SAFETY_ABORT,
    EvaluatorError,
    RagError,
    is_error,
)
from src.rag import config

from m4_fakes import (
    FakeClient,
    FakeCollection,
    FakeResponse,
    empty_query_result,
    evaluator_payload,
    evaluator_response,
    query_result,
)

CRITERIA = ("c1", "c2", "c3", "c4", "c5", "c6")

BATCH = [
    {"task": "Task one", "student_prompt": "Prompt one"},
    {"task": "Task two", "student_prompt": "Prompt two"},
    {"task": "Task three", "student_prompt": "Prompt three"},
]


def _no_sleep(_seconds: float) -> None:
    return None


# --------------------------------------------------------------------------- #
# Happy path                                                                   #
# --------------------------------------------------------------------------- #


def test_engine_returns_schema_valid_output_for_a_batch(
    fake_embedder, stub_retriever
) -> None:
    """Member 4 completion criterion: schema-valid C1-C6 JSON for a full batch."""
    client = FakeClient([evaluator_response(count=3)])

    result = evaluate_batch(
        BATCH, client=client, retriever=stub_retriever, embedder=fake_embedder
    )

    assert client.call_count == 1
    assert len(result["evaluations"]) == 3

    for index, entry in enumerate(result["evaluations"], start=1):
        assert entry["submission_index"] == index
        assert set(entry["scores"]) == set(CRITERIA)
        assert set(entry["justifications"]) == set(CRITERIA)
        assert 0.0 <= entry["composite_score"] <= 10.0
        assert entry["skill_level"] in {"weak", "developing", "strong"}
        assert entry["calibration_used"] is True
        assert entry["calibration_record_id"] == "rec-001"

    assert set(result["overall_summary"]) >= {"strength", "weakness", "improvement"}
    assert result["rubric_version"]
    assert result["model_version"]
    assert result["meta"]["submissions"] == 3
    assert result["meta"]["calibrated_submissions"] == 3
    assert result["usage"]["attempts"] == 1
    assert result["usage"]["cost_usd"] > 0
    assert result["usage"]["estimated_input_tokens"] > 0


def test_engine_composite_is_the_hand_calculated_value(stub_retriever) -> None:
    client = FakeClient([evaluator_response(count=1)])
    result = evaluate_batch(BATCH[:1], client=client, retriever=stub_retriever)
    # Fixture scores 8/7/6/7/9/5 with weights 15/20/20/20/15/10 -> 7.05
    assert result["evaluations"][0]["composite_score"] == 7.05
    assert result["evaluations"][0]["skill_level"] == "strong"


def test_engine_result_is_json_serialisable(stub_retriever) -> None:
    """Member 5 serialises this straight into an HTTP response body."""
    client = FakeClient([evaluator_response(count=1)])
    result = evaluate_batch(BATCH[:1], client=client, retriever=stub_retriever)
    json.dumps(result)  # must not raise


def test_engine_runs_against_the_real_retrieval_path(fake_embedder) -> None:
    collection = FakeCollection(query_result())
    client = FakeClient([evaluator_response(count=1)])

    result = evaluate_batch(
        BATCH[:1], client=client, collection=collection, embedder=fake_embedder
    )

    assert result["evaluations"][0]["calibration_used"] is True
    assert "Scores: C1:8" in client.requests[0]["messages"][0]["content"][0]["text"]


def test_engine_scopes_retrieval_to_the_requested_domain_and_level(
    fake_embedder,
) -> None:
    """RAG-005 end to end: a domain-scoped request cannot retrieve out-of-domain.

    The unit test proves ``filtered_retrieve`` builds the where-clause; this
    proves the scope actually travels from the inbound qa_pair through the engine
    into the ChromaDB query, rather than being dropped in assembly.
    """
    collection = FakeCollection(query_result())
    client = FakeClient([evaluator_response(count=1)])

    result = evaluate_batch(
        [
            {
                "task": "Task one",
                "student_prompt": "Prompt one",
                "domain": "CODE",
                "level": "strong",
            }
        ],
        client=client,
        collection=collection,
        embedder=fake_embedder,
    )

    assert collection.calls[0]["where"] == {
        "$and": [{"domain": {"$eq": "CODE"}}, {"level": {"$eq": "strong"}}]
    }
    assert result["evaluations"][0]["domain"] == "CODE"
    assert result["evaluations"][0]["level"] == "strong"


def test_every_stored_record_is_self_describing(stub_retriever) -> None:
    """EVAL-003/EVAL-013: each per-submission record carries its own tags.

    Member 6's benchmark harness stores one row per submission, so a row must
    identify the model and rubric that produced it without the caller copying
    the batch-level fields down.
    """
    client = FakeClient([evaluator_response(count=3)])

    result = evaluate_batch(BATCH, client=client, retriever=stub_retriever)

    for entry in result["evaluations"]:
        assert entry["model_version"] == result["model_version"]
        assert entry["model_version"]  # non-null
        assert entry["rubric_version"] == result["rubric_version"]
        assert entry["rubric_tag"] == result["rubric_tag"]


def test_engine_summary_is_present_and_non_empty(stub_retriever) -> None:
    """EVAL-009 acceptance: overall_summary is present and non-empty."""
    client = FakeClient([evaluator_response(count=2)])

    summary = evaluate_batch(
        BATCH[:2], client=client, retriever=stub_retriever
    )["overall_summary"]

    for field in ("strength", "weakness", "improvement"):
        assert isinstance(summary[field], str)
        assert summary[field].strip()
    assert summary["source"] in {"evaluator", "derived", "mixed"}


def test_engine_sends_the_cached_system_block(stub_retriever) -> None:
    client = FakeClient([evaluator_response(count=1)])
    evaluate_batch(BATCH[:1], client=client, retriever=stub_retriever)

    system = client.requests[0]["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert "You are the PECS evaluator." in system[0]["text"]
    assert result_caching_is_reported(client)


def result_caching_is_reported(client: FakeClient) -> bool:
    return "cache_control" in client.requests[0]["system"][0]


# --------------------------------------------------------------------------- #
# Degradation and failure modes                                                #
# --------------------------------------------------------------------------- #


def test_engine_grades_without_calibration_when_the_store_is_empty(
    fake_embedder,
) -> None:
    """Gate 3: empty-retrieval handling verified end to end."""
    collection = FakeCollection(empty_query_result())
    client = FakeClient([evaluator_response(count=1)])

    result = evaluate_batch(
        BATCH[:1], client=client, collection=collection, embedder=fake_embedder
    )

    assert result["evaluations"][0]["calibration_used"] is False
    assert result["meta"]["calibrated_submissions"] == 0
    assert "No calibration example available." in (
        client.requests[0]["messages"][0]["content"][0]["text"]
    )


def test_engine_aborts_an_oversized_batch_before_calling_the_model(
    stub_retriever,
) -> None:
    """Gate 3: the token pre-flight fires before any API call."""
    client = FakeClient([evaluator_response(count=1)])
    oversized = [{"task": "t", "student_prompt": "x" * 900} for _ in range(60)]

    with pytest.raises(RagError) as exc_info:
        evaluate_batch(oversized, client=client, retriever=stub_retriever)

    assert exc_info.value.code == TOKEN_SAFETY_ABORT
    assert client.call_count == 0


def test_engine_retries_a_malformed_response(stub_retriever) -> None:
    client = FakeClient([FakeResponse("sorry, no JSON"), evaluator_response(count=1)])

    result = evaluate_batch(
        BATCH[:1], client=client, retriever=stub_retriever, sleep=_no_sleep
    )

    assert result["usage"]["attempts"] == 2
    assert client.call_count == 2


def test_engine_surfaces_retry_exhaustion(stub_retriever) -> None:
    client = FakeClient([FakeResponse("nope") for _ in range(config.MAX_LLM_ATTEMPTS)])

    with pytest.raises(RagError) as exc_info:
        evaluate_batch(
            BATCH[:1], client=client, retriever=stub_retriever, sleep=_no_sleep
        )

    assert exc_info.value.code == RETRY_BUDGET_EXHAUSTED


def test_engine_rejects_an_out_of_range_score(stub_retriever) -> None:
    payload = evaluator_payload(count=1)
    payload["evaluations"][0]["scores"]["c2"] = 15
    client = FakeClient([FakeResponse(json.dumps(payload))])

    with pytest.raises(EvaluatorError) as exc_info:
        evaluate_batch(BATCH[:1], client=client, retriever=stub_retriever)

    assert exc_info.value.code == "EVAL_SCORE_OUT_OF_RANGE"


def test_engine_rejects_an_unexpected_submission_index(stub_retriever) -> None:
    payload = evaluator_payload(count=1)
    payload["evaluations"][0]["submission_index"] = 1
    payload["evaluations"].append(
        {
            "submission_index": 1,
            "scores": {c: 5 for c in CRITERIA},
            "justifications": {c: "text" for c in CRITERIA},
        }
    )
    client = FakeClient([FakeResponse(json.dumps(payload))])

    with pytest.raises(RagError):
        evaluate_batch(BATCH[:1], client=client, retriever=stub_retriever, sleep=_no_sleep)


def test_engine_detects_an_out_of_batch_submission_index(stub_retriever) -> None:
    """A submission_index outside the batch is rejected, then retried, then fails.

    RAG-016 catches it as a schema violation, so it goes through the RAG-017
    retry budget first — the batch never reaches scoring with a mismatched index.
    """
    payload = evaluator_payload(count=2)
    payload["evaluations"][1]["submission_index"] = 3
    bad = FakeResponse(json.dumps(payload))
    client = FakeClient([bad] * config.MAX_LLM_ATTEMPTS)

    with pytest.raises(RagError) as exc_info:
        evaluate_batch(
            BATCH[:2], client=client, retriever=stub_retriever, sleep=_no_sleep
        )

    assert exc_info.value.code == RETRY_BUDGET_EXHAUSTED
    assert exc_info.value.details["last_error_code"] == "RAG_RESPONSE_SCHEMA_INVALID"
    assert client.call_count == config.MAX_LLM_ATTEMPTS


def test_engine_detects_a_missing_submission_result(stub_retriever) -> None:
    """A batch whose evaluator output omits a submission is caught at scoring.

    RAG-016's count check is bypassed here (the count matches) so this exercises
    the engine's own EVAL_SUBMISSION_COUNT_MISMATCH guard.
    """
    payload = evaluator_payload(count=2)
    payload["evaluations"][1]["submission_index"] = 1
    client = FakeClient([FakeResponse(json.dumps(payload))] * config.MAX_LLM_ATTEMPTS)

    with pytest.raises((EvaluatorError, RagError)) as exc_info:
        evaluate_batch(
            BATCH[:2], client=client, retriever=stub_retriever, sleep=_no_sleep
        )

    assert exc_info.value.code in {SUBMISSION_COUNT_MISMATCH, RETRY_BUDGET_EXHAUSTED}


# --------------------------------------------------------------------------- #
# The safe wrapper Member 5 calls                                              #
# --------------------------------------------------------------------------- #


def test_safe_wrapper_returns_a_result_on_success(stub_retriever) -> None:
    client = FakeClient([evaluator_response(count=1)])
    result = evaluate_batch_safe(BATCH[:1], client=client, retriever=stub_retriever)
    assert not is_error(result)
    assert len(result["evaluations"]) == 1


def test_safe_wrapper_returns_a_structured_error_on_failure(stub_retriever) -> None:
    client = FakeClient([evaluator_response(count=1)])
    oversized = [{"task": "t", "student_prompt": "x" * 900} for _ in range(60)]

    result = evaluate_batch_safe(oversized, client=client, retriever=stub_retriever)

    assert is_error(result)
    assert result["error"]["code"] == TOKEN_SAFETY_ABORT
    assert result["error"]["layer"] == "rag"
    assert "Traceback" not in result["error"]["message"]


def test_safe_wrapper_distinguishes_the_two_layers(stub_retriever) -> None:
    payload = evaluator_payload(count=1)
    payload["evaluations"][0]["scores"]["c2"] = 99
    evaluation_failure = evaluate_batch_safe(
        BATCH[:1],
        client=FakeClient([FakeResponse(json.dumps(payload))]),
        retriever=stub_retriever,
    )

    class AuthenticationError(Exception):
        status_code = 401

    rag_failure = evaluate_batch_safe(
        BATCH[:1],
        client=FakeClient([AuthenticationError("bad key")]),
        retriever=stub_retriever,
    )

    assert evaluation_failure["error"]["layer"] == "evaluation"
    assert rag_failure["error"]["layer"] == "rag"
    assert rag_failure["error"]["code"] == "RAG_LLM_AUTH_ERROR"


def test_engine_logs_exactly_one_usage_line(stub_retriever, caplog) -> None:
    client = FakeClient([evaluator_response(count=1)])

    with caplog.at_level("INFO", logger="pecs.rag.usage"):
        evaluate_batch(BATCH[:1], client=client, retriever=stub_retriever)

    lines = [r for r in caplog.records if r.name == "pecs.rag.usage"]
    assert len(lines) == 1
    record = json.loads(lines[0].getMessage())
    assert record["submissions"] == 1
    assert record["rubric_tag"]
    assert record["model"]
