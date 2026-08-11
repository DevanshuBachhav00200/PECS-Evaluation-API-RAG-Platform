"""
Tests for RAG-009 … RAG-013 and RAG-019: batch assembly, truncation, the master
system prompt, request construction, prompt caching, and the token pre-flight.
"""

from __future__ import annotations

import pytest

from src.evaluation.error_types import TOKEN_SAFETY_ABORT, RagError, TokenSafetyAbort
from src.rag import config
from src.rag.build_batch_content import (
    build_batch_content,
    build_batch_content_with_items,
    build_batch_items,
    render_batch_content,
)
from src.rag.build_request import (
    build_request,
    caching_diagnostics,
    validate_payload,
)
from src.rag.empty_retrieval_handler import NO_CALIBRATION_TEXT
from src.rag.system_prompt import (
    PECS_SYSTEM_PROMPT,
    system_prompt_fingerprint,
)
from src.rag.token_safety import (
    estimate_request_tokens,
    estimate_tokens,
    preflight_check,
)
from src.rag.truncate import truncate_prompt, truncate_with_flag

from m4_fakes import FakeCollection, empty_query_result, query_result


PAIRS = [
    {"task": "Task one", "student_prompt": "Prompt one"},
    {"task": "Task two", "student_prompt": "Prompt two"},
    {"task": "Task three", "student_prompt": "Prompt three"},
]


# --------------------------------------------------------------------------- #
# RAG-010 — per-submission truncation                                          #
# --------------------------------------------------------------------------- #


def test_oversized_prompt_is_truncated_to_the_configured_limit() -> None:
    """RAG-010 acceptance: a 5,000-character prompt is truncated to the limit."""
    text, truncated = truncate_with_flag("x" * 5000)
    assert truncated is True
    assert text.startswith("x" * config.MAX_PROMPT_CHARS)
    assert text.endswith(config.TRUNCATION_SUFFIX)
    assert len(text) == config.MAX_PROMPT_CHARS + len(config.TRUNCATION_SUFFIX)


def test_prompt_within_limit_is_untouched() -> None:
    text, truncated = truncate_with_flag("short prompt")
    assert text == "short prompt"
    assert truncated is False


def test_truncate_rejects_bad_limit() -> None:
    with pytest.raises(ValueError):
        truncate_prompt("abc", limit=0)


def test_truncation_flag_reaches_the_evaluator_input(stub_retriever) -> None:
    items = build_batch_items(
        [{"task": "t", "student_prompt": "y" * 4000}], retriever=stub_retriever
    )
    assert items[0].truncated is True
    assert len(items[0].student_prompt) <= config.MAX_PROMPT_CHARS + len(
        config.TRUNCATION_SUFFIX
    )


# --------------------------------------------------------------------------- #
# RAG-009 — build_batch_content                                                #
# --------------------------------------------------------------------------- #


def test_batch_content_has_one_entry_per_pair_in_order(stub_retriever) -> None:
    """RAG-009 acceptance: one correctly formatted entry per qa_pair, in order."""
    content = build_batch_content(PAIRS, retriever=stub_retriever)

    assert content.count("### Submission ") == 3
    assert content.index("### Submission 1") < content.index("### Submission 2")
    assert content.index("### Submission 2") < content.index("### Submission 3")
    for pair in PAIRS:
        assert pair["task"] in content
        assert pair["student_prompt"] in content
    assert content.count("Calibration: Scores: C1:8") == 3


def test_batch_items_carry_retrieval_provenance(stub_retriever) -> None:
    items = build_batch_items(PAIRS, retriever=stub_retriever)
    assert [item.submission_index for item in items] == [1, 2, 3]
    assert all(item.calibration_available for item in items)
    assert all(item.record_id == "rec-001" for item in items)


def test_batch_content_uses_real_retrieval_path(fake_embedder) -> None:
    collection = FakeCollection(query_result())
    content, items = build_batch_content_with_items(
        PAIRS[:1], collection=collection, embedder=fake_embedder
    )
    assert "Scores: C1:8 | C2:7" in content
    assert items[0].calibration_available is True


def test_batch_content_degrades_when_nothing_is_retrieved(fake_embedder) -> None:
    collection = FakeCollection(empty_query_result())
    content, items = build_batch_content_with_items(
        PAIRS[:1], collection=collection, embedder=fake_embedder
    )
    assert NO_CALIBRATION_TEXT in content
    assert items[0].calibration_available is False


def test_batch_content_can_skip_retrieval_entirely() -> None:
    content, items = build_batch_content_with_items(
        PAIRS[:1], retrieve_calibration=False
    )
    assert NO_CALIBRATION_TEXT in content
    assert items[0].calibration_available is False


def test_retrieval_failure_becomes_a_structured_rag_error() -> None:
    def boom(*args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        raise RuntimeError("chroma is down")

    with pytest.raises(RagError) as exc_info:
        build_batch_items(PAIRS[:1], retriever=boom)

    assert exc_info.value.code == "RAG_RETRIEVAL_FAILED"
    assert exc_info.value.details["submission_index"] == 1


def test_empty_batch_is_rejected() -> None:
    with pytest.raises(RagError):
        build_batch_items([])


def test_malformed_pair_is_rejected() -> None:
    from src.evaluation.error_types import EvaluatorError

    with pytest.raises(EvaluatorError) as exc_info:
        build_batch_items([{"task": "t"}], retrieve_calibration=False)
    assert exc_info.value.code == "EVAL_INVALID_INPUT"


def test_tuple_pairs_are_accepted() -> None:
    items = build_batch_items(
        [("Task", "Prompt")], retrieve_calibration=False
    )
    assert items[0].task == "Task"
    assert items[0].student_prompt == "Prompt"


def test_render_batch_content_is_independent_of_retrieval(stub_retriever) -> None:
    items = build_batch_items(PAIRS, retriever=stub_retriever)
    assert render_batch_content(items) == build_batch_content(
        PAIRS, retriever=stub_retriever
    )


# --------------------------------------------------------------------------- #
# RAG-011 — master system prompt                                               #
# --------------------------------------------------------------------------- #


def test_system_prompt_encodes_every_criterion_and_weight() -> None:
    for criterion, weight in (
        ("C1", "15%"),
        ("C2", "20%"),
        ("C3", "20%"),
        ("C4", "20%"),
        ("C5", "15%"),
        ("C6", "10%"),
    ):
        assert f"{criterion} " in PECS_SYSTEM_PROMPT
        assert weight in PECS_SYSTEM_PROMPT


def test_system_prompt_uses_ann002_criterion_names() -> None:
    for name in (
        "Foundations",
        "Design & Patterns",
        "Output Specification",
        "Domain Application",
        "Ethics & Safety",
        "Metacognition",
    ):
        assert name in PECS_SYSTEM_PROMPT


def test_system_prompt_states_the_skill_bands() -> None:
    assert "weak 0.00-3.99" in PECS_SYSTEM_PROMPT
    assert "developing 4.00-6.99" in PECS_SYSTEM_PROMPT
    assert "strong 7.00-10.00" in PECS_SYSTEM_PROMPT


def test_system_prompt_demands_json_only() -> None:
    assert "Return raw JSON only" in PECS_SYSTEM_PROMPT
    assert "no code fences" in PECS_SYSTEM_PROMPT


def test_system_prompt_is_deterministic() -> None:
    """Caching is a prefix match — the prompt must not vary between calls."""
    assert system_prompt_fingerprint() == system_prompt_fingerprint()
    assert len(system_prompt_fingerprint()) == 12


# --------------------------------------------------------------------------- #
# RAG-012 / RAG-013 — request construction and prompt caching                  #
# --------------------------------------------------------------------------- #


def test_request_matches_the_messages_api_shape() -> None:
    """RAG-013 acceptance: the payload passes dry-run validation."""
    request = build_request("Evaluate this batch.")

    assert request["model"] == config.MODEL_ID
    assert request["max_tokens"] == config.MAX_TOKENS
    assert request["temperature"] == config.TEMPERATURE
    assert request["system"][0]["type"] == "text"
    assert request["messages"][0]["role"] == "user"
    assert request["messages"][0]["content"][0]["text"] == "Evaluate this batch."

    validate_payload(request)  # does not raise


def test_system_block_is_marked_for_ephemeral_caching() -> None:
    """RAG-012: the rubric block carries cache_control."""
    request = build_request("batch")
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_caching_can_be_disabled() -> None:
    request = build_request("batch", enable_caching=False)
    assert "cache_control" not in request["system"][0]


def test_volatile_content_never_enters_the_cached_prefix() -> None:
    """The batch content must live in messages, not in the system block."""
    first = build_request("batch one")
    second = build_request("batch two")
    assert first["system"] == second["system"]
    assert first["messages"] != second["messages"]


def test_caching_diagnostics_report_the_prefix_minimum() -> None:
    diagnostics = caching_diagnostics(build_request("batch"))
    assert diagnostics["requested"] is True
    assert diagnostics["minimum_tokens"] == 4096
    assert diagnostics["prefix_tokens"] > 0
    # Honest reporting: on Haiku 4.5 the condensed rubric alone is below the
    # 4,096-token minimum cacheable prefix, so a cache hit is not expected.
    assert diagnostics["expected_effective"] == (
        diagnostics["prefix_tokens"] >= diagnostics["minimum_tokens"]
    )


def test_validate_payload_rejects_structural_problems() -> None:
    good = build_request("batch")

    bad_model = {**good, "model": ""}
    with pytest.raises(ValueError, match="model"):
        validate_payload(bad_model)

    bad_tokens = {**good, "max_tokens": 0}
    with pytest.raises(ValueError, match="max_tokens"):
        validate_payload(bad_tokens)

    two_messages = {**good, "messages": good["messages"] * 2}
    with pytest.raises(ValueError, match="exactly one user message"):
        validate_payload(two_messages)


def test_build_request_rejects_empty_batch_content() -> None:
    with pytest.raises(ValueError):
        build_request("   ")


# --------------------------------------------------------------------------- #
# RAG-019 — token pre-flight and safety abort                                  #
# --------------------------------------------------------------------------- #


def test_estimate_tokens_uses_the_documented_heuristic() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("x" * 350) == 100  # 350 / 3.5
    assert estimate_tokens("x" * 351) == 101  # rounds up


def test_preflight_passes_a_normal_batch(stub_retriever) -> None:
    content = build_batch_content(PAIRS, retriever=stub_retriever)
    result = preflight_check(PECS_SYSTEM_PROMPT, content)

    assert result["aborted"] is False
    assert result["estimated_input_tokens"] == estimate_request_tokens(
        PECS_SYSTEM_PROMPT, content
    )
    assert result["cap"] == config.TOKEN_SAFETY_CAP


def test_oversized_batch_aborts_before_any_llm_call() -> None:
    """RAG-019 acceptance: the abort fires before a call is made."""
    with pytest.raises(TokenSafetyAbort) as exc_info:
        preflight_check(PECS_SYSTEM_PROMPT, "y" * 100_000)

    error = exc_info.value
    assert error.code == TOKEN_SAFETY_ABORT
    assert "SAFETY ABORT" in error.message
    assert error.details["estimated_input_tokens"] > error.details["cap"]
    assert error.to_dict()["error"]["layer"] == "rag"


def test_preflight_cap_is_overridable() -> None:
    with pytest.raises(TokenSafetyAbort):
        preflight_check("system", "content", cap=1)
