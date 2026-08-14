"""
TEST-009 — Malformed-input and boundary tests against /evaluate.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Sends malformed JSON, boundary-value payloads and oversized batches straight at
the live endpoint, the way a downstream caller would.

Acceptance (TEST-009): no malformed request produces an unhandled 500 with a
leaked stack trace. Two assertions carry that, and both are applied to every
case in the table rather than to a chosen few:

* the status code is one the integration contract documents (API-010), and
* the response body contains no traceback, no file path, and no exception class
  name — because a stack trace in a 500 is how an internal path, a dependency
  version, or a key name reaches a caller who should never see it.

This module also holds the API-side half of BENCH-008: the edge cases whose
payloads are invalid at the boundary are declared in ``benchmark/edge_cases.py``
and asserted here, so the benchmark suite and the test suite cannot disagree
about what those cases are.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import pytest

from benchmark.edge_cases import EDGE_CASES

#: Status codes the integration contract documents (docs/integration_contract_errors.md).
CONTRACTED_STATUSES = {200, 401, 403, 413, 422, 429, 500, 502, 504}

#: Substrings that must never appear in a response body.
LEAK_MARKERS = (
    "Traceback",
    "File \"",
    ".py\", line",
    "site-packages",
    "ANTHROPIC_API_KEY",
    "sk-ant",
)


def _assert_no_leak(body_text: str) -> None:
    for marker in LEAK_MARKERS:
        assert marker not in body_text, f"response leaked {marker!r}"
    # An un-caught exception rendered into the body would carry its class name.
    assert not re.search(r"\b\w*(Error|Exception)\b: ", body_text), (
        "response body looks like a rendered Python exception"
    )


def _assert_handled(response: Any) -> None:
    assert response.status_code in CONTRACTED_STATUSES, (
        f"undocumented status {response.status_code}"
    )
    _assert_no_leak(response.text)


# ---------------------------------------------------------------------------
# Malformed bodies — must be 422 from Pydantic, never reaching the engine
# ---------------------------------------------------------------------------

MALFORMED_BODIES: List[Dict[str, Any]] = [
    {"label": "missing student_prompt", "json": {"submissions": [{"task": "t"}]}},
    {"label": "missing task", "json": {"submissions": [{"student_prompt": "p"}]}},
    {"label": "empty submissions list", "json": {"submissions": []}},
    {"label": "submissions is a string", "json": {"submissions": "nope"}},
    {"label": "submissions is null", "json": {"submissions": None}},
    {"label": "no submissions key", "json": {"items": []}},
    {"label": "empty object", "json": {}},
    {"label": "top-level list", "json": [{"task": "t", "student_prompt": "p"}]},
    {"label": "empty task string", "json": {"submissions": [{"task": "", "student_prompt": "p"}]}},
    {"label": "whitespace-only prompt", "json": {"submissions": [{"task": "t", "student_prompt": "   "}]}},
    {"label": "task is a number", "json": {"submissions": [{"task": 5, "student_prompt": "p"}]}},
    {"label": "prompt is a list", "json": {"submissions": [{"task": "t", "student_prompt": ["p"]}]}},
    {"label": "nested null submission", "json": {"submissions": [None]}},
]


@pytest.mark.parametrize("case", MALFORMED_BODIES, ids=lambda case: case["label"])
def test_malformed_body_is_rejected_with_422(
    api_client: Any, auth_headers: Dict[str, str], case: Dict[str, Any]
) -> None:
    response = api_client.post("/evaluate", json=case["json"], headers=auth_headers)

    _assert_handled(response)
    assert response.status_code == 422, (
        f"{case['label']} should be a validation error, got {response.status_code}"
    )


def test_raw_invalid_json_is_rejected_without_a_traceback(
    api_client: Any, auth_headers: Dict[str, str]
) -> None:
    response = api_client.post(
        "/evaluate",
        content='{"submissions": [{"task": "t", ',
        headers={**auth_headers, "Content-Type": "application/json"},
    )
    _assert_handled(response)
    assert response.status_code == 422


def test_non_json_content_type_is_handled(
    api_client: Any, auth_headers: Dict[str, str]
) -> None:
    response = api_client.post(
        "/evaluate",
        content="task=t&student_prompt=p",
        headers={**auth_headers, "Content-Type": "application/x-www-form-urlencoded"},
    )
    _assert_handled(response)


def test_validation_error_body_names_the_offending_field(
    api_client: Any, auth_headers: Dict[str, str]
) -> None:
    """A 422 has to be actionable for the caller, not just a status code."""
    response = api_client.post(
        "/evaluate",
        json={"submissions": [{"task": "t"}]},
        headers=auth_headers,
    )
    assert "student_prompt" in response.text


# ---------------------------------------------------------------------------
# Authentication boundaries (API-007) — checked before any body parsing
# ---------------------------------------------------------------------------


def test_missing_api_key_is_rejected(api_client: Any) -> None:
    response = api_client.post(
        "/evaluate", json={"submissions": [{"task": "t", "student_prompt": "p"}]}
    )
    _assert_handled(response)
    assert response.status_code == 403


def test_wrong_api_key_is_rejected(api_client: Any) -> None:
    response = api_client.post(
        "/evaluate",
        json={"submissions": [{"task": "t", "student_prompt": "p"}]},
        headers={"x-api-key": "not-the-key"},
    )
    _assert_handled(response)
    assert response.status_code == 403


def test_rejected_auth_does_not_echo_the_expected_key(api_client: Any) -> None:
    response = api_client.post(
        "/evaluate",
        json={"submissions": [{"task": "t", "student_prompt": "p"}]},
        headers={"x-api-key": "not-the-key"},
    )
    assert "member6-test-key" not in response.text


def test_health_needs_no_api_key(api_client: Any) -> None:
    """API-004: the container health check runs unauthenticated."""
    assert api_client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# Boundary values — the smallest and largest inputs that still validate
# ---------------------------------------------------------------------------


def test_single_character_fields_pass_validation(
    api_client: Any, auth_headers: Dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """min_length=1 means one character is valid input, not a malformed one."""
    from src.api import main as api_main

    monkeypatch.setattr(
        api_main,
        "evaluate_batch_safe",
        lambda pairs, **kwargs: {
            "error": {
                "code": "RAG_LLM_AUTH_ERROR",
                "layer": "rag",
                "message": "no credential in test",
                "details": {},
                "retryable": False,
            }
        },
    )

    response = api_client.post(
        "/evaluate",
        json={"submissions": [{"task": "t", "student_prompt": "p"}]},
        headers=auth_headers,
    )
    # Reaches the engine (so validation passed) and comes back mapped, not 422.
    assert response.status_code != 422
    _assert_handled(response)


def test_oversized_batch_returns_the_documented_413(
    api_client: Any, auth_headers: Dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """RAG-019's safety abort maps to 413, with the estimate in the body."""
    from src.api import main as api_main

    monkeypatch.setattr(
        api_main,
        "evaluate_batch_safe",
        lambda pairs, **kwargs: {
            "error": {
                "code": "RAG_TOKEN_SAFETY_ABORT",
                "layer": "rag",
                "message": "SAFETY ABORT: estimated input tokens exceed the cap",
                "details": {"estimated_input_tokens": 9001, "safety_cap": 4000},
                "retryable": False,
            }
        },
    )

    response = api_client.post(
        "/evaluate",
        json={
            "submissions": [
                {"task": "t", "student_prompt": "x" * 4000} for _ in range(50)
            ]
        },
        headers=auth_headers,
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "RAG_TOKEN_SAFETY_ABORT"
    _assert_no_leak(response.text)


def test_very_large_body_does_not_crash_the_server(
    api_client: Any, auth_headers: Dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 2 MB body must be handled, whatever the outcome."""
    from src.api import main as api_main

    monkeypatch.setattr(
        api_main,
        "evaluate_batch_safe",
        lambda pairs, **kwargs: {
            "error": {
                "code": "RAG_TOKEN_SAFETY_ABORT",
                "layer": "rag",
                "message": "SAFETY ABORT",
                "details": {},
                "retryable": False,
            }
        },
    )

    response = api_client.post(
        "/evaluate",
        json={"submissions": [{"task": "t", "student_prompt": "y" * 2_000_000}]},
        headers=auth_headers,
    )
    _assert_handled(response)


@pytest.mark.parametrize(
    "hostile",
    [
        "'; DROP TABLE prompts; --",
        "<script>alert('x')</script>",
        "{{7*7}}",
        "../../etc/passwd",
        "\x00\x01\x02",
        "🙂" * 500,
    ],
)
def test_hostile_looking_prompt_text_is_treated_as_data(
    api_client: Any,
    auth_headers: Dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    hostile: str,
) -> None:
    """Prompt text is content to be graded, never interpreted."""
    from src.api import main as api_main

    seen: Dict[str, Any] = {}

    def _capture(pairs: Any, **kwargs: Any) -> Dict[str, Any]:
        seen["pairs"] = pairs
        return {
            "error": {
                "code": "RAG_LLM_AUTH_ERROR",
                "layer": "rag",
                "message": "no credential in test",
                "details": {},
                "retryable": False,
            }
        }

    monkeypatch.setattr(api_main, "evaluate_batch_safe", _capture)

    response = api_client.post(
        "/evaluate",
        json={"submissions": [{"task": "Review this.", "student_prompt": hostile}]},
        headers=auth_headers,
    )

    _assert_handled(response)
    if response.status_code not in (413, 422):
        assert seen["pairs"][0]["student_prompt"] == hostile.strip()


# ---------------------------------------------------------------------------
# BENCH-008's boundary-rejected cases, asserted here (see module docstring)
# ---------------------------------------------------------------------------

BOUNDARY_EDGE_CASES = [case for case in EDGE_CASES if case.raw_payload is not None]


@pytest.mark.parametrize(
    "case", BOUNDARY_EDGE_CASES, ids=lambda case: case.case_id
)
def test_benchmark_edge_case_payload_is_rejected_at_the_boundary(
    api_client: Any, auth_headers: Dict[str, str], case: Any
) -> None:
    response = api_client.post("/evaluate", json=case.raw_payload, headers=auth_headers)
    _assert_handled(response)
    assert response.status_code == 422, (
        f"{case.case_id} ({case.description}) should be rejected before the "
        f"engine, got {response.status_code}"
    )


def test_every_malformed_edge_case_is_covered_here() -> None:
    """BENCH-008 declares the malformed-input category; it must not go untested."""
    assert BOUNDARY_EDGE_CASES, "no malformed-input edge cases declared"
    assert all(case.category == "malformed_input" for case in BOUNDARY_EDGE_CASES)
