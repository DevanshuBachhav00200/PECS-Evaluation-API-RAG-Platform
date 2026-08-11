"""
RAG-017 — Retry handling for malformed evaluator responses.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Retries the LLM call a bounded number of times when the response fails JSON
parsing or schema validation, then surfaces a structured error. A
malformed-then-valid sequence succeeds inside the retry budget; a run of
malformed responses ends as ``RAG_RETRY_BUDGET_EXHAUSTED`` carrying the last
underlying reason, so the failure is diagnosable from the error object alone.

Scope note: only *unusable response body* failures are retried here — invalid
JSON, schema violations, and an empty response
(``error_types.RETRYABLE_RESPONSE_CODES``). Three classes are deliberately
excluded:

* **Transport failures** (rate limit, timeout, connection, 5xx) — the Anthropic
  SDK already retries those internally with backoff, and stacking a second loop
  would multiply worst-case latency without adding reliability.
* **Refusals** — a refusal is a deterministic policy outcome. Re-sending
  identical input cannot change it, so a retry only burns budget and latency.
* **Configuration errors** (e.g. an inconsistent rubric) — retrying cannot fix
  a bad constant, and hiding it behind three attempts delays the real signal.

Every attempt's token usage is returned so the cost of a retried call is logged
in full (RAG-020) rather than only the successful attempt.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from src.evaluation.error_types import (
    RETRY_BUDGET_EXHAUSTED,
    RETRYABLE_RESPONSE_CODES,
    PecsEvaluationError,
    RagError,
)
from src.rag.config import MAX_LLM_ATTEMPTS
from src.rag.llm_call import call_llm, extract_text, usage_to_dict
from src.rag.parse_response import parse_llm_response
from src.rag.validate_llm_output import validate_llm_output

#: Base backoff in seconds between retry attempts (linear: base * attempt).
RETRY_BACKOFF_SECONDS = 0.5


def call_and_validate(
    request: Dict[str, Any],
    expected_count: Optional[int] = None,
    client: Optional[Any] = None,
    max_attempts: Optional[int] = None,
    sleep: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Call the evaluator and return a parsed, schema-valid result.

    Args:
        request:        Payload from ``build_request()``.
        expected_count: Number of submissions in the batch, forwarded to
                        RAG-016 so a dropped submission is caught.
        client:         Optional injected client (tests / benchmark harness).
        max_attempts:   Total attempts including the first. Defaults to
                        ``config.MAX_LLM_ATTEMPTS``.
        sleep:          Injectable sleep, so tests do not wait on backoff.

    Returns:
        dict with keys:
            output   – the validated payload from RAG-016
            attempts – how many calls were actually made
            usages   – per-attempt token usage, in attempt order
            response – the raw response of the successful attempt
            raw_text – the raw text of the successful attempt

    Raises:
        RagError: ``RAG_RETRY_BUDGET_EXHAUSTED`` when every attempt produced an
                  unusable response; the original transport error (unchanged)
                  when the failure was not a malformed response.
    """
    attempts_allowed = MAX_LLM_ATTEMPTS if max_attempts is None else int(max_attempts)
    if attempts_allowed < 1:
        raise ValueError(f"max_attempts must be at least 1, got {attempts_allowed}")

    pause = sleep if sleep is not None else time.sleep

    usages: List[Dict[str, int]] = []
    last_error: Optional[PecsEvaluationError] = None

    for attempt in range(1, attempts_allowed + 1):
        # Transport-layer failures propagate untouched — already classified by
        # RAG-018 and not worth a second retry loop.
        response = call_llm(request, client=client)
        usages.append(usage_to_dict(response))

        try:
            raw_text = extract_text(response)
            parsed = parse_llm_response(raw_text)
            output = validate_llm_output(parsed, expected_count=expected_count)
        except PecsEvaluationError as exc:
            # Only an unusable response body is worth another sample. A refusal,
            # or a configuration error, propagates on the first occurrence.
            if exc.code not in RETRYABLE_RESPONSE_CODES:
                raise

            last_error = exc
            if attempt < attempts_allowed:
                pause(RETRY_BACKOFF_SECONDS * attempt)
                continue
            break
        else:
            return {
                "output": output,
                "attempts": attempt,
                "usages": usages,
                "response": response,
                "raw_text": raw_text,
            }

    details: Dict[str, Any] = {
        "attempts": attempts_allowed,
        "usages": usages,
    }
    if last_error is not None:
        details["last_error_code"] = last_error.code
        details["last_error_message"] = last_error.message
        details["last_error_details"] = last_error.details

    raise RagError(
        f"Evaluator returned an unusable response {attempts_allowed} time(s); "
        "giving up.",
        code=RETRY_BUDGET_EXHAUSTED,
        details=details,
    )
