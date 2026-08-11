"""
RAG-019 — Pre-flight token estimation and safety abort.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Estimates input tokens with the documented chars/3.5 heuristic BEFORE any LLM
call is made, and aborts with a structured SAFETY ABORT when the estimate
exceeds the configured cap (4,000 estimated input tokens by default).

The heuristic is intentionally cheap and local: the point is to refuse a
runaway batch without paying for a round trip, not to be exact. It errs on the
side of over-counting for prompt-heavy English text, which is the safe direction
for a cap. The authoritative count is whatever the API reports in
``usage.input_tokens`` afterwards (see RAG-020).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from src.evaluation.error_types import TOKEN_SAFETY_ABORT, TokenSafetyAbort
from src.rag.config import CHARS_PER_TOKEN, TOKEN_SAFETY_CAP


def estimate_tokens(text: str, chars_per_token: Optional[float] = None) -> int:
    """Estimate the token count of ``text``.

    Args:
        text:            The text to estimate.
        chars_per_token: Override for the heuristic ratio. Defaults to
                         ``config.CHARS_PER_TOKEN`` (3.5).

    Returns:
        int: Estimated tokens, rounded up. Empty text is 0.

    Raises:
        TypeError:  If ``text`` is not a string.
        ValueError: If the ratio is not positive.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")

    ratio = CHARS_PER_TOKEN if chars_per_token is None else float(chars_per_token)
    if ratio <= 0:
        raise ValueError(f"chars_per_token must be positive, got {ratio}")

    if not text:
        return 0

    return math.ceil(len(text) / ratio)


def estimate_request_tokens(system_prompt: str, batch_content: str) -> int:
    """Estimate the total input tokens of one evaluation request."""
    return estimate_tokens(system_prompt) + estimate_tokens(batch_content)


def preflight_check(
    system_prompt: str,
    batch_content: str,
    cap: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the pre-flight estimate and abort if it exceeds the cap.

    Args:
        system_prompt:  The rubric system prompt (RAG-011).
        batch_content:  The assembled batch block (RAG-009).
        cap:            Override for ``config.TOKEN_SAFETY_CAP``.

    Returns:
        dict: ``{"estimated_input_tokens": int, "cap": int, "aborted": False}``
              when the request is within budget.

    Raises:
        TokenSafetyAbort: When the estimate exceeds the cap. Carries code
                          ``RAG_TOKEN_SAFETY_ABORT`` and the numbers that
                          triggered it, so the API layer can tell the caller
                          how far over budget the batch was.
    """
    effective_cap = TOKEN_SAFETY_CAP if cap is None else int(cap)

    system_tokens = estimate_tokens(system_prompt)
    content_tokens = estimate_tokens(batch_content)
    total = system_tokens + content_tokens

    if total > effective_cap:
        raise TokenSafetyAbort(
            "SAFETY ABORT: estimated input tokens "
            f"({total}) exceed the configured cap ({effective_cap}). "
            "No LLM call was made.",
            code=TOKEN_SAFETY_ABORT,
            details={
                "estimated_input_tokens": total,
                "system_prompt_tokens": system_tokens,
                "batch_content_tokens": content_tokens,
                "cap": effective_cap,
                "chars_per_token": CHARS_PER_TOKEN,
            },
        )

    return {
        "estimated_input_tokens": total,
        "system_prompt_tokens": system_tokens,
        "batch_content_tokens": content_tokens,
        "cap": effective_cap,
        "aborted": False,
    }
