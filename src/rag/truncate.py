"""
RAG-010 — Per-submission prompt truncation.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Any individual student prompt longer than ``config.MAX_PROMPT_CHARS`` (1,000 by
default) is truncated before it is added to the assembled batch content. This is
the first line of defence for the token pre-flight cap (RAG-019): one pathological
5,000-character submission should not abort a whole batch of otherwise-normal
ones.

Truncation is visible, not silent: the returned text carries a marker so the
evaluator (and any auditor reading the log) can tell the prompt was cut.
"""

from __future__ import annotations

from typing import Optional, Tuple

from src.rag.config import MAX_PROMPT_CHARS, TRUNCATION_SUFFIX


def truncate_prompt(text: str, limit: Optional[int] = None) -> str:
    """Truncate ``text`` to ``limit`` characters, appending a visible marker.

    Args:
        text:  The student prompt.
        limit: Character limit. Defaults to ``config.MAX_PROMPT_CHARS``.

    Returns:
        str: The original text when it is within the limit, otherwise the first
             ``limit`` characters followed by ``TRUNCATION_SUFFIX``.

    Raises:
        TypeError:  If ``text`` is not a string or ``limit`` is not an int.
        ValueError: If ``limit`` is not positive.
    """
    truncated, _ = truncate_with_flag(text, limit)
    return truncated


def truncate_with_flag(
    text: str,
    limit: Optional[int] = None,
) -> Tuple[str, bool]:
    """Same as ``truncate_prompt`` but also report whether truncation occurred.

    Returns:
        tuple[str, bool]: ``(text, was_truncated)``. The flag is what the usage
        logger and the API response use to disclose that a prompt was cut.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")

    effective_limit = MAX_PROMPT_CHARS if limit is None else limit

    if not isinstance(effective_limit, int) or isinstance(effective_limit, bool):
        raise TypeError(
            f"limit must be a positive integer, got {type(effective_limit).__name__}"
        )
    if effective_limit <= 0:
        raise ValueError(f"limit must be a positive integer, got {effective_limit}")

    if len(text) <= effective_limit:
        return text, False

    return text[:effective_limit] + TRUNCATION_SUFFIX, True
