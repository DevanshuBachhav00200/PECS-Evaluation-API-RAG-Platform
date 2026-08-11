"""
EVAL-005 — Per-category justification extraction and word-limit enforcement.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

The system prompt asks for at most 30 words per criterion; this module enforces
the contract rather than trusting it. An over-limit justification is truncated to
the configured cap (``config.JUSTIFICATION_WORD_CAP``, 40 by default — slightly
above the prompt's own instruction so ordinary variance is not clipped, while a
runaway paragraph still is).

Truncation is marked with an ellipsis so a reader can tell text was cut, and the
API response reports which criteria were truncated.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from src.evaluation.error_types import MISSING_CRITERION, EvaluatorError
from src.evaluation.rubric_loader import load_rubric
from src.rag.config import JUSTIFICATION_WORD_CAP, SUMMARY_WORD_CAP

#: Appended when text is cut, so truncation is never silent.
TRUNCATION_MARKER = "..."


def enforce_word_cap(
    text: str,
    cap: Optional[int] = None,
) -> Tuple[str, bool]:
    """Trim ``text`` to at most ``cap`` words.

    Args:
        text: The text to cap.
        cap:  Word cap. Defaults to ``config.JUSTIFICATION_WORD_CAP``.

    Returns:
        tuple[str, bool]: ``(text, was_truncated)``. Whitespace is collapsed to
        single spaces so a word count is meaningful.

    Raises:
        TypeError:  If ``text`` is not a string or ``cap`` is not an int.
        ValueError: If ``cap`` is not positive.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")

    effective_cap = JUSTIFICATION_WORD_CAP if cap is None else cap
    if not isinstance(effective_cap, int) or isinstance(effective_cap, bool):
        raise TypeError(f"cap must be an integer, got {type(effective_cap).__name__}")
    if effective_cap <= 0:
        raise ValueError(f"cap must be positive, got {effective_cap}")

    words = text.split()

    if len(words) <= effective_cap:
        return " ".join(words), False

    return " ".join(words[:effective_cap]) + TRUNCATION_MARKER, True


def extract_justifications(
    justifications: Mapping[str, Any],
    cap: Optional[int] = None,
    submission_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract and cap the per-criterion justifications for one submission.

    Args:
        justifications:   Mapping of criterion -> justification text.
        cap:              Word cap override.
        submission_index: Threaded into error details.

    Returns:
        dict with keys:
            justifications – criterion -> capped text, in rubric order
            truncated      – list of criteria whose text was cut

    Raises:
        EvaluatorError: ``EVAL_MISSING_CRITERION`` if a criterion is absent or
                        its text is empty.
    """
    if not isinstance(justifications, Mapping):
        raise EvaluatorError(
            f"justifications must be a mapping, got {type(justifications).__name__}",
            code=MISSING_CRITERION,
            details={"submission_index": submission_index},
        )

    criteria = load_rubric().criteria
    resolved: Dict[str, str] = {}
    truncated: list[str] = []

    for criterion in criteria:
        if criterion not in justifications:
            raise EvaluatorError(
                f"missing justification for {criterion.upper()}",
                code=MISSING_CRITERION,
                details={
                    "submission_index": submission_index,
                    "criterion": criterion,
                },
            )

        raw = justifications[criterion]
        if not isinstance(raw, str) or not raw.strip():
            raise EvaluatorError(
                f"justification for {criterion.upper()} must be non-empty text",
                code=MISSING_CRITERION,
                details={
                    "submission_index": submission_index,
                    "criterion": criterion,
                },
            )

        capped, was_truncated = enforce_word_cap(raw, cap)
        resolved[criterion] = capped
        if was_truncated:
            truncated.append(criterion)

    return {"justifications": resolved, "truncated": truncated}


def enforce_summary_word_cap(text: str) -> Tuple[str, bool]:
    """Apply the batch-summary word cap (``config.SUMMARY_WORD_CAP``)."""
    return enforce_word_cap(text, SUMMARY_WORD_CAP)
