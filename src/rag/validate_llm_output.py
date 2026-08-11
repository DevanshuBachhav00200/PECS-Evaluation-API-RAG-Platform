"""
RAG-016 — JSON schema validation of the parsed evaluator response.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Validates the parsed evaluator JSON against the expected C1-C6 structure before
anything reaches scoring. A response missing a required field (say ``c5``) is
rejected here with a clear validation error rather than producing a composite
score computed from five criteria.

Structural validation only. Range enforcement is EVAL-004's job, and word-limit
enforcement is EVAL-005's — this module checks that the fields exist and have
the right types, so those layers can assume shape and only police values.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from src.evaluation.error_types import (
    RESPONSE_SCHEMA_INVALID,
    ResponseValidationError,
)
from src.evaluation.rubric_loader import load_rubric

#: Fields required on the batch-level overall_summary (EVAL-009).
SUMMARY_FIELDS = ("strength", "weakness", "improvement")


def _fail(message: str, **details: Any) -> None:
    raise ResponseValidationError(
        message,
        code=RESPONSE_SCHEMA_INVALID,
        details=details,
    )


def validate_llm_output(
    parsed: Mapping[str, Any],
    expected_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Validate the evaluator's parsed output.

    Args:
        parsed:         The dict returned by RAG-015 ``parse_llm_response()``.
        expected_count: Number of submissions sent. When given, the evaluations
                        list length and the ``submission_index`` values are
                        checked against it — this is what catches the evaluator
                        silently dropping or merging a submission in a batch.

    Returns:
        dict: The validated payload, with ``evaluations`` sorted by
              ``submission_index`` so downstream code can rely on the order.

    Raises:
        ResponseValidationError: On any structural problem. Code
                                 ``RAG_RESPONSE_SCHEMA_INVALID``.
    """
    criteria = load_rubric().criteria

    if not isinstance(parsed, Mapping):
        _fail(
            f"evaluator output must be an object, got {type(parsed).__name__}",
            parsed_type=type(parsed).__name__,
        )

    evaluations = parsed.get("evaluations")
    if not isinstance(evaluations, list):
        _fail(
            "evaluator output is missing the 'evaluations' list",
            present_keys=sorted(parsed.keys()),
        )
    if not evaluations:
        _fail("evaluator output contains zero evaluations", count=0)

    if expected_count is not None and len(evaluations) != expected_count:
        _fail(
            f"evaluator returned {len(evaluations)} evaluations for "
            f"{expected_count} submission(s)",
            returned=len(evaluations),
            expected=expected_count,
        )

    validated: List[Dict[str, Any]] = []
    seen_indices: List[int] = []

    for position, entry in enumerate(evaluations, start=1):
        if not isinstance(entry, Mapping):
            _fail(
                f"evaluations[{position - 1}] must be an object",
                position=position,
                entry_type=type(entry).__name__,
            )

        index = entry.get("submission_index", position)
        if isinstance(index, bool) or not isinstance(index, int):
            _fail(
                f"evaluations[{position - 1}].submission_index must be an integer",
                position=position,
                submission_index=repr(index),
            )
        if index < 1:
            _fail(
                "submission_index must be 1-based",
                position=position,
                submission_index=index,
            )
        if expected_count is not None and index > expected_count:
            _fail(
                f"submission_index {index} is outside the submitted batch "
                f"(1-{expected_count})",
                submission_index=index,
                expected=expected_count,
            )
        if index in seen_indices:
            _fail(
                f"duplicate submission_index {index} in evaluator output",
                submission_index=index,
            )
        seen_indices.append(index)

        scores = entry.get("scores")
        if not isinstance(scores, Mapping):
            _fail(
                f"submission {index} is missing the 'scores' object",
                submission_index=index,
            )

        missing_scores = [c for c in criteria if c not in scores]
        if missing_scores:
            _fail(
                f"submission {index} is missing score(s) for "
                f"{[c.upper() for c in missing_scores]}",
                submission_index=index,
                missing=missing_scores,
            )

        for criterion in criteria:
            value = scores[criterion]
            if value is None:
                _fail(
                    f"submission {index} has a null score for {criterion.upper()}",
                    submission_index=index,
                    criterion=criterion,
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                _fail(
                    f"submission {index} score for {criterion.upper()} must be "
                    f"numeric, got {type(value).__name__}",
                    submission_index=index,
                    criterion=criterion,
                )

        justifications = entry.get("justifications")
        if not isinstance(justifications, Mapping):
            _fail(
                f"submission {index} is missing the 'justifications' object",
                submission_index=index,
            )

        missing_justifications = [c for c in criteria if c not in justifications]
        if missing_justifications:
            _fail(
                f"submission {index} is missing justification(s) for "
                f"{[c.upper() for c in missing_justifications]}",
                submission_index=index,
                missing=missing_justifications,
            )

        for criterion in criteria:
            text = justifications[criterion]
            if not isinstance(text, str):
                _fail(
                    f"submission {index} justification for {criterion.upper()} "
                    f"must be a string, got {type(text).__name__}",
                    submission_index=index,
                    criterion=criterion,
                )
            if not text.strip():
                _fail(
                    f"submission {index} justification for {criterion.upper()} "
                    "is empty",
                    submission_index=index,
                    criterion=criterion,
                )

        validated_entry: Dict[str, Any] = {
            "submission_index": index,
            "scores": {criterion: scores[criterion] for criterion in criteria},
            "justifications": {
                criterion: justifications[criterion] for criterion in criteria
            },
        }

        # Optional evaluator confidence signal (EVAL-010). Kept verbatim here;
        # parsed and normalised by src/evaluation/confidence.py.
        if "confidence" in entry:
            validated_entry["confidence"] = entry["confidence"]

        validated.append(validated_entry)

    summary = parsed.get("overall_summary")
    if not isinstance(summary, Mapping):
        _fail(
            "evaluator output is missing the 'overall_summary' object",
            present_keys=sorted(parsed.keys()),
        )

    missing_summary = [field for field in SUMMARY_FIELDS if field not in summary]
    if missing_summary:
        _fail(
            f"overall_summary is missing {missing_summary}",
            missing=missing_summary,
        )

    for field in SUMMARY_FIELDS:
        value = summary[field]
        if not isinstance(value, str):
            _fail(
                f"overall_summary.{field} must be a string, got "
                f"{type(value).__name__}",
                field=field,
            )
        if not value.strip():
            _fail(f"overall_summary.{field} is empty", field=field)

    validated.sort(key=lambda item: item["submission_index"])

    return {
        "evaluations": validated,
        "overall_summary": {field: summary[field].strip() for field in SUMMARY_FIELDS},
    }
