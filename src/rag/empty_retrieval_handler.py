"""
RAG-006 — Empty-retrieval handling.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Defines the fallback behaviour when the vector store returns no match — an empty
collection, a metadata filter with zero results, or a record whose document
cannot be deserialized. The evaluation must degrade to uncalibrated grading
rather than crash: the evaluator is still perfectly capable of scoring against
the rubric without a reference example, it is just less calibrated.

Reasons are enumerated so Member 6's benchmarking can tell "the store is empty"
apart from "this domain filter matched nothing".
"""

from __future__ import annotations

from typing import Any, Dict, Optional

#: Text substituted into the prompt in place of a calibration line.
NO_CALIBRATION_TEXT = "No calibration example available."

#: Enumerated causes of an empty retrieval.
REASON_EMPTY_COLLECTION = "empty_collection"
REASON_NO_FILTER_MATCH = "no_filter_match"
REASON_MALFORMED_RECORD = "malformed_record"
REASON_RETRIEVAL_DISABLED = "retrieval_disabled"

VALID_REASONS = frozenset(
    {
        REASON_EMPTY_COLLECTION,
        REASON_NO_FILTER_MATCH,
        REASON_MALFORMED_RECORD,
        REASON_RETRIEVAL_DISABLED,
    }
)


def no_calibration_result(
    reason: str = REASON_EMPTY_COLLECTION,
    detail: Optional[str] = None,
    **context: Any,
) -> Dict[str, Any]:
    """Build the structured 'no calibration available' retrieval result.

    Args:
        reason:  One of the ``REASON_*`` constants in this module.
        detail:  Optional human-readable detail for logs and reports.
        context: Extra fields to record (e.g. ``domain="CODE"``).

    Returns:
        dict: Same shape as a successful ``retrieve_context()`` result but with
              ``available=False``, so downstream prompt assembly needs no
              special-casing beyond reading ``calibration``.

    Raises:
        ValueError: If ``reason`` is not a recognised constant.
    """
    if reason not in VALID_REASONS:
        raise ValueError(
            f"Unknown empty-retrieval reason '{reason}'; "
            f"must be one of {sorted(VALID_REASONS)}"
        )

    result: Dict[str, Any] = {
        "available": False,
        "calibration": NO_CALIBRATION_TEXT,
        "reason": reason,
        "detail": detail,
        "record_id": None,
        "distance": None,
        "metadata": None,
        "examples": [],
    }
    result.update(context)
    return result


def is_empty(results: Any) -> bool:
    """Return True when a ChromaDB query result carries no records.

    Tolerates every shape ChromaDB can return for a miss: a missing ``ids``
    key, ``None``, an empty outer list, or an empty inner list.
    """
    if not results:
        return True

    ids = results.get("ids") if isinstance(results, dict) else None
    if not ids:
        return True

    first = ids[0]
    return not first
