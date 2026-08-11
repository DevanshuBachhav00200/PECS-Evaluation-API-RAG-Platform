"""
RAG-008 — Compact calibration-string formatter.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Formats a retrieved calibration example into the single compact line consumed by
the prompt builder (RAG-009 / RAG-013):

    Scores: C1:8 | C2:7 | C3:9 | C4:6 | C5:10 | C6:5 -> Justification: <text>

The arrow is the Unicode RIGHTWARDS ARROW (U+2192), matching the documented
compact format. Keeping the calibration on one line matters: the batched call
carries one of these per submission and the token pre-flight cap (RAG-019) is
only 4,000 estimated input tokens.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

CRITERIA = ("c1", "c2", "c3", "c4", "c5", "c6")

#: Separator between the score block and the justification block.
ARROW = "→"


def _format_score(value: Any) -> str:
    """Render a score without a spurious trailing ``.0``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"score must be a number, got {type(value).__name__}")
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):g}"


def _normalize_scores(scores: Mapping[str, Any]) -> Dict[str, Any]:
    """Accept either ``c1..c6`` or ``c1_score..c6_score`` key styles."""
    if not isinstance(scores, Mapping):
        raise TypeError(f"scores must be a mapping, got {type(scores).__name__}")

    keys = set(scores.keys())

    if set(CRITERIA).issubset(keys):
        return {criterion: scores[criterion] for criterion in CRITERIA}

    suffixed = {f"{criterion}_score" for criterion in CRITERIA}
    if suffixed.issubset(keys):
        return {criterion: scores[f"{criterion}_score"] for criterion in CRITERIA}

    missing = sorted(set(CRITERIA) - keys)
    raise ValueError(
        "scores must contain all 6 calibration keys "
        f"(c1..c6 or c1_score..c6_score); missing {missing}"
    )


def format_calibration(scores: Mapping[str, Any], justification: str) -> str:
    """Format one calibration example into the compact single-line form.

    Args:
        scores:        Mapping containing C1-C6 scores (either key style).
        justification: The reviewed justification text for those scores.

    Returns:
        str: e.g. ``"Scores: C1:8 | C2:7 | C3:9 | C4:6 | C5:10 | C6:5 -> Justification: ..."``
             (with U+2192 in place of ``->``).

    Raises:
        TypeError:  On non-mapping scores, non-string justification, or a
                    non-numeric score value.
        ValueError: If any of the six criteria is missing, or the justification
                    is empty/whitespace only.
    """
    if not isinstance(justification, str):
        raise TypeError(
            f"justification must be a string, got {type(justification).__name__}"
        )
    if not justification.strip():
        raise ValueError("justification cannot be empty or whitespace only")

    resolved = _normalize_scores(scores)

    score_block = " | ".join(
        f"{criterion.upper()}:{_format_score(resolved[criterion])}"
        for criterion in CRITERIA
    )

    return f"Scores: {score_block} {ARROW} Justification: {justification.strip()}"


def format_calibration_from_document(
    document: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> str:
    """Format a calibration example straight from a ChromaDB record.

    Bridges VDB-003's stored document representation to the compact line above.
    When the document carries no usable scores (older or partial records) the
    C1-C6 values are recovered from the VDB-004 metadata instead.

    Args:
        document: The JSON document string stored in ChromaDB (VDB-003).
        metadata: Optional VDB-004 metadata dict, used as the score fallback.

    Returns:
        str: The compact calibration line.
    """
    # Imported lazily: `src.vectorstore` instantiates a persistent ChromaDB
    # client at package-import time (VDB-006), and formatting a string must not
    # drag a database into processes that never retrieve anything.
    from src.vectorstore.document_format import deserialize_document

    try:
        payload = deserialize_document(document)
    except (TypeError, ValueError):
        # VDB-003's deserializer rejects a partial score set outright. Metadata
        # (VDB-004) is schema-validated to carry all six criteria, so when it is
        # available the record is still usable — recover the justification from
        # the raw document and take the scores from metadata.
        if metadata is None:
            raise
        return format_calibration(metadata, _justification_from_raw(document))

    try:
        return format_calibration(payload["scores"], payload["justification"])
    except ValueError:
        if metadata is None:
            raise
        return format_calibration(metadata, payload["justification"])


def _justification_from_raw(document: str) -> str:
    """Pull the justification out of a document that failed full validation."""
    try:
        raw = json.loads(document)
    except json.JSONDecodeError as exc:
        raise ValueError(f"document is not a valid JSON string: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("document payload must be a JSON object")

    justification = raw.get("justification")
    if not isinstance(justification, str) or not justification.strip():
        raise ValueError("document payload has no usable justification")

    return justification
