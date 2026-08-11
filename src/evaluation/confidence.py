"""
EVAL-010 — [Optional] Evaluator confidence field handling.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

**Status in this MVP build: not requested, but parsed if present.**

The master system prompt (RAG-011) does not ask the evaluator for a confidence
signal, so in normal operation this field is absent and every result carries
``confidence: null`` with ``confidence_status: "not_implemented_in_mvp"``. That is
the documented omission the task allows for.

If a future rubric revision does ask for confidence, this module already parses
and normalises it — numeric 0-1, numeric 0-100, or the ``low`` / ``medium`` /
``high`` vocabulary Member 2's annotation schema uses (ANN-001) — so enabling the
field is a prompt change, not a code change.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

#: Recorded when the evaluator supplied nothing (the MVP's normal case).
STATUS_NOT_IMPLEMENTED = "not_implemented_in_mvp"

#: Recorded when a usable confidence value was parsed.
STATUS_PRESENT = "present"

#: Recorded when a value was supplied but could not be interpreted.
STATUS_UNPARSEABLE = "unparseable"

#: Categorical vocabulary shared with ANN-001, mapped to a 0-1 scale.
LABEL_TO_VALUE = {"low": 0.25, "medium": 0.6, "high": 0.9}

VALUE_TO_LABEL_BOUNDS = ((0.4, "low"), (0.75, "medium"))


def _label_for(value: float) -> str:
    for upper, label in VALUE_TO_LABEL_BOUNDS:
        if value < upper:
            return label
    return "high"


def parse_confidence(raw: Any) -> Dict[str, Any]:
    """Normalise a raw confidence value from the evaluator.

    Args:
        raw: Whatever the evaluator returned in ``confidence`` (may be None).

    Returns:
        dict with keys:
            confidence        – float in 0-1, or None
            confidence_label  – ``low``/``medium``/``high``, or None
            confidence_status – one of the ``STATUS_*`` constants
    """
    if raw is None:
        return {
            "confidence": None,
            "confidence_label": None,
            "confidence_status": STATUS_NOT_IMPLEMENTED,
        }

    if isinstance(raw, str):
        key = raw.strip().lower()
        if key in LABEL_TO_VALUE:
            value = LABEL_TO_VALUE[key]
            return {
                "confidence": value,
                "confidence_label": key,
                "confidence_status": STATUS_PRESENT,
            }
        try:
            raw = float(key)
        except ValueError:
            return {
                "confidence": None,
                "confidence_label": None,
                "confidence_status": STATUS_UNPARSEABLE,
            }

    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return {
            "confidence": None,
            "confidence_label": None,
            "confidence_status": STATUS_UNPARSEABLE,
        }

    value = float(raw)

    # Accept a 0-100 percentage as well as a 0-1 probability.
    if 1.0 < value <= 100.0:
        value = value / 100.0

    if value < 0.0 or value > 1.0:
        return {
            "confidence": None,
            "confidence_label": None,
            "confidence_status": STATUS_UNPARSEABLE,
        }

    return {
        "confidence": round(value, 3),
        "confidence_label": _label_for(value),
        "confidence_status": STATUS_PRESENT,
    }


def extract_confidence(entry: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Extract the confidence block from one validated evaluation entry."""
    if not isinstance(entry, Mapping):
        return parse_confidence(None)
    return parse_confidence(entry.get("confidence"))
