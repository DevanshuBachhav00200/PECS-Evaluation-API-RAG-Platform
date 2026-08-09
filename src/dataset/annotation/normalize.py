"""Canonical score normalization for Member 2."""
from __future__ import annotations
from typing import Any

def normalize_score(value: Any) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError("score must be numeric and between 1 and 10")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("score must be numeric and between 1 and 10") from exc
    if score != score or score < 1 or score > 10:
        raise ValueError("score must be between 1 and 10")
    return score
