from __future__ import annotations

from .annotation_schema import CRITERIA, WEIGHTS


def weighted_score(scores: dict[str, float]) -> float:
    missing = [c for c in CRITERIA if c not in scores]
    if missing:
        raise ValueError(f"Missing criteria: {missing}")
    for c in CRITERIA:
        if not 1 <= float(scores[c]) <= 10:
            raise ValueError(f"{c} must be between 1 and 10")
    return round(sum(float(scores[c]) * WEIGHTS[c] for c in CRITERIA), 2)


def skill_level(score: float) -> str:
    if not 0 <= float(score) <= 10:
        raise ValueError("composite score must be between 0 and 10")
    if score < 4.0:
        return "weak"
    if score < 7.0:
        return "developing"
    return "strong"
