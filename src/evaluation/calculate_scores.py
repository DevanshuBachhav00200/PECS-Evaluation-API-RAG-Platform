"""
EVAL-006 — Weighted composite score calculation.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Applies the canonical C1-C6 weights (15/20/20/20/15/10) to produce a single
composite score on the same 0-10 scale as the individual criteria.

The formula is the plain weighted sum:

    composite = sum(score[c] * weight[c] for c in c1..c6)

Because the weights sum to 1.0 and each score lies in 1-10, the composite lands
in 1-10 with no rescaling. There is deliberately **no x10 multiplication** — that
multiplication was the documented scoring error this task exists to avoid, and
the skill bands (EVAL-007) are defined on the 0-10 scale.

Weights are read through EVAL-002's rubric loader, so editing the canonical
weights changes scoring on the next call with no edit here.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from src.evaluation.rubric_loader import load_rubric
from src.evaluation.score_range_guard import enforce_score_range
from src.evaluation.skill_level_map import map_skill_level

#: Decimal places the composite is rounded to. Two is enough to place a score
#: unambiguously inside a band whose boundaries are stated to two decimals.
COMPOSITE_PRECISION = 2


def calculate_composite(
    scores: Mapping[str, Any],
    validate: bool = True,
    submission_index: Optional[int] = None,
) -> float:
    """Compute the weighted composite score for one submission.

    Args:
        scores:           Mapping of criterion -> score (1-10).
        validate:         Run EVAL-004's range guard first (default True).
                          Pass False only when the caller has already validated.
        submission_index: Threaded into range-guard error details.

    Returns:
        float: Composite score in 0-10, rounded to two decimals.

    Raises:
        EvaluatorError: From the range guard, on a missing or out-of-range score.
    """
    rubric = load_rubric()

    resolved = (
        enforce_score_range(scores, submission_index=submission_index)
        if validate
        else {criterion: float(scores[criterion]) for criterion in rubric.criteria}
    )

    total = sum(
        resolved[criterion] * float(rubric.weight(criterion))
        for criterion in rubric.criteria
    )

    return round(total, COMPOSITE_PRECISION)


def calculate_scores(
    scores: Mapping[str, Any],
    validate: bool = True,
    submission_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute the full score block for one submission.

    Returns:
        dict with keys:
            scores          – the validated per-criterion scores
            composite_score – weighted composite on the 0-10 scale
            skill_level     – band from EVAL-007 (``weak``/``developing``/``strong``)
            weights         – the weights actually applied, for auditability

    Raises:
        EvaluatorError: From the range guard, on a missing or out-of-range score.
    """
    rubric = load_rubric()

    resolved = (
        enforce_score_range(scores, submission_index=submission_index)
        if validate
        else {criterion: float(scores[criterion]) for criterion in rubric.criteria}
    )

    composite = calculate_composite(
        resolved, validate=False, submission_index=submission_index
    )

    return {
        "scores": resolved,
        "composite_score": composite,
        "skill_level": map_skill_level(composite),
        "weights": dict(rubric.weights),
    }
