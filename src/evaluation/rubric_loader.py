"""
EVAL-002 — Rubric loading from a single versioned source.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

One canonical, versioned place to load the rubric from, so that a rubric change
propagates everywhere at once: the system prompt sent to the evaluator, the
weights used to compute the composite score, the bands used to map skill level,
and the version tag stored on the result.

Where each part actually lives (this module composes, it does not restate):

    weights, criteria  -> src/dataset/annotation/annotation_schema.py  (ANN-001)
    bands              -> src/evaluation/skill_level_map.py            (EVAL-007)
    system prompt      -> src/rag/system_prompt.py                     (RAG-011)
    version tag        -> src/evaluation/rubric_version.py             (EVAL-003)

Editing any of those changes scoring behaviour on the next call with no other
code edit — that is the acceptance criterion for this task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

from src.dataset.annotation.annotation_schema import CRITERIA, WEIGHTS
from src.evaluation.error_types import RUBRIC_UNAVAILABLE, EvaluatorError
from src.evaluation.rubric_version import RUBRIC_VERSION, rubric_tag
from src.evaluation.skill_level_map import BAND_RANGES, BANDS

#: Tolerance for the weight-sum sanity check.
_WEIGHT_SUM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Rubric:
    """The complete, versioned rubric the evaluator runs under."""

    version: str
    tag: str
    criteria: Tuple[str, ...]
    weights: Mapping[str, float]
    bands: Tuple[str, ...]
    band_ranges: Mapping[str, Tuple[float, float]]
    system_prompt: str
    criterion_names: Mapping[str, str]

    def weight(self, criterion: str) -> float:
        try:
            return self.weights[criterion]
        except KeyError as exc:
            raise EvaluatorError(
                f"criterion {criterion!r} has no rubric weight",
                code=RUBRIC_UNAVAILABLE,
                details={"criterion": criterion},
            ) from exc

    def to_dict(self) -> Dict[str, object]:
        """Render for API documentation and the handoff package."""
        return {
            "version": self.version,
            "tag": self.tag,
            "criteria": list(self.criteria),
            "weights": dict(self.weights),
            "bands": list(self.bands),
            "band_ranges": {k: list(v) for k, v in self.band_ranges.items()},
            "criterion_names": dict(self.criterion_names),
        }


_cached: Rubric | None = None


def load_rubric() -> Rubric:
    """Load (and cache) the canonical rubric.

    Returns:
        Rubric: The live rubric, validated for completeness.

    Raises:
        EvaluatorError: If the rubric is internally inconsistent — a missing
                        weight, a weight outside 0-1, or weights that do not
                        sum to 1.0. Code ``EVAL_RUBRIC_UNAVAILABLE``.
    """
    global _cached

    if _cached is not None:
        return _cached

    # Imported here rather than at module scope so that loading the rubric does
    # not drag the prompt module into every importer of the evaluation package.
    from src.rag.system_prompt import CRITERION_NAMES, PECS_SYSTEM_PROMPT

    missing = [criterion for criterion in CRITERIA if criterion not in WEIGHTS]
    if missing:
        raise EvaluatorError(
            f"rubric is missing weights for {missing}",
            code=RUBRIC_UNAVAILABLE,
            details={"missing": missing},
        )

    for criterion in CRITERIA:
        weight = WEIGHTS[criterion]
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise EvaluatorError(
                f"weight for {criterion} must be numeric",
                code=RUBRIC_UNAVAILABLE,
                details={"criterion": criterion, "weight": repr(weight)},
            )
        if not 0.0 < float(weight) <= 1.0:
            raise EvaluatorError(
                f"weight for {criterion} must lie in (0, 1], got {weight}",
                code=RUBRIC_UNAVAILABLE,
                details={"criterion": criterion, "weight": weight},
            )

    total = sum(float(WEIGHTS[criterion]) for criterion in CRITERIA)
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise EvaluatorError(
            f"rubric weights must sum to 1.0, got {total}",
            code=RUBRIC_UNAVAILABLE,
            details={"weight_sum": total, "weights": dict(WEIGHTS)},
        )

    _cached = Rubric(
        version=RUBRIC_VERSION,
        tag=rubric_tag(),
        criteria=tuple(CRITERIA),
        weights=dict(WEIGHTS),
        bands=tuple(BANDS),
        band_ranges=dict(BAND_RANGES),
        system_prompt=PECS_SYSTEM_PROMPT,
        criterion_names=dict(CRITERION_NAMES),
    )
    return _cached


def reload_rubric() -> Rubric:
    """Drop the cache and reload. Used by tests and by long-lived services."""
    global _cached
    _cached = None
    return load_rubric()


def get_weights() -> Dict[str, float]:
    """Convenience accessor for the canonical C1-C6 weights."""
    return dict(load_rubric().weights)


def get_system_prompt() -> str:
    """Convenience accessor for the canonical rubric system prompt."""
    return load_rubric().system_prompt
