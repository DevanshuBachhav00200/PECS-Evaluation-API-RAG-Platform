"""
EVAL-004 — C1-C6 score-range validation.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Rejects (or, if configured, clamps) any parsed evaluator score outside the valid
1-10 per-criterion range, before scoring proceeds. A mocked out-of-range score
such as ``c2 = 15`` is caught here, not silently folded into a composite.

``reject`` is the default. Clamping is available for operators who would rather
degrade than fail a whole batch, but it hides a real signal — an out-of-range
score usually means the evaluator misread the rubric, which is exactly what the
benchmark (BENCH-004) needs to see.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from src.evaluation.error_types import (
    MISSING_CRITERION,
    SCORE_OUT_OF_RANGE,
    EvaluatorError,
)
from src.evaluation.rubric_loader import load_rubric
from src.rag.config import SCORE_RANGE_MODE

#: Inclusive per-criterion score bounds.
MIN_SCORE = 1.0
MAX_SCORE = 10.0

MODE_REJECT = "reject"
MODE_CLAMP = "clamp"
VALID_MODES = frozenset({MODE_REJECT, MODE_CLAMP})


def enforce_score_range(
    scores: Mapping[str, Any],
    mode: Optional[str] = None,
    submission_index: Optional[int] = None,
) -> Dict[str, float]:
    """Validate one submission's C1-C6 scores.

    Args:
        scores:           Mapping of criterion -> score.
        mode:             ``"reject"`` (default) or ``"clamp"``. Defaults to
                          ``config.SCORE_RANGE_MODE``.
        submission_index: Included in error details so a failing batch names the
                          offending submission.

    Returns:
        dict[str, float]: All six criteria as floats, in rubric order, within
                          1-10.

    Raises:
        EvaluatorError: ``EVAL_MISSING_CRITERION`` when a criterion is absent,
                        ``EVAL_SCORE_OUT_OF_RANGE`` when a value is non-numeric
                        or (in reject mode) outside 1-10.
        ValueError:     If ``mode`` is not a recognised mode.
    """
    effective_mode = SCORE_RANGE_MODE if mode is None else mode
    if effective_mode not in VALID_MODES:
        raise ValueError(
            f"Unknown score-range mode {effective_mode!r}; "
            f"must be one of {sorted(VALID_MODES)}"
        )

    if not isinstance(scores, Mapping):
        raise EvaluatorError(
            f"scores must be a mapping, got {type(scores).__name__}",
            code=SCORE_OUT_OF_RANGE,
            details={"submission_index": submission_index},
        )

    criteria = load_rubric().criteria
    validated: Dict[str, float] = {}

    for criterion in criteria:
        if criterion not in scores:
            raise EvaluatorError(
                f"missing score for {criterion.upper()}",
                code=MISSING_CRITERION,
                details={
                    "submission_index": submission_index,
                    "criterion": criterion,
                },
            )

        value = scores[criterion]

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvaluatorError(
                f"score for {criterion.upper()} must be numeric, got "
                f"{type(value).__name__}",
                code=SCORE_OUT_OF_RANGE,
                details={
                    "submission_index": submission_index,
                    "criterion": criterion,
                    "value": repr(value),
                },
            )

        numeric = float(value)

        if numeric < MIN_SCORE or numeric > MAX_SCORE:
            if effective_mode == MODE_REJECT:
                raise EvaluatorError(
                    f"score for {criterion.upper()} must lie in "
                    f"{MIN_SCORE:.0f}-{MAX_SCORE:.0f}, got {numeric:g}",
                    code=SCORE_OUT_OF_RANGE,
                    details={
                        "submission_index": submission_index,
                        "criterion": criterion,
                        "value": numeric,
                        "min": MIN_SCORE,
                        "max": MAX_SCORE,
                    },
                )
            numeric = min(max(numeric, MIN_SCORE), MAX_SCORE)

        validated[criterion] = numeric

    return validated
