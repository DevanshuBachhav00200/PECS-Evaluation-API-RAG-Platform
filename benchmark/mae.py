"""
BENCH-005 — Mean Absolute Error calculation.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

MAE between AI-produced scores and the frozen gold baseline (BENCH-002), both
on the composite score and per criterion.

Implemented in plain Python rather than through numpy/pandas on purpose: the
acceptance criterion is that the computation is verified against a
hand-calculated example, and a five-line formula anyone can check by hand is
worth more here than a vectorised one nobody audits. It also means the metric
layer imports nothing, so it runs in the container, in CI, and in a bare
checkout alike.

    MAE = (1/n) * Σ |ai - gi|

Acceptance (BENCH-005): verified against a hand-calculated example — see
``tests/member6/test_benchmark_metrics.py::test_mae_matches_a_hand_calculation``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from benchmark.test_set_loader import CRITERIA

#: Decimal places metrics are rounded to for reporting. Four keeps a 0.0001
#: difference visible without implying precision the sample size cannot support.
METRIC_PRECISION = 4


class MetricError(ValueError):
    """Raised when a metric is asked for on unusable input."""


def _paired(
    predicted: Sequence[float], reference: Sequence[float]
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if len(predicted) != len(reference):
        raise MetricError(
            f"paired series must be the same length, got {len(predicted)} "
            f"predicted and {len(reference)} reference"
        )
    if not predicted:
        raise MetricError("cannot compute a metric over an empty series")

    try:
        left = tuple(float(value) for value in predicted)
        right = tuple(float(value) for value in reference)
    except (TypeError, ValueError) as exc:
        raise MetricError(f"series must contain numbers only: {exc}") from exc

    return left, right


def mean_absolute_error(
    predicted: Sequence[float], reference: Sequence[float]
) -> float:
    """Mean absolute error between two equal-length numeric series.

    Args:
        predicted: AI-produced values.
        reference: Gold baseline values, in the same order.

    Returns:
        float: MAE, rounded to ``METRIC_PRECISION``.

    Raises:
        MetricError: On mismatched lengths, an empty series, or non-numeric values.
    """
    left, right = _paired(predicted, reference)
    total = sum(abs(a - b) for a, b in zip(left, right))
    return round(total / len(left), METRIC_PRECISION)


def max_absolute_error(
    predicted: Sequence[float], reference: Sequence[float]
) -> float:
    """Largest single absolute error in the pair — the worst case behind an MAE."""
    left, right = _paired(predicted, reference)
    return round(max(abs(a - b) for a, b in zip(left, right)), METRIC_PRECISION)


def bias(predicted: Sequence[float], reference: Sequence[float]) -> float:
    """Mean signed error (predicted - reference).

    MAE alone cannot distinguish "scatters both ways" from "grades every
    submission a point high". A positive bias means the evaluator is systematically
    more generous than the baseline; that is a different defect from noise and is
    reported separately in BENCH-010.
    """
    left, right = _paired(predicted, reference)
    return round(sum(a - b for a, b in zip(left, right)) / len(left), METRIC_PRECISION)


def within_tolerance(
    predicted: Sequence[float], reference: Sequence[float], tolerance: float = 1.0
) -> float:
    """Fraction of pairs whose absolute error is within ``tolerance`` points."""
    left, right = _paired(predicted, reference)
    if tolerance < 0:
        raise MetricError(f"tolerance must be non-negative, got {tolerance}")
    hits = sum(1 for a, b in zip(left, right) if abs(a - b) <= tolerance)
    return round(hits / len(left), METRIC_PRECISION)


def error_series(
    pairs: Iterable[Mapping[str, Any]],
    predicted_key: str,
    reference_key: str,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Extract a predicted/reference pair of series from comparison rows."""
    predicted: list[float] = []
    reference: list[float] = []

    for row in pairs:
        if predicted_key not in row or reference_key not in row:
            continue
        predicted_value = row[predicted_key]
        reference_value = row[reference_key]
        if predicted_value is None or reference_value is None:
            continue
        predicted.append(float(predicted_value))
        reference.append(float(reference_value))

    return tuple(predicted), tuple(reference)


def composite_mae(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """MAE and friends over the composite score across comparison rows.

    Args:
        rows: Comparison rows from BENCH-004, each carrying ``ai_composite``
              and ``gold_composite``.

    Returns:
        dict: ``n``, ``mae``, ``max_error``, ``bias``, ``within_1_point``.
    """
    predicted, reference = error_series(rows, "ai_composite", "gold_composite")

    if not predicted:
        raise MetricError(
            "no comparable composite scores — every row is missing "
            "'ai_composite' or 'gold_composite'"
        )

    return {
        "n": len(predicted),
        "mae": mean_absolute_error(predicted, reference),
        "max_error": max_absolute_error(predicted, reference),
        "bias": bias(predicted, reference),
        "within_1_point": within_tolerance(predicted, reference, 1.0),
    }


def per_criterion_mae(
    rows: Sequence[Mapping[str, Any]],
    criteria: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """MAE per criterion (c1…c6) across comparison rows.

    A composite MAE can look acceptable while one criterion is badly
    miscalibrated — the weights average the error away. This is the breakdown
    that surfaces it.
    """
    resolved = tuple(criteria or CRITERIA)
    results: Dict[str, Dict[str, Any]] = {}

    for criterion in resolved:
        predicted, reference = error_series(
            rows, f"ai_{criterion}", f"gold_{criterion}"
        )
        if not predicted:
            results[criterion] = {"n": 0, "mae": None, "max_error": None, "bias": None}
            continue

        results[criterion] = {
            "n": len(predicted),
            "mae": mean_absolute_error(predicted, reference),
            "max_error": max_absolute_error(predicted, reference),
            "bias": bias(predicted, reference),
        }

    return results
