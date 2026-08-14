"""
Unit tests for the benchmark metric layer (BENCH-005 / BENCH-006).

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

BENCH-005's acceptance criterion is "MAE computation verified against a
hand-calculated example", and BENCH-006's is "correlation function runs
correctly; result interpreted only once sample size is adequate". Both are
asserted here, against arithmetic written out in the test rather than derived
from the implementation.

The known-answer cases matter more than they look. A benchmark's metric layer is
the one piece of code whose output nobody can sanity-check by eye: an MAE of
0.42 looks equally plausible whether the formula is right or subtly wrong. So
every metric is pinned at a value that can be recomputed on paper.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from benchmark.correlation import (
    DEFAULT_MIN_SAMPLE,
    TARGET_CORRELATION,
    correlate,
    pearson,
    spearman,
)
from benchmark.mae import (
    MetricError,
    bias,
    composite_mae,
    max_absolute_error,
    mean_absolute_error,
    per_criterion_mae,
    within_tolerance,
)
from benchmark.perf import percentile
from benchmark.test_set_loader import CRITERIA


def _row(gold: float, ai: float, **criteria: float) -> Dict[str, Any]:
    row: Dict[str, Any] = {"gold_composite": gold, "ai_composite": ai}
    for criterion in CRITERIA:
        row[f"gold_{criterion}"] = criteria.get(f"gold_{criterion}", gold)
        row[f"ai_{criterion}"] = criteria.get(f"ai_{criterion}", ai)
    return row


# ---------------------------------------------------------------------------
# BENCH-005 — MAE, hand-calculated
# ---------------------------------------------------------------------------


def test_mae_matches_a_hand_calculation() -> None:
    """|8-7| + |6-6| + |4-7| + |9-8| = 1 + 0 + 3 + 1 = 5; 5/4 = 1.25."""
    assert mean_absolute_error([8, 6, 4, 9], [7, 6, 7, 8]) == 1.25


def test_mae_of_identical_series_is_zero() -> None:
    assert mean_absolute_error([1, 2, 3], [1, 2, 3]) == 0.0


def test_mae_is_symmetric() -> None:
    assert mean_absolute_error([8, 6], [7, 9]) == mean_absolute_error([7, 9], [8, 6])


def test_mae_ignores_the_sign_of_the_error() -> None:
    """A generous and a harsh evaluator with equal spread share an MAE."""
    assert mean_absolute_error([9, 9], [8, 8]) == mean_absolute_error([7, 7], [8, 8])


def test_max_absolute_error_reports_the_worst_case() -> None:
    assert max_absolute_error([8, 6, 4, 9], [7, 6, 7, 8]) == 3.0


def test_bias_is_signed_and_hand_calculable() -> None:
    """(+1) + 0 + (-3) + (+1) = -1; -1/4 = -0.25."""
    assert bias([8, 6, 4, 9], [7, 6, 7, 8]) == -0.25


def test_bias_separates_systematic_from_scattered_error() -> None:
    """Same MAE, opposite meaning: one is a constant offset, one is noise."""
    systematic_ai, systematic_gold = [8, 8, 8], [7, 7, 7]
    scattered_ai, scattered_gold = [8, 6, 8], [7, 7, 7]

    assert mean_absolute_error(systematic_ai, systematic_gold) == mean_absolute_error(
        scattered_ai, scattered_gold
    )
    assert bias(systematic_ai, systematic_gold) == 1.0
    assert bias(scattered_ai, scattered_gold) == pytest.approx(0.3333, abs=1e-4)


def test_within_tolerance_counts_the_fraction_inside_one_point() -> None:
    """3 of 4 pairs differ by <= 1.0 → 0.75."""
    assert within_tolerance([8, 6, 4, 9], [7, 6, 7, 8], tolerance=1.0) == 0.75


def test_mismatched_series_lengths_are_rejected() -> None:
    with pytest.raises(MetricError):
        mean_absolute_error([1, 2], [1])


def test_empty_series_is_rejected_rather_than_returning_zero() -> None:
    """An MAE of 0 over nothing would read as a perfect score."""
    with pytest.raises(MetricError):
        mean_absolute_error([], [])


# ---------------------------------------------------------------------------
# BENCH-005 — over comparison rows
# ---------------------------------------------------------------------------


def test_composite_mae_over_rows() -> None:
    rows = [_row(8, 7), _row(6, 6), _row(4, 7), _row(9, 8)]
    result = composite_mae(rows)

    assert result["n"] == 4
    assert result["mae"] == 1.25
    assert result["max_error"] == 3.0
    # bias is AI minus gold: (7-8) + (6-6) + (7-4) + (8-9) = 1; 1/4 = +0.25.
    # Positive means the evaluator graded above the baseline on average.
    assert result["bias"] == 0.25
    assert result["within_1_point"] == 0.75


def test_composite_mae_skips_ungraded_rows() -> None:
    """A failed record contributes no score, and must not count as a zero error."""
    rows = [_row(8, 7), {"gold_composite": 6.0, "ai_composite": None}]
    assert composite_mae(rows)["n"] == 1


def test_composite_mae_raises_when_nothing_is_comparable() -> None:
    with pytest.raises(MetricError):
        composite_mae([{"gold_composite": 6.0, "ai_composite": None}])


def test_per_criterion_mae_breaks_the_composite_down() -> None:
    """One badly miscalibrated criterion is visible per-criterion, not in the mean."""
    rows = [
        _row(7, 7, gold_c3=3.0, ai_c3=9.0),
        _row(7, 7, gold_c3=3.0, ai_c3=9.0),
    ]
    result = per_criterion_mae(rows)

    assert result["c1"]["mae"] == 0.0
    assert result["c3"]["mae"] == 6.0
    assert result["c3"]["bias"] == 6.0


def test_per_criterion_mae_reports_absent_criteria_as_zero_n() -> None:
    rows = [{"gold_composite": 7, "ai_composite": 7}]
    assert per_criterion_mae(rows)["c1"] == {
        "n": 0,
        "mae": None,
        "max_error": None,
        "bias": None,
    }


# ---------------------------------------------------------------------------
# BENCH-006 — correlation
# ---------------------------------------------------------------------------


def test_pearson_of_a_perfect_linear_relationship_is_one() -> None:
    assert pearson([1, 2, 3, 4], [2, 4, 6, 8]) == 1.0


def test_pearson_of_a_perfect_inverse_relationship_is_minus_one() -> None:
    assert pearson([1, 2, 3, 4], [8, 6, 4, 2]) == -1.0


def test_pearson_matches_a_hand_calculation() -> None:
    """x=[1,2,3], y=[2,4,7]: cov=5.0, sd terms √2·√12.667 → r ≈ 0.9934."""
    assert pearson([1, 2, 3], [2, 4, 7]) == pytest.approx(0.9934, abs=1e-4)


def test_pearson_is_undefined_not_zero_for_a_constant_series() -> None:
    """An evaluator that returns the same score for everything is broken, not
    uncorrelated — reporting r=0 would understate that."""
    assert pearson([5, 5, 5], [1, 2, 3]) is None


def test_spearman_is_one_for_a_monotonic_but_non_linear_relationship() -> None:
    assert spearman([1, 2, 3, 4], [1, 4, 9, 16]) == 1.0


def test_spearman_averages_tied_ranks() -> None:
    assert spearman([1, 2, 2, 3], [1, 2, 2, 3]) == 1.0


def test_correlation_needs_at_least_two_pairs() -> None:
    result = correlate([_row(7, 7)])
    assert result["pearson"] is None
    assert result["interpretable"] is False


def test_small_sample_is_computed_but_flagged_uninterpretable() -> None:
    """The [Requires Validation] gate: the number exists, the verdict does not."""
    rows = [_row(gold, gold + 0.5) for gold in (4.0, 5.0, 6.0, 7.0)]
    result = correlate(rows)

    assert result["pearson"] == 1.0
    assert result["interpretable"] is False
    assert result["sample_size_status"] == "insufficient_sample"
    assert result["meets_target"] is None
    assert "DATA NEEDED" in result["note"]


def test_adequate_sample_is_interpreted_against_the_target() -> None:
    rows = [_row(4.0 + index * 0.1, 4.0 + index * 0.1) for index in range(DEFAULT_MIN_SAMPLE)]
    result = correlate(rows)

    assert result["interpretable"] is True
    assert result["sample_size_status"] == "adequate"
    assert result["meets_target"] is True
    assert result["target"] == TARGET_CORRELATION


def test_adequate_sample_below_target_is_reported_as_failing() -> None:
    """The gate must be able to say no, not only yes."""
    golds = [4.0 + index * 0.2 for index in range(DEFAULT_MIN_SAMPLE)]
    ais = [10.0 - value for value in golds]  # perfectly inverse
    rows = [_row(gold, ai) for gold, ai in zip(golds, ais)]

    result = correlate(rows)
    assert result["interpretable"] is True
    assert result["meets_target"] is False


# ---------------------------------------------------------------------------
# BENCH-009 — percentiles
# ---------------------------------------------------------------------------


def test_percentile_uses_nearest_rank() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 0.95) == 5.0
    assert percentile(values, 1.0) == 5.0


def test_percentile_of_an_empty_sample_is_none() -> None:
    assert percentile([], 0.5) is None


def test_percentile_rejects_an_out_of_range_fraction() -> None:
    with pytest.raises(ValueError):
        percentile([1.0], 1.5)
