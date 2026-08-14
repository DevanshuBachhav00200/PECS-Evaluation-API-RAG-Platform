"""
TEST-005 — Scoring and skill-level unit tests.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Covers ``calculate_scores`` (EVAL-006) and skill-level mapping (EVAL-007)
against hand-computed values, across all three bands.

Acceptance (TEST-005): all three skill-level boundaries are covered by a test
case. Both edges of every band are asserted below — 3.99/4.00 and 6.99/7.00 —
because an off-by-one comparison in the band logic only shows up at the edge,
and a certification band is exactly the kind of output where being one hundredth
of a point out changes a candidate's result.

The composite arithmetic is checked against arithmetic written out longhand in
the test, not against the implementation's own helper. A test that computes the
expected value the same way the code does cannot catch a wrong formula.
"""

from __future__ import annotations

from typing import Dict

import pytest

from src.dataset.annotation.scoring import weighted_score
from src.evaluation.calculate_scores import calculate_composite, calculate_scores
from src.evaluation.error_types import EvaluatorError
from src.evaluation.rubric_loader import load_rubric
from src.evaluation.skill_level_map import BAND_RANGES, map_skill_level

CRITERIA = ("c1", "c2", "c3", "c4", "c5", "c6")

#: The canonical weights (15/20/20/20/15/10), written out so the expected
#: values below are independent of what the rubric loader returns.
WEIGHTS = {"c1": 0.15, "c2": 0.20, "c3": 0.20, "c4": 0.20, "c5": 0.15, "c6": 0.10}


def _by_hand(scores: Dict[str, float]) -> float:
    return round(sum(scores[key] * WEIGHTS[key] for key in CRITERIA), 2)


# ---------------------------------------------------------------------------
# EVAL-006 — the weighted composite
# ---------------------------------------------------------------------------


def test_rubric_weights_are_the_documented_ones() -> None:
    rubric = load_rubric()
    assert {key: round(float(value), 2) for key, value in rubric.weights.items()} == WEIGHTS


def test_weights_sum_to_one() -> None:
    """If they did not, the composite would leave the 0-10 scale silently."""
    assert round(sum(WEIGHTS.values()), 6) == 1.0


def test_composite_matches_a_hand_calculation() -> None:
    """8/7/6/7/9/5 → 1.20+1.40+1.20+1.40+1.35+0.50 = 7.05."""
    scores = {"c1": 8, "c2": 7, "c3": 6, "c4": 7, "c5": 9, "c6": 5}
    assert calculate_composite(scores) == 7.05
    assert calculate_composite(scores) == _by_hand(scores)


def test_composite_has_no_times_ten_multiplication() -> None:
    """All-10s must be 10.0. 100 is the documented scoring bug this guards."""
    assert calculate_composite({key: 10 for key in CRITERIA}) == 10.0


def test_composite_floor_is_the_minimum_valid_score() -> None:
    """All-1s is 1.0 — the composite never dips below the criterion floor."""
    assert calculate_composite({key: 1 for key in CRITERIA}) == 1.0


def test_composite_agrees_with_member_2s_canonical_formula() -> None:
    """EVAL-006 and ANN-004 must never disagree; they grade the same records."""
    scores = {"c1": 6, "c2": 8, "c3": 5, "c4": 9, "c5": 7, "c6": 4}
    assert calculate_composite(scores) == weighted_score(scores)


@pytest.mark.parametrize(
    "scores,expected",
    [
        # 1.35+1.60+1.60+1.80+1.35+0.70
        ({"c1": 9, "c2": 8, "c3": 8, "c4": 9, "c5": 9, "c6": 7}, 8.40),
        # 0.75+1.20+1.00+1.20+1.05+0.40
        ({"c1": 5, "c2": 6, "c3": 5, "c4": 6, "c5": 7, "c6": 4}, 5.60),
        # 0.30+0.40+0.60+0.60+0.60+0.20
        ({"c1": 2, "c2": 2, "c3": 3, "c4": 3, "c5": 4, "c6": 2}, 2.70),
    ],
)
def test_composite_across_all_three_bands(scores: Dict[str, float], expected: float) -> None:
    assert calculate_composite(scores) == expected == _by_hand(scores)


# ---------------------------------------------------------------------------
# EVAL-004 — range enforcement feeding the composite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, 11, 15, -3])
def test_out_of_range_score_is_rejected_before_scoring(bad: float) -> None:
    with pytest.raises(EvaluatorError):
        calculate_scores({**{key: 7 for key in CRITERIA}, "c2": bad})


def test_missing_criterion_is_rejected() -> None:
    partial = {key: 7 for key in CRITERIA if key != "c5"}
    with pytest.raises(EvaluatorError):
        calculate_scores(partial)


def test_boundary_scores_one_and_ten_are_accepted() -> None:
    calculate_scores({key: 1 for key in CRITERIA})
    calculate_scores({key: 10 for key in CRITERIA})


# ---------------------------------------------------------------------------
# EVAL-007 — skill-level bands, both edges of each
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "composite,band",
    [
        (0.00, "weak"),
        (3.98, "weak"),
        (3.99, "weak"),
        (4.00, "developing"),
        (4.01, "developing"),
        (6.98, "developing"),
        (6.99, "developing"),
        (7.00, "strong"),
        (7.01, "strong"),
        (10.00, "strong"),
    ],
)
def test_band_boundaries_map_correctly(composite: float, band: str) -> None:
    assert map_skill_level(composite) == band


def test_declared_band_ranges_match_the_mapping() -> None:
    """The documented ranges and the live mapping must not drift apart."""
    for band, (low, high) in BAND_RANGES.items():
        assert map_skill_level(low) == band
        assert map_skill_level(high) == band


def test_band_mapping_rejects_a_score_off_the_scale() -> None:
    """A composite of 76.5 means the ×10 error is back; it must not map."""
    with pytest.raises(ValueError):
        map_skill_level(76.5)


def test_band_mapping_rejects_a_non_numeric_score() -> None:
    with pytest.raises(TypeError):
        map_skill_level("7.0")


# ---------------------------------------------------------------------------
# calculate_scores — the composed block
# ---------------------------------------------------------------------------


def test_score_block_carries_scores_composite_band_and_weights() -> None:
    result = calculate_scores({"c1": 8, "c2": 7, "c3": 6, "c4": 7, "c5": 9, "c6": 5})

    assert result["composite_score"] == 7.05
    assert result["skill_level"] == "strong"
    assert set(result["scores"]) == set(CRITERIA)
    assert {key: round(float(value), 2) for key, value in result["weights"].items()} == WEIGHTS


def test_score_block_normalises_scores_to_floats() -> None:
    result = calculate_scores({key: 7 for key in CRITERIA})
    assert all(isinstance(value, float) for value in result["scores"].values())


def test_band_agrees_with_the_composite_it_was_derived_from() -> None:
    """Guards against a block whose band and composite come from different paths."""
    for scores in (
        {"c1": 9, "c2": 8, "c3": 8, "c4": 9, "c5": 9, "c6": 7},
        {"c1": 5, "c2": 6, "c3": 5, "c4": 6, "c5": 7, "c6": 4},
        {"c1": 2, "c2": 2, "c3": 3, "c4": 3, "c5": 4, "c6": 2},
    ):
        result = calculate_scores(scores)
        assert result["skill_level"] == map_skill_level(result["composite_score"])
