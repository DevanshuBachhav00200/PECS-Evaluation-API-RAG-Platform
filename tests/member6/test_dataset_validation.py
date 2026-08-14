"""
TEST-001 — Dataset-validation unit tests.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

The tracker scopes this task as "schema validation, enum validation, and
quarantine logic from Workstream B". Workstream B's standalone modules
(``schema.py``, ``validate_schema.py``, ``enums.py``, ``quarantine.py``) are not
present in the repository — see ``docs/known_issues_and_limitations.md``. What
does exist, and what actually guards the dataset that reaches the vector store,
is:

* ``src/dataset/annotation/validation.py`` — score-range, formula and
  skill-level validation over the annotation frame, plus the split-leakage check;
* ``src/dataset/annotation/scoring.py`` — the canonical weighted score and band
  mapping every layer downstream defers to;
* ``src/vectorstore/metadata_schema.py`` — the enum validation that actually
  rejects an out-of-enum domain or level at ingestion time;
* ``src/common/normalization.py`` — DS-005 normalization.

These tests cover that surface, with at least one pass and one fail case per
validator, as the acceptance criterion requires. Where a Workstream B module is
missing, the gap is recorded in the known-issues document rather than papered
over with a test of something else.
"""

from __future__ import annotations

import pandas as pd
import pytest

from benchmark.test_set_loader import REPO_ROOT
from src.common.normalization import normalize_text
from src.dataset.annotation.annotation_schema import CRITERIA
from src.dataset.annotation.normalize import normalize_score
from src.dataset.annotation.scoring import skill_level, weighted_score
from src.dataset.annotation.validation import leakage_report, validate_annotation_frame
from src.vectorstore.metadata_schema import (
    VALID_DOMAINS,
    VALID_LEVELS,
    build_metadata,
    validate_metadata,
)

VALID_SCORES = {"c1": 8.0, "c2": 7.0, "c3": 6.0, "c4": 7.0, "c5": 9.0, "c6": 5.0}


def _frame(**overrides) -> pd.DataFrame:
    row = {
        "prompt_id": "PELS-L1-CODE-001",
        "skill_level": "strong",
        "domain": "CODE",
        "prompt_text": "Review this endpoint and list issues by severity.",
        "final_score": 7.05,
        "dataset_split": "train",
        **{f"{criterion}_score": VALID_SCORES[criterion] for criterion in CRITERIA},
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Score-range validation (DS-010 / ANN-004)
# ---------------------------------------------------------------------------


def test_in_range_scores_are_accepted() -> None:
    """A record with every criterion inside 1-10 raises no range error."""
    result = validate_annotation_frame(_frame())
    assert not bool(result.loc[0, "score_range_error"])


@pytest.mark.parametrize("bad_value", [0, 11, -1, 10.5])
def test_out_of_range_score_is_flagged(bad_value: float) -> None:
    """A score outside 1-10 is flagged rather than silently accepted."""
    result = validate_annotation_frame(_frame(c2_score=bad_value))
    assert bool(result.loc[0, "score_range_error"])


def test_non_numeric_score_is_flagged_not_crashed() -> None:
    """A text score is coerced to NaN and flagged, not raised as a TypeError."""
    result = validate_annotation_frame(_frame(c3_score="not-a-number"))
    assert bool(result.loc[0, "score_range_error"])


def test_missing_required_column_is_rejected() -> None:
    """A frame missing a required column fails loudly at validation time."""
    frame = _frame().drop(columns=["c5_score"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_annotation_frame(frame)


# ---------------------------------------------------------------------------
# Composite formula and band mapping (ANN-004 / ANN-005 / EVAL-006 / EVAL-007)
# ---------------------------------------------------------------------------


def test_composite_matches_hand_calculation() -> None:
    """8/7/6/7/9/5 with weights 15/20/20/20/15/10 is 7.05, hand-computed.

    1.20 + 1.40 + 1.20 + 1.40 + 1.35 + 0.50 = 7.05
    """
    assert weighted_score(VALID_SCORES) == 7.05


def test_composite_is_not_multiplied_by_ten() -> None:
    """All-10s must be 10.0, not 100 — the documented ×10 scoring error."""
    assert weighted_score({criterion: 10 for criterion in CRITERIA}) == 10.0


def test_recorded_final_score_mismatch_is_flagged() -> None:
    """A final_score that disagrees with the recomputed composite is flagged."""
    result = validate_annotation_frame(_frame(final_score=9.99))
    assert bool(result.loc[0, "final_score_error"])


def test_recorded_final_score_agreement_is_not_flagged() -> None:
    result = validate_annotation_frame(_frame(final_score=7.05))
    assert not bool(result.loc[0, "final_score_error"])


@pytest.mark.parametrize(
    "score,expected",
    [(0.0, "weak"), (3.99, "weak"), (4.0, "developing"), (6.99, "developing"),
     (7.0, "strong"), (10.0, "strong")],
)
def test_skill_band_boundaries(score: float, expected: str) -> None:
    """The three bands are exact at their documented boundaries."""
    assert skill_level(score) == expected


def test_skill_level_mismatch_is_flagged() -> None:
    """A record labelled 'weak' whose scores compute to 'strong' is flagged."""
    result = validate_annotation_frame(_frame(skill_level="weak"))
    assert bool(result.loc[0, "skill_level_mismatch"])


def test_composite_out_of_scale_is_rejected() -> None:
    with pytest.raises(ValueError):
        skill_level(76.5)


# ---------------------------------------------------------------------------
# Score normalization (ANN-004)
# ---------------------------------------------------------------------------


def test_normalize_score_accepts_a_valid_value() -> None:
    assert normalize_score("8") == 8.0


@pytest.mark.parametrize("bad", [0, 11, "abc", None])
def test_normalize_score_rejects_invalid_values(bad) -> None:
    with pytest.raises((ValueError, TypeError)):
        normalize_score(bad)


# ---------------------------------------------------------------------------
# Enum validation (DS-004, enforced at the vector-store boundary by VDB-004)
# ---------------------------------------------------------------------------


def test_valid_domain_and_level_are_accepted() -> None:
    metadata = build_metadata(
        domain="CODE", level="strong", scores=VALID_SCORES, source="train",
        version="pecs-v1.0",
    )
    validate_metadata(metadata)  # must not raise
    assert set(metadata) == {"domain", "level", "source", "version", *CRITERIA}


def test_out_of_enum_domain_is_rejected_with_a_clear_message() -> None:
    with pytest.raises(ValueError, match="Invalid domain"):
        build_metadata(
            domain="FINANCE", level="strong", scores=VALID_SCORES, source="train",
            version="pecs-v1.0",
        )


def test_out_of_enum_level_is_rejected_with_a_clear_message() -> None:
    with pytest.raises(ValueError, match="Invalid level"):
        build_metadata(
            domain="CODE", level="expert", scores=VALID_SCORES, source="train",
            version="pecs-v1.0",
        )


def test_metadata_enum_covers_only_the_two_ingested_domains() -> None:
    """Documents the enum's current reach — four master domains are outside it.

    ``data/reports/quality_summary.json`` lists six domains in the master
    dataset. VDB-004 admits two. This is asserted so that widening the enum is a
    deliberate change with a failing test attached, not a silent one.
    """
    assert VALID_DOMAINS == {"CODE", "EDGE"}
    assert VALID_LEVELS == {"weak", "developing", "strong"}


def test_extra_metadata_keys_are_rejected() -> None:
    """VDB-004 admits exactly ten keys — an extra one is an error, not ignored."""
    metadata = build_metadata(
        domain="CODE", level="strong", scores=VALID_SCORES, source="train",
        version="pecs-v1.0",
    )
    metadata["prompt_id"] = "PELS-L1-CODE-001"
    with pytest.raises(ValueError, match="Unexpected/extra metadata keys"):
        validate_metadata(metadata)


# ---------------------------------------------------------------------------
# Normalization (DS-005) — quarantine's upstream neighbour
# ---------------------------------------------------------------------------


def test_normalization_is_idempotent() -> None:
    """Normalizing twice equals normalizing once (DS-005's round-trip criterion)."""
    raw = "  Review   THIS\tEndpoint\n\nplease  "
    once = normalize_text(raw)
    assert normalize_text(once) == once


def test_normalization_collapses_whitespace_and_case() -> None:
    assert normalize_text("  Hello   WORLD  ") == "hello world"


# ---------------------------------------------------------------------------
# Split leakage (DS-014 / ANN-011) — the check that protects the benchmark
# ---------------------------------------------------------------------------


def _split_frame(rows: list[dict]) -> pd.DataFrame:
    """Frame carrying the ``normalized_prompt`` column leakage_report reads."""
    frame = pd.DataFrame(rows)
    frame["normalized_prompt"] = frame["prompt_text"].map(normalize_text)
    return frame


def _row(report: pd.DataFrame, split_a: str, split_b: str) -> pd.Series:
    match = report[(report.split_a == split_a) & (report.split_b == split_b)]
    assert len(match) == 1, f"expected one {split_a}->{split_b} row, got {len(match)}"
    return match.iloc[0]


def test_leakage_report_flags_a_duplicate_across_splits() -> None:
    """The same prompt in train and test is reported as leakage.

    The report always returns one row per split pair; the finding is carried by
    ``leakage_free_at_threshold``, not by the row count.
    """
    shared = "Review this endpoint and list issues by severity."
    report = leakage_report(
        _split_frame(
            [
                {"prompt_id": "A", "prompt_text": shared, "dataset_split": "train"},
                {"prompt_id": "B", "prompt_text": shared, "dataset_split": "test"},
            ]
        )
    )

    train_test = _row(report, "train", "test")
    assert train_test.exact_prompt_text_overlap_count == 1
    assert not bool(train_test.leakage_free_at_threshold)


def test_leakage_report_is_clean_for_distinct_prompts() -> None:
    report = leakage_report(
        _split_frame(
            [
                {
                    "prompt_id": "A",
                    "prompt_text": "Draft a migration plan for a Postgres upgrade.",
                    "dataset_split": "train",
                },
                {
                    "prompt_id": "B",
                    "prompt_text": "Write a haiku about the sea in winter.",
                    "dataset_split": "test",
                },
            ]
        )
    )

    train_test = _row(report, "train", "test")
    assert train_test.exact_prompt_text_overlap_count == 0
    assert train_test.near_duplicate_count == 0
    assert bool(train_test.leakage_free_at_threshold)


def test_leakage_report_catches_a_near_duplicate_below_exact_match() -> None:
    """A reworded near-duplicate crossing a split boundary is still caught."""
    report = leakage_report(
        _split_frame(
            [
                {
                    "prompt_id": "A",
                    "prompt_text": "Review this endpoint and list issues by severity.",
                    "dataset_split": "train",
                },
                {
                    "prompt_id": "B",
                    "prompt_text": "Review this endpoint and list issues by severity!",
                    "dataset_split": "test",
                },
            ]
        )
    )

    train_test = _row(report, "train", "test")
    assert train_test.exact_prompt_text_overlap_count == 0
    assert train_test.near_duplicate_count == 1
    assert not bool(train_test.leakage_free_at_threshold)


def test_gold_eval_set_is_leakage_free_against_the_training_split() -> None:
    """The shipped splits satisfy ANN-011's zero-overlap acceptance criterion.

    Asserted against the real artifacts, not a fixture: this is the property the
    entire benchmark depends on, and a regression here invalidates every figure
    in the benchmark report.
    """
    report = pd.read_csv(REPO_ROOT / "data" / "reports" / "leakage_report.csv")
    train_test = _row(report, "train", "test")
    assert bool(train_test.leakage_free_at_threshold)


def test_leakage_report_requires_the_split_column() -> None:
    frame = pd.DataFrame([{"prompt_id": "A", "prompt_text": "x"}])
    with pytest.raises(ValueError, match="dataset_split"):
        leakage_report(frame)
