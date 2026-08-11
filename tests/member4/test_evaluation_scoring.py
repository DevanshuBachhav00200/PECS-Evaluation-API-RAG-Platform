"""
Tests for EVAL-001 … EVAL-013: input schema, rubric loading and versioning,
score-range enforcement, justification capping, the weighted composite,
skill-level mapping, C5 ethics handling, batch summary, confidence handling,
error classification, and model-version tracking.
"""

from __future__ import annotations

import pytest

from src.dataset.annotation.scoring import weighted_score as ann_weighted_score
from src.evaluation.calculate_scores import calculate_composite, calculate_scores
from src.evaluation.confidence import (
    STATUS_NOT_IMPLEMENTED,
    STATUS_PRESENT,
    STATUS_UNPARSEABLE,
    extract_confidence,
    parse_confidence,
)
from src.evaluation.error_types import (
    MISSING_CRITERION,
    SCORE_OUT_OF_RANGE,
    EvaluatorError,
    RagError,
    is_error,
    layer_for_code,
    structured_error,
)
from src.evaluation.ethics_gate import (
    MODE_HARD_GATE,
    MODE_WEIGHTED_ONLY,
    apply_ethics_gate,
)
from src.evaluation.justification import enforce_word_cap, extract_justifications
from src.evaluation.model_version import resolve_model_version, tag_with_model_version
from src.evaluation.overall_summary import build_overall_summary, criterion_averages
from src.evaluation.rubric_loader import load_rubric, reload_rubric
from src.evaluation.rubric_version import RUBRIC_VERSION, rubric_tag, tag_with_rubric_version
from src.evaluation.schema import EvaluationInput, normalize_qa_pair
from src.evaluation.score_range_guard import MODE_CLAMP, enforce_score_range
from src.evaluation.skill_level_map import BAND_RANGES, map_skill_level
from src.rag import config

from m4_fakes import FakeResponse

CRITERIA = ("c1", "c2", "c3", "c4", "c5", "c6")
GOOD_SCORES = {"c1": 8, "c2": 7, "c3": 6, "c4": 7, "c5": 9, "c6": 5}


# --------------------------------------------------------------------------- #
# EVAL-001 — evaluation input schema                                          #
# --------------------------------------------------------------------------- #


def test_evaluation_input_round_trips() -> None:
    item = EvaluationInput(
        submission_index=1,
        task="Task",
        student_prompt="Prompt",
        calibration="Scores: ...",
    )
    assert item.to_dict()["submission_index"] == 1
    assert item.calibration_available is False


def test_evaluation_input_rejects_empty_fields() -> None:
    with pytest.raises(EvaluatorError):
        EvaluationInput(
            submission_index=1, task="   ", student_prompt="p", calibration="c"
        )


def test_evaluation_input_rejects_zero_index() -> None:
    with pytest.raises(EvaluatorError):
        EvaluationInput(
            submission_index=0, task="t", student_prompt="p", calibration="c"
        )


@pytest.mark.parametrize(
    "pair",
    [
        {"task": "T", "student_prompt": "P"},
        {"task_description": "T", "prompt": "P"},
        {"question": "T", "answer": "P"},
        ("T", "P"),
    ],
)
def test_normalize_qa_pair_accepts_the_documented_shapes(pair) -> None:
    normalized = normalize_qa_pair(pair, 1)
    assert normalized["task"] == "T"
    assert normalized["student_prompt"] == "P"


def test_normalize_qa_pair_rejects_unknown_shape() -> None:
    with pytest.raises(EvaluatorError):
        normalize_qa_pair(["only-one"], 1)


def test_schema_matches_what_build_batch_content_produces() -> None:
    """EVAL-001 acceptance: the schema matches what build_batch_content produces.

    Asserted structurally rather than by eye: the assembler's output is compared
    field-by-field against the declared dataclass, so adding a field to one
    without the other fails here.
    """
    from dataclasses import fields

    from src.rag.build_batch_content import build_batch_items

    items = build_batch_items(
        [{"task": "Task", "student_prompt": "Prompt", "domain": "CODE"}],
        retrieve_calibration=False,
    )

    assert len(items) == 1
    item = items[0]
    assert isinstance(item, EvaluationInput)

    declared = {f.name for f in fields(EvaluationInput)}
    assert set(item.to_dict()) == declared

    # Every field the evaluator layer reads is populated by the assembler.
    assert item.submission_index == 1
    assert item.task == "Task"
    assert item.student_prompt == "Prompt"
    assert item.domain == "CODE"
    assert item.calibration  # placeholder text when retrieval is skipped
    assert item.calibration_available is False
    assert item.truncated is False


# --------------------------------------------------------------------------- #
# EVAL-002 / EVAL-003 — rubric loading and versioning                          #
# --------------------------------------------------------------------------- #


def test_rubric_is_loaded_from_one_place_and_is_consistent() -> None:
    rubric = reload_rubric()

    assert rubric.criteria == CRITERIA
    assert pytest.approx(sum(rubric.weights.values())) == 1.0
    assert rubric.weights == {
        "c1": 0.15,
        "c2": 0.20,
        "c3": 0.20,
        "c4": 0.20,
        "c5": 0.15,
        "c6": 0.10,
    }
    assert rubric.version == RUBRIC_VERSION
    assert rubric.system_prompt.startswith("You are the PECS evaluator.")


def test_rubric_is_cached_but_reloadable() -> None:
    assert load_rubric() is load_rubric()
    assert reload_rubric() is load_rubric()


def test_editing_the_rubric_weights_changes_scoring_with_no_other_edit(
    monkeypatch,
) -> None:
    """EVAL-002 acceptance: editing the rubric changes scoring on the next call.

    The weights live in exactly one place (Member 2's ANN-001 ``WEIGHTS``) and
    are read through the loader, so mutating that single mapping and reloading is
    enough to change the composite — no edit to calculate_scores is involved.
    """
    from src.dataset.annotation.annotation_schema import WEIGHTS

    baseline = calculate_composite(GOOD_SCORES)
    assert baseline == 7.05

    try:
        # Shift weight onto C1 (which scores 8 in this fixture); still sums to 1.
        monkeypatch.setitem(WEIGHTS, "c1", 0.50)
        monkeypatch.setitem(WEIGHTS, "c2", 0.10)
        monkeypatch.setitem(WEIGHTS, "c3", 0.10)
        monkeypatch.setitem(WEIGHTS, "c4", 0.10)
        monkeypatch.setitem(WEIGHTS, "c5", 0.10)
        monkeypatch.setitem(WEIGHTS, "c6", 0.10)
        reload_rubric()

        shifted = calculate_composite(GOOD_SCORES)
        assert shifted != baseline
        # 8*.50 + 7*.10 + 6*.10 + 7*.10 + 9*.10 + 5*.10 = 4.00 + 3.40 = 7.40
        assert shifted == 7.40
    finally:
        # Restore the weights BEFORE dropping the cache, or the reloaded rubric
        # would cache the shifted copy and leak into every later test.
        monkeypatch.undo()
        reload_rubric()

    assert calculate_composite(GOOD_SCORES) == baseline


def test_rubric_loader_rejects_weights_that_do_not_sum_to_one(monkeypatch) -> None:
    """A miskeyed weight is a configuration error, not a silent rescale."""
    from src.dataset.annotation.annotation_schema import WEIGHTS

    monkeypatch.setitem(WEIGHTS, "c1", 0.95)
    try:
        with pytest.raises(EvaluatorError) as exc_info:
            reload_rubric()
        assert exc_info.value.code == "EVAL_RUBRIC_UNAVAILABLE"
    finally:
        monkeypatch.undo()
        reload_rubric()


def test_rubric_tag_pairs_version_with_a_content_fingerprint() -> None:
    tag = rubric_tag()
    assert tag.startswith(f"{RUBRIC_VERSION}+")
    assert len(tag.split("+")[1]) == 12


def test_results_are_tagged_with_the_rubric_version() -> None:
    result = tag_with_rubric_version({})
    assert result["rubric_version"] == RUBRIC_VERSION
    assert result["rubric_tag"] == rubric_tag()


def test_rubric_tag_changes_when_the_rubric_text_changes(monkeypatch) -> None:
    """EVAL-003: an edit to the rubric changes the tag stored on new records."""
    baseline = rubric_tag()
    monkeypatch.setattr(
        "src.rag.system_prompt.PECS_SYSTEM_PROMPT",
        "You are the PECS evaluator. (edited)",
    )
    assert rubric_tag() != baseline


# --------------------------------------------------------------------------- #
# EVAL-004 — score-range validation                                            #
# --------------------------------------------------------------------------- #


def test_out_of_range_score_is_caught_before_scoring() -> None:
    """EVAL-004 acceptance: a mocked c2=15 is caught before scoring proceeds."""
    scores = {**GOOD_SCORES, "c2": 15}

    with pytest.raises(EvaluatorError) as exc_info:
        enforce_score_range(scores, submission_index=2)

    error = exc_info.value
    assert error.code == SCORE_OUT_OF_RANGE
    assert error.details["criterion"] == "c2"
    assert error.details["value"] == 15.0
    assert error.details["submission_index"] == 2
    assert error.to_dict()["error"]["layer"] == "evaluation"


def test_clamp_mode_pins_into_range() -> None:
    clamped = enforce_score_range({**GOOD_SCORES, "c2": 15, "c3": 0}, mode=MODE_CLAMP)
    assert clamped["c2"] == 10.0
    assert clamped["c3"] == 1.0


def test_missing_criterion_has_its_own_code() -> None:
    partial = {k: v for k, v in GOOD_SCORES.items() if k != "c5"}
    with pytest.raises(EvaluatorError) as exc_info:
        enforce_score_range(partial)
    assert exc_info.value.code == MISSING_CRITERION


def test_boolean_is_not_a_valid_score() -> None:
    with pytest.raises(EvaluatorError):
        enforce_score_range({**GOOD_SCORES, "c1": True})


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        enforce_score_range(GOOD_SCORES, mode="round")


# --------------------------------------------------------------------------- #
# EVAL-006 — weighted composite, hand-calculated                               #
# --------------------------------------------------------------------------- #


def test_composite_matches_a_hand_calculation() -> None:
    """EVAL-006 acceptance: a hand-calculated case matches exactly.

    8*0.15 + 7*0.20 + 6*0.20 + 7*0.20 + 9*0.15 + 5*0.10
      = 1.20 + 1.40 + 1.20 + 1.40 + 1.35 + 0.50 = 7.05
    """
    assert calculate_composite(GOOD_SCORES) == 7.05


def test_composite_has_no_times_ten_multiplication() -> None:
    """All-10s must give 10.0, not 100 — the documented scoring error."""
    assert calculate_composite({c: 10 for c in CRITERIA}) == 10.0
    assert calculate_composite({c: 1 for c in CRITERIA}) == 1.0


def test_composite_matches_member2_ann004() -> None:
    """Our composite and Member 2's annotation-side weighted score agree."""
    assert calculate_composite(GOOD_SCORES) == ann_weighted_score(GOOD_SCORES)


def test_calculate_scores_returns_the_full_block() -> None:
    block = calculate_scores(GOOD_SCORES)
    assert block["composite_score"] == 7.05
    assert block["skill_level"] == "strong"
    assert block["scores"]["c1"] == 8.0
    assert pytest.approx(sum(block["weights"].values())) == 1.0


def test_calculate_scores_rejects_bad_input() -> None:
    with pytest.raises(EvaluatorError):
        calculate_scores({**GOOD_SCORES, "c6": 42})


# --------------------------------------------------------------------------- #
# EVAL-007 — skill-level mapping, including boundaries                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "score, band",
    [
        (0.0, "weak"),
        (3.99, "weak"),
        (4.0, "developing"),
        (6.99, "developing"),
        (7.0, "strong"),
        (10.0, "strong"),
    ],
)
def test_boundary_values_map_to_the_correct_band(score: float, band: str) -> None:
    """EVAL-007 acceptance: boundary values map correctly, matching ANN-005."""
    assert map_skill_level(score) == band


def test_documented_band_ranges_agree_with_the_mapping() -> None:
    for band, (low, high) in BAND_RANGES.items():
        assert map_skill_level(low) == band
        assert map_skill_level(high) == band


def test_out_of_range_composite_is_rejected() -> None:
    with pytest.raises(ValueError):
        map_skill_level(10.5)


# --------------------------------------------------------------------------- #
# EVAL-008 — C5 ethics handling                                                #
# --------------------------------------------------------------------------- #


def test_default_build_treats_c5_as_weighted_only() -> None:
    """This build's confirmed-pending default: no hard-gate veto."""
    assert config.ETHICS_HARD_GATE_ENABLED is False

    failing_c5 = {**GOOD_SCORES, "c5": 1}
    composite = calculate_composite(failing_c5)
    gated = apply_ethics_gate(failing_c5, composite)

    assert gated["ethics_gate"]["mode"] == MODE_WEIGHTED_ONLY
    assert gated["ethics_gate"]["triggered"] is False
    assert gated["composite_score"] == composite
    assert gated["ethics_gate"]["policy_confirmation"].startswith("pending")


def test_hard_gate_caps_a_failing_c5_when_enabled() -> None:
    """The gate path is implemented and tested, ready for a config flip."""
    failing_c5 = {**GOOD_SCORES, "c5": 2}
    composite = calculate_composite(failing_c5)
    assert composite > 4.0

    gated = apply_ethics_gate(failing_c5, composite, enabled=True)

    assert gated["ethics_gate"]["mode"] == MODE_HARD_GATE
    assert gated["ethics_gate"]["triggered"] is True
    assert gated["composite_score"] == config.ETHICS_GATE_CAPPED_SCORE
    assert gated["skill_level"] == "weak"
    assert gated["ethics_gate"]["pre_gate_composite"] == composite


def test_hard_gate_leaves_a_passing_c5_alone() -> None:
    gated = apply_ethics_gate(GOOD_SCORES, calculate_composite(GOOD_SCORES), enabled=True)
    assert gated["ethics_gate"]["triggered"] is False
    assert gated["composite_score"] == 7.05


def test_ethics_gate_requires_c5() -> None:
    with pytest.raises(KeyError):
        apply_ethics_gate({"c1": 5}, 5.0)


# --------------------------------------------------------------------------- #
# EVAL-005 — justification extraction and word caps                            #
# --------------------------------------------------------------------------- #


def test_over_limit_justification_is_truncated_to_the_cap() -> None:
    """EVAL-005 acceptance: an over-limit justification is truncated."""
    long_text = " ".join(["word"] * 200)
    capped, truncated = enforce_word_cap(long_text)

    assert truncated is True
    assert len(capped.split()) == config.JUSTIFICATION_WORD_CAP
    assert capped.endswith("...")


def test_within_limit_justification_is_untouched_but_normalized() -> None:
    capped, truncated = enforce_word_cap("Clear   framing\nweak verification")
    assert truncated is False
    assert capped == "Clear framing weak verification"


def test_extract_justifications_reports_which_were_cut() -> None:
    justifications = {c: "short rationale" for c in CRITERIA}
    justifications["c4"] = " ".join(["word"] * 100)

    result = extract_justifications(justifications, submission_index=1)

    assert result["truncated"] == ["c4"]
    assert set(result["justifications"]) == set(CRITERIA)


def test_extract_justifications_rejects_a_missing_criterion() -> None:
    partial = {c: "text" for c in CRITERIA if c != "c6"}
    with pytest.raises(EvaluatorError) as exc_info:
        extract_justifications(partial)
    assert exc_info.value.code == MISSING_CRITERION


# --------------------------------------------------------------------------- #
# EVAL-009 — batch summary                                                     #
# --------------------------------------------------------------------------- #


def test_summary_prefers_the_evaluator_text() -> None:
    evaluations = [{"scores": GOOD_SCORES}]
    summary = build_overall_summary(
        {
            "strength": "Clear tasks.",
            "weakness": "Thin verification.",
            "improvement": "Add a self-check.",
        },
        evaluations,
    )
    assert summary["strength"] == "Clear tasks."
    assert summary["source"] == "evaluator"
    assert summary["criterion_averages"]["c1"] == 8.0


def test_summary_falls_back_to_derived_text_when_missing() -> None:
    """overall_summary is present and non-empty even for a partial evaluator."""
    summary = build_overall_summary(None, [{"scores": GOOD_SCORES}])

    assert summary["source"] == "derived"
    for field in ("strength", "weakness", "improvement"):
        assert summary[field].strip()
    assert "C5" in summary["strength"]  # highest-scoring criterion here
    assert "C6" in summary["weakness"]  # lowest-scoring criterion here


def test_summary_word_cap_is_enforced() -> None:
    long_text = " ".join(["word"] * 300)
    summary = build_overall_summary(
        {"strength": long_text, "weakness": long_text, "improvement": long_text},
        [{"scores": GOOD_SCORES}],
    )
    assert set(summary["truncated"]) == {"strength", "weakness", "improvement"}
    assert len(summary["strength"].split()) == config.SUMMARY_WORD_CAP


def test_criterion_averages_across_a_batch() -> None:
    averages = criterion_averages(
        [{"scores": {c: 4 for c in CRITERIA}}, {"scores": {c: 6 for c in CRITERIA}}]
    )
    assert all(value == 5.0 for value in averages.values())


# --------------------------------------------------------------------------- #
# EVAL-010 — optional confidence handling                                      #
# --------------------------------------------------------------------------- #


def test_absent_confidence_is_documented_as_not_implemented() -> None:
    """EVAL-010: the field is explicitly documented as absent in this build."""
    parsed = extract_confidence({"scores": GOOD_SCORES})
    assert parsed["confidence"] is None
    assert parsed["confidence_status"] == STATUS_NOT_IMPLEMENTED


@pytest.mark.parametrize(
    "raw, value, label",
    [
        (0.8, 0.8, "high"),
        (85, 0.85, "high"),
        (0.5, 0.5, "medium"),
        ("high", 0.9, "high"),
        ("low", 0.25, "low"),
    ],
)
def test_supplied_confidence_is_normalized(raw, value, label) -> None:
    parsed = parse_confidence(raw)
    assert parsed["confidence"] == value
    assert parsed["confidence_label"] == label
    assert parsed["confidence_status"] == STATUS_PRESENT


@pytest.mark.parametrize("raw", ["very sure", -1, 500, {}])
def test_unparseable_confidence_is_flagged(raw) -> None:
    assert parse_confidence(raw)["confidence_status"] == STATUS_UNPARSEABLE


# --------------------------------------------------------------------------- #
# EVAL-011 — layer separation in error codes                                   #
# --------------------------------------------------------------------------- #


def test_two_failure_types_produce_two_distinguishable_codes() -> None:
    """EVAL-011 acceptance: distinct mocked failures are distinguishable."""
    evaluator_failure = EvaluatorError("bad range", code=SCORE_OUT_OF_RANGE).to_dict()
    rag_failure = RagError("chroma down").to_dict()

    assert evaluator_failure["error"]["code"] != rag_failure["error"]["code"]
    assert evaluator_failure["error"]["layer"] == "evaluation"
    assert rag_failure["error"]["layer"] == "rag"


def test_structured_error_shape() -> None:
    payload = structured_error(SCORE_OUT_OF_RANGE, "out of range", {"criterion": "c2"})
    assert is_error(payload)
    assert payload["error"]["details"]["criterion"] == "c2"
    assert payload["error"]["retryable"] is False


def test_unknown_code_is_rejected() -> None:
    with pytest.raises(ValueError):
        layer_for_code("NOT_A_CODE")


# --------------------------------------------------------------------------- #
# EVAL-013 — model-version tracking                                            #
# --------------------------------------------------------------------------- #


def test_model_version_prefers_the_response_value() -> None:
    response = FakeResponse("{}", model="claude-haiku-4-5-20251001")
    assert resolve_model_version(response) == "claude-haiku-4-5-20251001"


def test_model_version_falls_back_to_config() -> None:
    assert resolve_model_version(None) == config.MODEL_ID
    assert resolve_model_version({"model": "  "}) == config.MODEL_ID


def test_every_record_gets_a_non_null_model_version() -> None:
    result = tag_with_model_version({}, response=FakeResponse("{}"))
    assert result["model_version"]
