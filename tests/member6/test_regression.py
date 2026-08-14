"""
TEST-008 — Regression test fixtures.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Captures known-good outputs of the deterministic layers as committed fixtures
and diffs them on every run, so unintended scoring drift shows up as a failing
test rather than as a shifted MAE three weeks later.

Acceptance (TEST-008): fixtures are stored and diffed automatically on each test
run. ``tests/member6/fixtures/regression_cases.json`` is the store;
``test_scoring_matches_the_stored_fixture`` is the diff.

What is fixtured, and what deliberately is not:

* **Fixtured** — the composite formula, band mapping, ethics-gate block,
  calibration formatting, and the assembled batch content. All are pure
  functions of their input. If any of them changes, a stored evaluation from
  last month stops meaning what it meant, which is exactly the drift this task
  exists to catch.
* **Not fixtured** — anything a model produced. Pinning a sampled model's output
  as "known good" would make the suite fail on the model's ordinary variation
  and teach everyone to update the fixture without reading the diff. Grading
  stability against the model is BENCH-007's job, measured as variance rather
  than asserted as equality.

Regenerate deliberately, after reading the diff, with::

    python -m tests.member6.test_regression --update
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.evaluation.calculate_scores import calculate_scores
from src.evaluation.ethics_gate import apply_ethics_gate
from src.evaluation.rubric_loader import load_rubric
from src.rag.build_batch_content import build_batch_content
from src.rag.format_calibration import format_calibration
from src.evaluation.skill_level_map import map_skill_level

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "regression_cases.json"

#: Inputs the fixtures are generated from. Adding a case here and regenerating
#: is the intended way to widen coverage.
SCORE_CASES: List[Dict[str, Any]] = [
    {"case_id": "REG-SCORE-001", "scores": {"c1": 8, "c2": 7, "c3": 6, "c4": 7, "c5": 9, "c6": 5}},
    {"case_id": "REG-SCORE-002", "scores": {"c1": 10, "c2": 10, "c3": 10, "c4": 10, "c5": 10, "c6": 10}},
    {"case_id": "REG-SCORE-003", "scores": {"c1": 1, "c2": 1, "c3": 1, "c4": 1, "c5": 1, "c6": 1}},
    {"case_id": "REG-SCORE-004", "scores": {"c1": 9, "c2": 8, "c3": 8, "c4": 9, "c5": 9, "c6": 7}},
    {"case_id": "REG-SCORE-005", "scores": {"c1": 5, "c2": 6, "c3": 5, "c4": 6, "c5": 7, "c6": 4}},
    {"case_id": "REG-SCORE-006", "scores": {"c1": 2, "c2": 2, "c3": 3, "c4": 3, "c5": 4, "c6": 2}},
    # A high composite with a failing C5: the case where the ethics-gate policy
    # decision (EVAL-008) would change the answer if it were ever switched on.
    {"case_id": "REG-SCORE-007", "scores": {"c1": 9, "c2": 9, "c3": 9, "c4": 9, "c5": 2, "c6": 9}},
    {"case_id": "REG-SCORE-008", "scores": {"c1": 4, "c2": 4, "c3": 4, "c4": 4, "c5": 4, "c6": 4}},
]

CALIBRATION_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "REG-CALIB-001",
        "scores": {"c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5},
        "justification": "Clear task framing, weak verification step.",
    },
    {
        "case_id": "REG-CALIB-002",
        "scores": {"c1": 2.5, "c2": 6.67, "c3": 2.5, "c4": 6, "c5": 2.5, "c6": 2.67},
        "justification": "Role plus audience lifts C2. No output format.",
    },
]

BATCH_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "REG-BATCH-001",
        "pairs": [
            {"task": "Review an endpoint.", "student_prompt": "List issues by severity."},
            {"task": "Explain recursion.", "student_prompt": "explain recursion simply"},
        ],
    }
]

STATIC_CALIBRATION = (
    "Scores: C1:8 | C2:7 | C3:9 | C4:6 | C5:10 | C6:5 "
    "→ Justification: reference example."
)


def _fixed_retriever(prompt, domain, level, collection, embedder):  # noqa: ANN001
    """Deterministic retrieval stand-in, so the fixture is about assembly only."""
    return {
        "available": True,
        "calibration": STATIC_CALIBRATION,
        "record_id": "rec-fixture",
        "metadata": None,
    }


def generate_fixtures() -> Dict[str, Any]:
    """Compute the current outputs for every declared regression case."""
    rubric = load_rubric()

    scoring: Dict[str, Any] = {}
    for case in SCORE_CASES:
        block = calculate_scores(case["scores"])
        gated = apply_ethics_gate(block["scores"], block["composite_score"])
        scoring[case["case_id"]] = {
            "input_scores": case["scores"],
            "composite_score": block["composite_score"],
            "skill_level": block["skill_level"],
            "gated_composite": gated["composite_score"],
            "gated_skill_level": gated["skill_level"],
            "ethics_mode": gated["ethics_gate"]["mode"],
            "ethics_triggered": gated["ethics_gate"]["triggered"],
        }

    calibration = {
        case["case_id"]: format_calibration(case["scores"], case["justification"])
        for case in CALIBRATION_CASES
    }

    batches = {
        case["case_id"]: build_batch_content(case["pairs"], retriever=_fixed_retriever)
        for case in BATCH_CASES
    }

    return {
        "schema_version": "1.0",
        "note": (
            "Known-good outputs of the deterministic evaluation layers. "
            "Regenerate only after reading the diff: a change here means a "
            "stored evaluation from before the change no longer means the same "
            "thing."
        ),
        "rubric_version": rubric.version,
        "weights": {key: float(value) for key, value in rubric.weights.items()},
        "scoring": scoring,
        "calibration_formatting": calibration,
        "batch_assembly": batches,
    }


def load_fixtures() -> Dict[str, Any]:
    if not FIXTURE_PATH.exists():
        pytest.fail(
            f"regression fixtures missing at {FIXTURE_PATH}. Generate them with "
            "`python -m tests.member6.test_regression --update`."
        )
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def write_fixtures() -> Path:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(
        json.dumps(generate_fixtures(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return FIXTURE_PATH


# ---------------------------------------------------------------------------
# The diff
# ---------------------------------------------------------------------------


def test_scoring_matches_the_stored_fixture() -> None:
    """Composite, band and ethics block for every case are unchanged."""
    stored = load_fixtures()["scoring"]
    current = generate_fixtures()["scoring"]

    assert set(stored) == set(current), "regression case set changed"

    for case_id in sorted(stored):
        assert current[case_id] == stored[case_id], (
            f"scoring drift on {case_id}: {stored[case_id]} → {current[case_id]}"
        )


def test_rubric_weights_match_the_stored_fixture() -> None:
    """A weight change silently rescales every historical composite."""
    stored = load_fixtures()
    assert generate_fixtures()["weights"] == stored["weights"]
    assert generate_fixtures()["rubric_version"] == stored["rubric_version"]


def test_calibration_formatting_matches_the_stored_fixture() -> None:
    """The evaluator reads this string; its shape is part of the contract."""
    stored = load_fixtures()["calibration_formatting"]
    current = generate_fixtures()["calibration_formatting"]
    assert current == stored


def test_batch_assembly_matches_the_stored_fixture() -> None:
    stored = load_fixtures()["batch_assembly"]
    current = generate_fixtures()["batch_assembly"]
    assert current == stored


def test_every_band_is_represented_in_the_fixtures() -> None:
    """A regression suite that never produces a 'weak' cannot detect drift in it."""
    bands = {entry["skill_level"] for entry in load_fixtures()["scoring"].values()}
    assert bands == {"weak", "developing", "strong"}


def test_fixture_composites_are_on_the_zero_to_ten_scale() -> None:
    """Guards the stored data itself against the ×10 error."""
    for case_id, entry in load_fixtures()["scoring"].items():
        assert 0.0 <= entry["composite_score"] <= 10.0, f"{case_id} is off-scale"
        assert map_skill_level(entry["composite_score"]) == entry["skill_level"]


def test_failing_c5_case_records_the_pending_policy() -> None:
    """EVAL-008 is unconfirmed; the fixture must record which policy applied.

    If the hard gate is ever switched on, REG-SCORE-007's stored composite
    changes and this suite fails — which is the intended alarm, not a nuisance.
    """
    entry = load_fixtures()["scoring"]["REG-SCORE-007"]
    assert entry["ethics_mode"] == "weighted_only"
    assert entry["ethics_triggered"] is False


if __name__ == "__main__":  # pragma: no cover - regeneration entry point
    print(f"wrote {write_fixtures()}")
