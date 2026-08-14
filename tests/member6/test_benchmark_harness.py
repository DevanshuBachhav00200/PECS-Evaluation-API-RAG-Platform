"""
Tests for the benchmark harness (BENCH-001 … BENCH-004, BENCH-007 … BENCH-010).

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

These are the tests that make the harness trustworthy before a single live
model call is spent on it. A benchmark whose own machinery is unverified will
happily produce a confident number from a broken comparison, and nothing
downstream would catch it.

The key case is ``test_a_perfect_evaluator_scores_zero_error``: an evaluator
that echoes the baseline back must yield MAE 0, bias 0 and correlation 1. If it
does not, the join between AI scores and gold scores is wrong — which is the
single most likely defect in a comparison harness, and the hardest to see in a
live run.

Everything runs offline against a scripted adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from benchmark.correlation import correlate
from benchmark.edge_cases import CATEGORIES, EDGE_CASES, gradeable_cases, run_edge_cases
from benchmark.evaluators import (
    BatchOutcome,
    EvaluatorUnavailable,
    HttpEvaluator,
    ReplayEvaluator,
    build_evaluator,
    normalize_evaluation,
)
from benchmark.gold_baseline import (
    BaselineError,
    baseline_index,
    build_baseline,
    load_baseline,
    write_baseline,
)
from benchmark.grading_comparison import (
    ComparisonError,
    chunk,
    graded_rows,
    level_agreement,
    load_run,
    run_comparison,
    write_run,
)
from benchmark.mae import composite_mae, per_criterion_mae
from benchmark.perf import measure_run
from benchmark.report import analyse_run, render_report
from benchmark.test_set_loader import (
    CRITERIA,
    TestSetError,
    load_gold_eval_set,
    summarize,
)
from m6_fakes import evaluation_entry, offset_script, perfect_script, recording_from_records


# ---------------------------------------------------------------------------
# BENCH-001 — the loader, against the real shipped artifact
# ---------------------------------------------------------------------------


def test_gold_eval_set_loads_with_zero_schema_errors() -> None:
    """BENCH-001's acceptance criterion, asserted against the real CSV."""
    records = load_gold_eval_set()

    assert records
    for record in records:
        assert record.record_id
        assert record.student_prompt.strip()
        assert set(record.scores) == set(CRITERIA)
        assert all(1.0 <= value <= 10.0 for value in record.scores.values())
        assert 0.0 <= record.composite <= 10.0


def test_gold_record_ids_are_unique() -> None:
    records = load_gold_eval_set()
    assert len({record.record_id for record in records}) == len(records)


def test_gold_set_contains_only_the_held_out_test_split() -> None:
    """A train-split row in the gold set would be leakage into the benchmark."""
    for record in load_gold_eval_set():
        assert record.extra["dataset_split"] in ("", "test")


def test_loader_rejects_an_off_scale_composite(tmp_path: Path) -> None:
    """A final_score of 76.5 means the ×10 error is in the reference data."""
    header = (
        "record_id,prompt_id,prompt_text,domain,skill_level,final_score,"
        + ",".join(f"{criterion}_score" for criterion in CRITERIA)
    )
    row = "r1,P1,some prompt,CODE,strong,76.5,8,7,6,7,9,5"
    path = tmp_path / "bad.csv"
    path.write_text(f"{header}\n{row}\n", encoding="utf-8")

    with pytest.raises(TestSetError, match="×10|0-10 scale"):
        load_gold_eval_set(path, require_split=None)


def test_loader_rejects_a_duplicate_record_id(tmp_path: Path) -> None:
    header = (
        "record_id,prompt_id,prompt_text,domain,skill_level,final_score,"
        + ",".join(f"{criterion}_score" for criterion in CRITERIA)
    )
    row = "r1,P1,some prompt,CODE,strong,7.05,8,7,6,7,9,5"
    path = tmp_path / "dupe.csv"
    path.write_text(f"{header}\n{row}\n{row}\n", encoding="utf-8")

    with pytest.raises(TestSetError, match="duplicate record_id"):
        load_gold_eval_set(path, require_split=None)


def test_summary_describes_the_population(gold_records: List[Any]) -> None:
    summary = summarize(gold_records)
    assert summary["records"] == 3
    assert summary["by_level"] == {"developing": 1, "strong": 1, "weak": 1}


def test_qa_pair_omits_scope_by_default(gold_records: List[Any]) -> None:
    """A benchmark run must see the same unscoped retrieval production does."""
    pair = gold_records[0].to_qa_pair()
    assert set(pair) == {"task", "student_prompt"}

    scoped = gold_records[0].to_qa_pair(scope_retrieval=True)
    assert scoped["domain"] == "CODE"


# ---------------------------------------------------------------------------
# BENCH-002 — the frozen baseline
# ---------------------------------------------------------------------------


def test_baseline_id_is_stable_across_regeneration(gold_records: List[Any]) -> None:
    """Same gold set in, same baseline_id out — otherwise it is not a lock."""
    first = build_baseline(records=gold_records)
    second = build_baseline(records=gold_records)
    assert first["baseline_id"] == second["baseline_id"]


def test_baseline_id_changes_when_a_reference_score_changes(
    gold_records: List[Any],
) -> None:
    original = build_baseline(records=gold_records)

    gold_records[0].scores["c1"] = 3.0
    altered = build_baseline(records=gold_records)

    assert altered["baseline_id"] != original["baseline_id"]


def test_tampering_with_the_baseline_file_is_detected(
    tmp_path: Path, gold_records: List[Any]
) -> None:
    """The integrity check is the whole point of freezing the baseline."""
    path = tmp_path / "gold_baseline.json"
    payload = build_baseline(records=gold_records)
    path.write_text(json.dumps(payload), encoding="utf-8")

    load_baseline(path)  # clean load

    payload["records"][0]["composite"] = 9.99
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BaselineError, match="integrity check failed"):
        load_baseline(path)


def test_shipped_baseline_passes_its_own_integrity_check() -> None:
    payload = load_baseline()
    assert payload["records"]
    assert payload["baseline_id"]


def test_baseline_records_its_reference_provenance() -> None:
    """BENCH-010 must be able to say what the 'gold' scores actually are."""
    provenance = load_baseline()["reference_provenance"]
    assert provenance["human_sme_signoff"] == "pending"
    assert "3-judge" in provenance["note"]


def test_baseline_index_is_keyed_by_record_id(gold_records: List[Any]) -> None:
    index = baseline_index(build_baseline(records=gold_records))
    assert set(index) == {record.record_id for record in gold_records}


def test_write_baseline_round_trips(tmp_path: Path) -> None:
    path = write_baseline(tmp_path / "baseline.json")
    assert load_baseline(path)["records"]


# ---------------------------------------------------------------------------
# BENCH-004 — the comparison harness
# ---------------------------------------------------------------------------


def test_chunking_preserves_order_and_covers_every_record(
    gold_records: List[Any],
) -> None:
    batches = chunk(gold_records, 2)
    assert [len(batch) for batch in batches] == [2, 1]
    assert [record.record_id for batch in batches for record in batch] == [
        record.record_id for record in gold_records
    ]


def test_a_perfect_evaluator_scores_zero_error(
    gold_records: List[Any], baseline_document: Dict[str, Any], scripted_evaluator: Any
) -> None:
    """The harness's own known-answer case — see the module docstring."""
    evaluator = scripted_evaluator(perfect_script(gold_records, batch_size=2))

    run = run_comparison(
        evaluator,
        records=gold_records,
        baseline=baseline_document,
        batch_size=2,
        run_id="selftest-perfect",
    )
    rows = graded_rows(run)

    assert len(rows) == len(gold_records)

    composite = composite_mae(rows)
    assert composite["mae"] == 0.0
    assert composite["bias"] == 0.0
    assert composite["within_1_point"] == 1.0

    for criterion, entry in per_criterion_mae(rows).items():
        assert entry["mae"] == 0.0, f"{criterion} should have zero error"

    assert correlate(rows)["pearson"] == 1.0
    assert level_agreement(rows)["agreement"] == 1.0


def test_a_uniformly_generous_evaluator_shows_that_bias(
    gold_records: List[Any], baseline_document: Dict[str, Any], scripted_evaluator: Any
) -> None:
    """Every criterion one point high → MAE 1.0 and bias +1.0, by construction."""
    evaluator = scripted_evaluator(offset_script(gold_records, batch_size=3, delta=1.0))

    run = run_comparison(
        evaluator,
        records=gold_records,
        baseline=baseline_document,
        batch_size=3,
        run_id="selftest-offset",
    )
    composite = composite_mae(graded_rows(run))

    assert composite["mae"] == 1.0
    assert composite["bias"] == 1.0


def test_comparison_table_has_one_row_per_record_per_criterion(
    gold_records: List[Any], baseline_document: Dict[str, Any], scripted_evaluator: Any
) -> None:
    """BENCH-004's stated deliverable: a per-criterion, per-record table."""
    evaluator = scripted_evaluator(perfect_script(gold_records, batch_size=3))
    run = run_comparison(
        evaluator, records=gold_records, baseline=baseline_document, batch_size=3,
        run_id="selftest-table",
    )

    assert len(run["rows"]) == len(gold_records)
    for row in run["rows"]:
        for criterion in CRITERIA:
            assert f"gold_{criterion}" in row
            assert f"ai_{criterion}" in row
            assert f"abs_error_{criterion}" in row


def test_every_run_records_the_baseline_it_was_scored_against(
    gold_records: List[Any], baseline_document: Dict[str, Any], scripted_evaluator: Any
) -> None:
    """BENCH-002's acceptance criterion: referenced by ID in every run."""
    evaluator = scripted_evaluator(perfect_script(gold_records, batch_size=3))
    run = run_comparison(
        evaluator, records=gold_records, baseline=baseline_document, batch_size=3,
        run_id="selftest-id",
    )
    assert run["context"]["baseline_id"] == baseline_document["baseline_id"]


def test_a_failed_batch_is_recorded_not_dropped(
    gold_records: List[Any], baseline_document: Dict[str, Any], scripted_evaluator: Any
) -> None:
    """A shrinking denominator would flatter the MAE; the rows stay, ungraded."""
    evaluator = scripted_evaluator(
        [
            {"error": {"code": "RAG_LLM_TIMEOUT", "message": "upstream timed out"}},
            [evaluation_entry(submission_index=1, record_id="rec-weak")],
        ]
    )

    run = run_comparison(
        evaluator, records=gold_records, baseline=baseline_document, batch_size=2,
        run_id="selftest-failure",
    )

    assert run["coverage"]["records_submitted"] == 3
    assert run["coverage"]["records_failed"] == 2
    failed = [row for row in run["rows"] if not row["graded"]]
    assert {row["error_code"] for row in failed} == {"RAG_LLM_TIMEOUT"}


def test_submissions_are_sent_without_benchmark_only_keys(
    gold_records: List[Any], baseline_document: Dict[str, Any], scripted_evaluator: Any
) -> None:
    """The graded request must be identical to a production one."""
    evaluator = scripted_evaluator(perfect_script(gold_records, batch_size=3))
    run_comparison(
        evaluator, records=gold_records, baseline=baseline_document, batch_size=3,
        run_id="selftest-payload",
    )

    for pair in evaluator.calls[0]["qa_pairs"]:
        assert set(pair) <= {"task", "student_prompt", "domain", "level"}


def test_a_record_missing_from_the_baseline_is_a_hard_error(
    gold_records: List[Any], baseline_document: Dict[str, Any], scripted_evaluator: Any
) -> None:
    stripped = {**baseline_document, "records": baseline_document["records"][:1]}
    evaluator = scripted_evaluator(perfect_script(gold_records, batch_size=3))

    with pytest.raises(ComparisonError, match="no frozen baseline entry"):
        run_comparison(
            evaluator, records=gold_records, baseline=stripped, batch_size=3,
            run_id="selftest-missing",
        )


def test_run_artifact_round_trips_through_disk(
    tmp_path: Path,
    gold_records: List[Any],
    baseline_document: Dict[str, Any],
    scripted_evaluator: Any,
) -> None:
    evaluator = scripted_evaluator(perfect_script(gold_records, batch_size=3))
    run = run_comparison(
        evaluator, records=gold_records, baseline=baseline_document, batch_size=3,
        run_id="selftest-io",
    )

    paths = write_run(run, tmp_path)
    assert paths["csv"].exists()
    assert load_run(paths["json"])["context"]["run_id"] == "selftest-io"

    header = paths["csv"].read_text(encoding="utf-8").splitlines()[0]
    for criterion in CRITERIA:
        assert f"abs_error_{criterion}" in header


def test_level_agreement_reports_the_mismatches(
    gold_records: List[Any], baseline_document: Dict[str, Any], scripted_evaluator: Any
) -> None:
    """A band flip is what a downstream consumer of skill_level actually feels."""
    entries = [
        evaluation_entry(submission_index=1, composite=8.4, skill_level="developing"),
        evaluation_entry(submission_index=2, composite=5.6, skill_level="developing"),
        evaluation_entry(submission_index=3, composite=2.7, skill_level="weak"),
    ]
    evaluator = scripted_evaluator([entries])

    run = run_comparison(
        evaluator, records=gold_records, baseline=baseline_document, batch_size=3,
        run_id="selftest-bands",
    )
    agreement = level_agreement(graded_rows(run))

    assert agreement["n"] == 3
    assert agreement["agreement"] == pytest.approx(2 / 3, abs=1e-4)
    assert agreement["mismatches"][0]["record_id"] == "rec-strong"


# ---------------------------------------------------------------------------
# Evaluator adapters
# ---------------------------------------------------------------------------


def test_replay_adapter_grades_from_a_recording(gold_records: List[Any]) -> None:
    recording = recording_from_records(gold_records)
    evaluator = ReplayEvaluator(recording=recording["records"], label=recording["label"])

    outcome = evaluator.grade(
        [record.to_qa_pair() for record in gold_records],
        [record.record_id for record in gold_records],
    )

    assert len(outcome.evaluations) == 3
    assert outcome.ok


def test_replay_adapter_reports_a_miss_rather_than_inventing_a_score(
    gold_records: List[Any],
) -> None:
    evaluator = ReplayEvaluator(recording={}, label="empty")
    outcome = evaluator.grade(
        [gold_records[0].to_qa_pair()], [gold_records[0].record_id]
    )

    assert not outcome.ok
    assert outcome.error["code"] == "BENCH_REPLAY_MISS"


def test_replay_runs_are_labelled_as_replayed(gold_records: List[Any]) -> None:
    """A replayed run must never read as a live measurement."""
    evaluator = ReplayEvaluator(
        recording=recording_from_records(gold_records)["records"], label="synthetic"
    )
    described = evaluator.describe()

    assert described["mode"] == "replay"
    assert "not produced by a live model call" in described["note"]


def test_http_adapter_refuses_without_a_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PECS_API_URL", raising=False)
    with pytest.raises(EvaluatorUnavailable, match="base URL"):
        HttpEvaluator(api_key="k").grade([{"task": "t", "student_prompt": "p"}], ["r1"])


def test_http_adapter_never_records_the_api_key() -> None:
    """``describe()`` is written verbatim into a committed artifact."""
    described = HttpEvaluator(base_url="http://localhost:9000", api_key="secret").describe()
    assert "secret" not in json.dumps(described)
    assert described["api_key_configured"] is True


def test_unknown_evaluator_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown evaluator mode"):
        build_evaluator("telepathy")


def test_normalize_evaluation_flattens_the_fields_metrics_read() -> None:
    flat = normalize_evaluation(evaluation_entry())
    assert flat["composite"] == 7.05
    assert set(CRITERIA) <= set(flat)


# ---------------------------------------------------------------------------
# BENCH-008 — edge-case suite structure
# ---------------------------------------------------------------------------


def test_every_required_edge_case_category_is_covered() -> None:
    covered = {case.category for case in EDGE_CASES}
    assert set(CATEGORIES) <= covered


def test_every_edge_case_declares_an_expectation_and_a_rationale() -> None:
    """An edge case with no stated expectation cannot pass or fail meaningfully."""
    for case in EDGE_CASES:
        assert case.expectation
        assert case.rationale.strip()


def test_edge_case_run_reports_coverage(scripted_evaluator: Any) -> None:
    gradeable = gradeable_cases()
    evaluator = scripted_evaluator(
        [[evaluation_entry(submission_index=1)] for _ in gradeable]
    )

    result = run_edge_cases(evaluator)

    assert result["coverage"]["all_categories_covered"] is True
    assert result["summary"]["deferred_to_api_tests"] == len(EDGE_CASES) - len(gradeable)


def test_edge_case_marked_scored_fails_when_the_engine_errors(
    scripted_evaluator: Any,
) -> None:
    """The suite must be able to fail; a runner that always passes measures nothing."""
    gradeable = gradeable_cases()
    evaluator = scripted_evaluator(
        [{"error": {"code": "RAG_LLM_TIMEOUT", "message": "timeout"}} for _ in gradeable]
    )

    result = run_edge_cases(evaluator)
    assert result["summary"]["unsatisfied"]


# ---------------------------------------------------------------------------
# BENCH-009 / BENCH-010 — measurement and reporting
# ---------------------------------------------------------------------------


def test_performance_summary_is_derived_from_the_run_artifact(
    gold_records: List[Any], baseline_document: Dict[str, Any], scripted_evaluator: Any
) -> None:
    evaluator = scripted_evaluator(perfect_script(gold_records, batch_size=2), latency=0.5)
    run = run_comparison(
        evaluator, records=gold_records, baseline=baseline_document, batch_size=2,
        run_id="selftest-perf",
    )

    performance = measure_run(run)

    assert performance["batches_measured"] == 2
    assert performance["latency_seconds"]["mean"] == 0.5
    assert performance["usage_totals"]["input_tokens"] == 2000
    assert performance["caching"]["cache_hits_observed"] is False


def test_scripted_run_is_not_valid_for_the_sla(
    gold_records: List[Any], baseline_document: Dict[str, Any], scripted_evaluator: Any
) -> None:
    """Only http/engine runs may feed INT-004's published SLA figures."""
    evaluator = scripted_evaluator(perfect_script(gold_records, batch_size=3))
    run = run_comparison(
        evaluator, records=gold_records, baseline=baseline_document, batch_size=3,
        run_id="selftest-sla",
    )
    assert measure_run(run)["sla_inputs"]["valid_for_sla"] is False


def test_report_marks_absent_sections_as_not_measured() -> None:
    """BENCH-010's core requirement: a missing measurement says so."""
    markdown = render_report(baseline=load_baseline())

    assert "**NOT MEASURED**" in markdown
    assert "python -m benchmark.run_benchmark" in markdown


def test_report_renders_the_measured_numbers_when_a_run_exists(
    gold_records: List[Any], baseline_document: Dict[str, Any], scripted_evaluator: Any
) -> None:
    evaluator = scripted_evaluator(offset_script(gold_records, batch_size=3, delta=1.0))
    run = run_comparison(
        evaluator, records=gold_records, baseline=baseline_document, batch_size=3,
        run_id="selftest-report",
    )

    markdown = render_report(
        baseline=baseline_document, run=run, run_source="benchmark/results/selftest.json"
    )

    assert "benchmark/results/selftest.json" in markdown
    assert baseline_document["baseline_id"] in markdown
    assert "| MAE | 1" in markdown


def test_report_flags_a_replayed_run(gold_records: List[Any], baseline_document: Dict[str, Any]) -> None:
    evaluator = ReplayEvaluator(
        recording=recording_from_records(gold_records)["records"], label="synthetic"
    )
    run = run_comparison(
        evaluator, records=gold_records, baseline=baseline_document, batch_size=3,
        run_id="selftest-replay",
    )

    markdown = render_report(baseline=baseline_document, run=run, run_source="x.json")
    assert "replayed recorded evaluator output" in markdown


def test_analysis_reports_a_run_with_no_graded_records(
    gold_records: List[Any], baseline_document: Dict[str, Any], scripted_evaluator: Any
) -> None:
    evaluator = scripted_evaluator(
        [{"error": {"code": "RAG_LLM_AUTH_ERROR", "message": "no key"}}]
    )
    run = run_comparison(
        evaluator, records=gold_records, baseline=baseline_document, batch_size=3,
        run_id="selftest-empty",
    )

    analysis = analyse_run(run)
    assert analysis["composite"] is None
    assert analysis["blocked_reason"]
