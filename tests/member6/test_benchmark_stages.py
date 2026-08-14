"""
Tests for the consistency, retrieval-quality and runner stages (BENCH-003,
BENCH-007, and the CLI that drives them).

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

These three modules are the parts of the harness whose failure mode is silence:
a consistency run that quietly varies its batch measures the wrong thing, a
retrieval check that treats "no records in this domain" as a pass reports
scoping that was never exercised, and a runner that swallows a missing
precondition produces an empty benchmark that looks finished.

Each of those is asserted here directly. The store-dependent paths of
BENCH-003 are covered against a real in-memory collection rather than mocked, so
the retrieval assertions exercise the same code the live run does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

import pytest

from benchmark.consistency_test import (
    MIN_TRIALS,
    ConsistencyError,
    paraphrase,
    run_consistency,
    verdict,
)
from benchmark.retrieval_quality import (
    NEAR_PAIR_PROBES,
    RELEVANCE_TOP_K,
    StoreUnavailable,
    ensure_populated_collection,
    ingested_ids,
    self_retrieval_cases,
)
from m6_fakes import evaluation_entry


# ---------------------------------------------------------------------------
# BENCH-007 — consistency
# ---------------------------------------------------------------------------


def _fixed_batch() -> List[Dict[str, str]]:
    return [
        {"task": "Review an endpoint.", "student_prompt": "List issues by severity."},
        {"task": "Explain recursion.", "student_prompt": "explain recursion simply"},
    ]


def test_a_deterministic_evaluator_reports_zero_variance(scripted_evaluator: Any) -> None:
    trials = MIN_TRIALS
    script = [
        [evaluation_entry(submission_index=1), evaluation_entry(submission_index=2)]
        for _ in range(trials)
    ]
    result = run_consistency(
        scripted_evaluator(script), _fixed_batch(), ["r1", "r2"], trials=trials
    )

    assert result["trials_succeeded"] == trials
    assert result["verdict"]["deterministic"] is True
    assert result["verdict"]["max_composite_stdev"] == 0.0
    assert result["per_submission"]["1"]["skill_level_stable"] is True


def test_a_varying_evaluator_reports_the_spread(scripted_evaluator: Any) -> None:
    """The measurement has to be able to show instability, or it measures nothing."""
    composites = [7.0, 7.4, 6.6, 7.2, 6.8]
    script = [
        [evaluation_entry(submission_index=1, composite=value),
         evaluation_entry(submission_index=2, composite=value)]
        for value in composites
    ]

    result = run_consistency(
        scripted_evaluator(script), _fixed_batch(), ["r1", "r2"], trials=len(composites)
    )
    spread = result["per_submission"]["1"]["composite"]

    assert spread["n"] == len(composites)
    assert spread["mean"] == pytest.approx(7.0, abs=1e-6)
    assert spread["range"] == pytest.approx(0.8, abs=1e-6)
    assert result["verdict"]["deterministic"] is False


def test_a_band_flip_across_trials_is_surfaced(scripted_evaluator: Any) -> None:
    """A composite wobbling across 6.99/7.00 changes the reported certification."""
    script = [
        [evaluation_entry(submission_index=1, composite=7.0, skill_level="strong"),
         evaluation_entry(submission_index=2)],
        [evaluation_entry(submission_index=1, composite=6.9, skill_level="developing"),
         evaluation_entry(submission_index=2)],
        [evaluation_entry(submission_index=1, composite=7.0, skill_level="strong"),
         evaluation_entry(submission_index=2)],
        [evaluation_entry(submission_index=1, composite=7.0, skill_level="strong"),
         evaluation_entry(submission_index=2)],
        [evaluation_entry(submission_index=1, composite=6.95, skill_level="developing"),
         evaluation_entry(submission_index=2)],
    ]

    result = run_consistency(
        scripted_evaluator(script), _fixed_batch(), ["r1", "r2"], trials=5
    )

    assert result["per_submission"]["1"]["skill_level_stable"] is False
    assert "r1" in result["verdict"]["submissions_with_unstable_band"]


def test_the_batch_is_identical_on_every_trial(scripted_evaluator: Any) -> None:
    """Batch composition is context; varying it would measure the wrong thing."""
    script = [
        [evaluation_entry(submission_index=1), evaluation_entry(submission_index=2)]
        for _ in range(MIN_TRIALS)
    ]
    evaluator = scripted_evaluator(script)

    run_consistency(evaluator, _fixed_batch(), ["r1", "r2"], trials=MIN_TRIALS)

    first = evaluator.calls[0]["qa_pairs"]
    for call in evaluator.calls[1:]:
        assert call["qa_pairs"] == first


def test_fewer_than_five_trials_is_rejected(scripted_evaluator: Any) -> None:
    """BENCH-007's acceptance floor, enforced rather than documented."""
    with pytest.raises(ConsistencyError, match=str(MIN_TRIALS)):
        run_consistency(
            scripted_evaluator([]), _fixed_batch(), ["r1", "r2"], trials=3
        )


def test_the_floor_can_be_relaxed_but_the_run_records_it(scripted_evaluator: Any) -> None:
    script = [[evaluation_entry(submission_index=1), evaluation_entry(submission_index=2)]]
    result = run_consistency(
        scripted_evaluator(script), _fixed_batch(), ["r1", "r2"], trials=1,
        enforce_minimum=False,
    )
    assert result["minimum_enforced"] is False


def test_an_empty_batch_is_rejected(scripted_evaluator: Any) -> None:
    with pytest.raises(ConsistencyError):
        run_consistency(scripted_evaluator([]), [], [], trials=MIN_TRIALS)


def test_failed_trials_do_not_contribute_to_the_spread(scripted_evaluator: Any) -> None:
    script: List[Any] = [
        [evaluation_entry(submission_index=1), evaluation_entry(submission_index=2)]
        for _ in range(4)
    ]
    script.append({"error": {"code": "RAG_LLM_TIMEOUT", "message": "timeout"}})

    result = run_consistency(
        scripted_evaluator(script), _fixed_batch(), ["r1", "r2"], trials=5
    )

    assert result["trials_requested"] == 5
    assert result["trials_succeeded"] == 4
    assert result["per_submission"]["1"]["composite"]["n"] == 4


def test_verdict_reports_when_nothing_was_measurable() -> None:
    assert verdict({})["measurable"] is False


def test_paraphrase_preserves_the_original_text() -> None:
    """A rewrite that changed the prompt's quality would confound the trial."""
    original = "Review this endpoint and list issues by severity."
    assert original in paraphrase(original)
    assert paraphrase(original) != original


# ---------------------------------------------------------------------------
# BENCH-003 — retrieval quality
# ---------------------------------------------------------------------------


def test_self_retrieval_cases_are_restricted_to_ingested_records() -> None:
    """A case built on an excluded record would report a miss that is correct."""
    payload = {
        "ids": ["in-store-1", "excluded-1", "in-store-2"],
        "documents": ["prompt one", "prompt two", "prompt three"],
        "metadatas": [{"domain": "CODE"}, {"domain": "CODE"}, {"domain": "CODE"}],
    }
    cases = self_retrieval_cases(payload, sample=10, available_ids={"in-store-1", "in-store-2"})

    assert {case["expected_record_id"] for case in cases} == {"in-store-1", "in-store-2"}


def test_self_retrieval_cases_are_sampled_across_the_corpus() -> None:
    """Taking the first N would only ever exercise one block of the corpus."""
    payload = {
        "ids": [f"rec-{index}" for index in range(100)],
        "documents": [f"prompt {index}" for index in range(100)],
        "metadatas": [{"domain": "CODE"} for _ in range(100)],
    }
    cases = self_retrieval_cases(payload, sample=5, available_ids=set(payload["ids"]))

    indices = [int(case["expected_record_id"].split("-")[1]) for case in cases]
    assert len(cases) == 5
    assert max(indices) > 50, "sampling did not reach the back of the corpus"


def test_self_retrieval_skips_records_with_no_recoverable_text() -> None:
    payload = {
        "ids": ["rec-1", "rec-2"],
        "documents": ["", "prompt two"],
        "metadatas": [{}, {}],
    }
    cases = self_retrieval_cases(payload, sample=5, available_ids={"rec-1", "rec-2"})
    assert [case["expected_record_id"] for case in cases] == ["rec-2"]


def test_near_pair_probes_declare_what_they_expect() -> None:
    """Relevance is a human judgement; an unlabelled probe cannot be judged."""
    assert NEAR_PAIR_PROBES
    for probe in NEAR_PAIR_PROBES:
        assert probe["expectation"].strip()
        assert probe["query"].strip()


def test_empty_store_is_reported_as_unavailable_not_as_zero_relevance() -> None:
    chromadb = pytest.importorskip("chromadb")

    client = chromadb.EphemeralClient()
    name = f"m6_empty_{id(client):x}"
    collection = client.create_collection(name=name)

    try:
        with pytest.raises(StoreUnavailable, match="build_vector_store"):
            ensure_populated_collection(collection)
    finally:
        client.delete_collection(name=name)


def test_populated_store_passes_the_precondition_and_lists_its_ids() -> None:
    chromadb = pytest.importorskip("chromadb")

    from src.vectorstore.document_format import serialize_document
    from src.vectorstore.metadata_schema import build_metadata
    from src.vectorstore.upsert import upsert_record

    scores = {"c1": 8.0, "c2": 7.0, "c3": 9.0, "c4": 6.0, "c5": 10.0, "c6": 5.0}
    client = chromadb.EphemeralClient()
    name = f"m6_populated_{id(client):x}"
    collection = client.create_collection(name=name)

    try:
        upsert_record(
            collection,
            "rec-a",
            [0.0] * 383 + [1.0],
            serialize_document(scores, "Rationale."),
            build_metadata(
                domain="CODE", level="strong", scores=scores, source="train",
                version="pecs-v1.0",
            ),
        )

        assert ensure_populated_collection(collection) is collection
        assert ingested_ids(collection) == {"rec-a"}
    finally:
        client.delete_collection(name=name)


def test_relevance_top_k_is_above_the_mvp_default() -> None:
    """Top-1 alone cannot distinguish 'rank 2 of 3' from 'absent entirely'."""
    from src.rag.config import DEFAULT_TOP_K

    assert RELEVANCE_TOP_K > DEFAULT_TOP_K


# ---------------------------------------------------------------------------
# The runner CLI
# ---------------------------------------------------------------------------


def test_runner_records_a_missing_precondition_instead_of_an_empty_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A benchmark that cannot run must say so, and say why."""
    from benchmark import run_benchmark

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    exit_code = run_benchmark.main(
        [
            "--mode", "engine",
            "--only", "grading",
            "--run-id", "unit",
            "--results-dir", str(tmp_path),
        ]
    )

    assert exit_code == 1, "a skipped stage must not report success"

    artifact = json.loads((tmp_path / "grading_unit.skipped.json").read_text(encoding="utf-8"))
    assert artifact["status"] == "skipped"
    assert "ANTHROPIC_API_KEY" in artifact["reason"]


def test_runner_report_stage_writes_a_report(tmp_path: Path) -> None:
    from benchmark import run_benchmark

    report_path = tmp_path / "benchmark_report.md"
    exit_code = run_benchmark.main(
        [
            "--only", "report",
            "--run-id", "unit",
            "--results-dir", str(tmp_path),
            "--report-path", str(report_path),
        ]
    )

    assert exit_code == 0
    assert report_path.exists()
    assert "NOT MEASURED" in report_path.read_text(encoding="utf-8")
