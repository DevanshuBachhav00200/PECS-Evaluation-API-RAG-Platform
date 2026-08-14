"""
Benchmark runner CLI — drives BENCH-003 … BENCH-010 in one command.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Every stage writes a stored artifact under ``benchmark/results/`` and the report
generator reads only those artifacts, so a published figure can always be traced
to the run that produced it (BENCH-010).

Usage::

    # Full live run against the in-process engine (needs ANTHROPIC_API_KEY
    # and a populated vector store)
    python -m benchmark.run_benchmark --mode engine --run-id live-001

    # Against a deployed API (the literal BENCH-004 path)
    python -m benchmark.run_benchmark --mode http --base-url http://localhost:9000

    # Offline: retrieval quality only (no API key needed)
    python -m benchmark.run_benchmark --only retrieval

    # Offline: verify the whole metric pipeline from a recording
    python -m benchmark.run_benchmark --mode replay --replay <file.json>

A stage that cannot run records *why* in its artifact and the run continues.
That is deliberate: a partial benchmark that says which parts are missing is
useful, and one that dies on the first missing precondition is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.consistency_test import MIN_TRIALS, run_consistency
from benchmark.edge_cases import run_edge_cases
from benchmark.evaluators import (
    MODE_ENGINE,
    MODE_HTTP,
    MODE_REPLAY,
    EvaluatorUnavailable,
    build_evaluator,
)
from benchmark.gold_baseline import load_baseline
from benchmark.grading_comparison import (
    DEFAULT_BATCH_SIZE,
    RESULTS_DIR,
    run_comparison,
    write_run,
)
from benchmark.report import build_report
from benchmark.retrieval_quality import StoreUnavailable, run_retrieval_quality
from benchmark.test_set_loader import load_gold_eval_set

STAGES = ("retrieval", "grading", "consistency", "edge", "report")


def _write(path: Path, payload: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _skipped(stage: str, reason: str) -> Dict[str, Any]:
    return {"stage": stage, "status": "skipped", "reason": reason}


def stage_retrieval(results_dir: Path, run_id: str) -> Dict[str, Any]:
    """BENCH-003 — retrieval quality. Needs a populated store, not an API key."""
    try:
        payload = run_retrieval_quality()
    except StoreUnavailable as exc:
        artifact = _skipped("retrieval", str(exc))
        _write(results_dir / f"retrieval_quality_{run_id}.json", artifact)
        return artifact

    _write(results_dir / f"retrieval_quality_{run_id}.json", payload)
    return {"stage": "retrieval", "status": "ok", "summary": payload["summary"]}


def stage_grading(
    results_dir: Path,
    run_id: str,
    mode: str,
    base_url: Optional[str],
    replay: Optional[Path],
    batch_size: int,
    limit: Optional[int],
    scope_retrieval: bool,
) -> Dict[str, Any]:
    """BENCH-004/005/006/009 — grade the gold set and compare to the baseline."""
    try:
        evaluator = build_evaluator(
            mode, base_url=base_url, replay_path=replay
        )
        run = run_comparison(
            evaluator,
            records=load_gold_eval_set(),
            baseline=load_baseline(),
            batch_size=batch_size,
            limit=limit,
            scope_retrieval=scope_retrieval,
            run_id=f"grading_{run_id}",
            notes=f"stage=grading mode={mode}",
        )
    except EvaluatorUnavailable as exc:
        artifact = _skipped("grading", str(exc))
        _write(results_dir / f"grading_{run_id}.skipped.json", artifact)
        return artifact

    paths = write_run(run, results_dir)
    return {
        "stage": "grading",
        "status": "ok",
        "coverage": run["coverage"],
        "artifacts": {key: str(value) for key, value in paths.items()},
    }


def stage_consistency(
    results_dir: Path,
    run_id: str,
    mode: str,
    base_url: Optional[str],
    replay: Optional[Path],
    trials: int,
    batch_size: int,
) -> Dict[str, Any]:
    """BENCH-007 — repeat a fixed batch and measure variance."""
    try:
        evaluator = build_evaluator(mode, base_url=base_url, replay_path=replay)
        records = load_gold_eval_set()[:batch_size]
        payload = run_consistency(
            evaluator,
            qa_pairs=[record.to_qa_pair() for record in records],
            record_ids=[record.record_id for record in records],
            trials=trials,
        )
    except EvaluatorUnavailable as exc:
        artifact = _skipped("consistency", str(exc))
        _write(results_dir / f"consistency_{run_id}.skipped.json", artifact)
        return artifact

    _write(results_dir / f"consistency_{run_id}.json", payload)
    return {"stage": "consistency", "status": "ok", "verdict": payload["verdict"]}


def stage_edge(
    results_dir: Path,
    run_id: str,
    mode: str,
    base_url: Optional[str],
    replay: Optional[Path],
) -> Dict[str, Any]:
    """BENCH-008 — the edge-case suite."""
    try:
        evaluator = build_evaluator(mode, base_url=base_url, replay_path=replay)
        payload = run_edge_cases(evaluator)
    except EvaluatorUnavailable as exc:
        artifact = _skipped("edge", str(exc))
        _write(results_dir / f"edge_cases_{run_id}.skipped.json", artifact)
        return artifact

    _write(results_dir / f"edge_cases_{run_id}.json", payload)
    return {"stage": "edge", "status": "ok", "summary": payload["summary"]}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PECS benchmark runner (BENCH-003…010)")
    parser.add_argument(
        "--mode",
        default=MODE_ENGINE,
        choices=[MODE_ENGINE, MODE_HTTP, MODE_REPLAY],
        help="how to reach the evaluator (default: engine)",
    )
    parser.add_argument("--run-id", default="run", help="identifier stamped on artifacts")
    parser.add_argument("--base-url", default=None, help="API base URL for --mode http")
    parser.add_argument("--replay", type=Path, default=None, help="recording for --mode replay")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="grade only the first N records")
    parser.add_argument("--trials", type=int, default=MIN_TRIALS, help="consistency trials")
    parser.add_argument(
        "--scope-retrieval",
        action="store_true",
        help="pass domain/level through so retrieval is metadata-scoped (RAG-005)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=STAGES,
        default=None,
        help="run only these stages (default: all)",
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="where to render the markdown report (default: benchmark/benchmark_report.md)",
    )
    args = parser.parse_args(argv)

    stages = set(args.only) if args.only else set(STAGES)
    results_dir = Path(args.results_dir)
    outcomes: List[Dict[str, Any]] = []

    if "retrieval" in stages:
        outcomes.append(stage_retrieval(results_dir, args.run_id))

    if "grading" in stages:
        outcomes.append(
            stage_grading(
                results_dir,
                args.run_id,
                args.mode,
                args.base_url,
                args.replay,
                args.batch_size,
                args.limit,
                args.scope_retrieval,
            )
        )

    if "consistency" in stages:
        outcomes.append(
            stage_consistency(
                results_dir,
                args.run_id,
                args.mode,
                args.base_url,
                args.replay,
                args.trials,
                args.batch_size,
            )
        )

    if "edge" in stages:
        outcomes.append(
            stage_edge(results_dir, args.run_id, args.mode, args.base_url, args.replay)
        )

    if "report" in stages:
        built = build_report(results_dir=results_dir, report_path=args.report_path)
        outcomes.append(
            {
                "stage": "report",
                "status": "ok",
                "report": str(built["report"]),
                "sections_present": built["sections_present"],
            }
        )

    print(json.dumps({"run_id": args.run_id, "stages": outcomes}, indent=2, default=str))

    # Exit non-zero only when a stage the caller explicitly asked for was skipped,
    # so CI can distinguish "ran and measured" from "could not run".
    skipped = [entry for entry in outcomes if entry.get("status") == "skipped"]
    return 1 if skipped else 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main(sys.argv[1:]))
