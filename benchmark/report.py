"""
BENCH-010 — Benchmark report generator.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Renders ``benchmark/benchmark_report.md`` from the artifacts produced by
BENCH-003…BENCH-009. The generator exists so the report cannot drift from the
data: every number it prints is read out of a stored artifact, and every section
whose artifact is absent prints **NOT MEASURED** with the reason, rather than
being quietly omitted.

Acceptance (BENCH-010): every number in the report traces back to a specific
benchmark run artifact. Each section therefore names its source file, and the
report header lists the ``baseline_id`` every comparison was scored against.

The tracker's phrasing — "the honest, measured state of grading quality — no
invented figures" — is the design constraint, not a tone. A missing measurement
is reported as missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from benchmark.correlation import correlate
from benchmark.grading_comparison import RESULTS_DIR, graded_rows, level_agreement
from benchmark.mae import composite_mae, per_criterion_mae
from benchmark.perf import measure_run
from benchmark.test_set_loader import CRITERIA

#: Rendered report location (the BENCH-010 deliverable).
DEFAULT_REPORT_PATH = Path(__file__).resolve().parent / "benchmark_report.md"

#: Machine-readable companion, so downstream docs (HAND-007) can cite figures
#: without re-parsing markdown.
DEFAULT_SUMMARY_PATH = RESULTS_DIR / "benchmark_summary.json"

NOT_MEASURED = "**NOT MEASURED**"


def _fmt(value: Any, suffix: str = "") -> str:
    """Render a value for the report, never inventing one."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}{suffix}".rstrip("0").rstrip(".") + ("" if suffix else "")
    return f"{value}{suffix}"


def _pct(value: Any) -> str:
    """Render a 0-1 rate as a percentage, or an em dash when unmeasured."""
    if value is None:
        return "—"
    return f"{float(value) * 100:.1f}%"


def analyse_run(run: Mapping[str, Any]) -> Dict[str, Any]:
    """Compute every metric BENCH-005/006/009 defines over one comparison run."""
    rows = graded_rows(run)

    analysis: Dict[str, Any] = {
        "run_id": run["context"]["run_id"],
        "mode": run["context"]["mode"],
        "baseline_id": run["context"]["baseline_id"],
        "coverage": dict(run.get("coverage", {})),
        "performance": measure_run(run),
    }

    if not rows:
        analysis["composite"] = None
        analysis["per_criterion"] = None
        analysis["correlation"] = None
        analysis["level_agreement"] = None
        analysis["blocked_reason"] = (
            "no records were graded — see the batch errors on the run artifact"
        )
        return analysis

    analysis["composite"] = composite_mae(rows)
    analysis["per_criterion"] = per_criterion_mae(rows)
    analysis["correlation"] = correlate(rows)
    analysis["level_agreement"] = level_agreement(rows)
    return analysis


def _section_grading(analysis: Optional[Mapping[str, Any]], source: Optional[str]) -> List[str]:
    lines = ["## 2. Grading quality vs. the gold baseline", ""]

    if analysis is None:
        lines += [
            f"{NOT_MEASURED} — no grading-comparison run artifact is present.",
            "",
            "BENCH-004 requires the gold set to be graded through `/evaluate` or "
            "the in-process engine, which needs `ANTHROPIC_API_KEY` and a "
            "populated vector store. Reproduce with:",
            "",
            "```bash",
            "python -m benchmark.run_benchmark --mode engine --run-id live-001",
            "```",
            "",
        ]
        return lines

    composite = analysis.get("composite")
    coverage = analysis.get("coverage", {})

    lines += [
        f"Source artifact: `{source}`  ",
        f"Mode: `{analysis['mode']}`  ",
        f"Baseline: `{analysis['baseline_id']}`",
        "",
        f"Records submitted: {coverage.get('records_submitted')} · "
        f"graded: {coverage.get('records_graded')} · "
        f"failed: {coverage.get('records_failed')}",
        "",
    ]

    if analysis["mode"] == "replay":
        lines += [
            "> **This run replayed recorded evaluator output.** The metric "
            "pipeline is exercised end to end, but the scores were not produced "
            "by a live model call in this run. Grading quality is not "
            "established by a replayed run.",
            "",
        ]

    if composite is None:
        lines += [
            f"{NOT_MEASURED} — {analysis.get('blocked_reason', 'no graded records')}.",
            "",
        ]
        return lines

    lines += [
        "### Composite score (BENCH-005)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Records compared | {composite['n']} |",
        f"| MAE | {_fmt(composite['mae'])} |",
        f"| Max absolute error | {_fmt(composite['max_error'])} |",
        f"| Bias (AI − gold) | {_fmt(composite['bias'])} |",
        f"| Within ±1.0 point | {_fmt(composite['within_1_point'])} |",
        "",
        "### Per-criterion MAE (BENCH-005)",
        "",
        "| Criterion | n | MAE | Max error | Bias |",
        "|---|---|---|---|---|",
    ]

    per_criterion = analysis.get("per_criterion") or {}
    for criterion in CRITERIA:
        entry = per_criterion.get(criterion, {})
        lines.append(
            f"| {criterion.upper()} | {entry.get('n', 0)} | "
            f"{_fmt(entry.get('mae'))} | {_fmt(entry.get('max_error'))} | "
            f"{_fmt(entry.get('bias'))} |"
        )

    correlation = analysis.get("correlation") or {}
    lines += [
        "",
        "### Correlation (BENCH-006) — *[Requires Validation]*",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Paired observations | {correlation.get('n')} |",
        f"| Pearson r | {_fmt(correlation.get('pearson'))} |",
        f"| Spearman ρ | {_fmt(correlation.get('spearman'))} |",
        f"| Programme target | {correlation.get('target')} |",
        f"| Meets target | {_fmt(correlation.get('meets_target'))} |",
        f"| Sample size status | `{correlation.get('sample_size_status')}` |",
        "",
        f"> {correlation.get('note', '')}",
        "",
    ]

    agreement = analysis.get("level_agreement") or {}
    lines += [
        "### Skill-band agreement",
        "",
        f"Bands compared: {agreement.get('n')} · "
        f"agreement: {_fmt(agreement.get('agreement'))}",
        "",
    ]

    mismatches = agreement.get("mismatches") or []
    if mismatches:
        lines += [
            "| record_id | gold band | AI band | gold composite | AI composite |",
            "|---|---|---|---|---|",
        ]
        for entry in mismatches[:15]:
            lines.append(
                f"| `{entry['record_id']}` | {entry['gold_level']} | "
                f"{entry['ai_level']} | {_fmt(entry['gold_composite'])} | "
                f"{_fmt(entry['ai_composite'])} |"
            )
        if len(mismatches) > 15:
            lines.append(f"| … | | | | ({len(mismatches) - 15} more) |")
        lines.append("")

    return lines


def _section_retrieval(retrieval: Optional[Mapping[str, Any]], source: Optional[str]) -> List[str]:
    lines = ["## 3. Retrieval quality (BENCH-003)", ""]

    if retrieval is None:
        lines += [
            f"{NOT_MEASURED} — no retrieval-quality artifact is present.",
            "",
            "Needs a populated ChromaDB store. Reproduce with:",
            "",
            "```bash",
            "python -m benchmark.run_benchmark --only retrieval",
            "```",
            "",
        ]
        return lines

    summary = retrieval.get("summary", {})
    self_retrieval = summary.get("self_retrieval", {})
    scope = summary.get("domain_scope", {})

    lines += [
        f"Source artifact: `{source}`  ",
        f"Corpus: {retrieval.get('corpus_size')} serialised records · "
        f"collection holds {retrieval.get('collection_count')} · "
        f"Top-K = {retrieval.get('top_k')}",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Self-retrieval, top-1 correct | {self_retrieval.get('top1_correct')} / "
        f"{self_retrieval.get('n')} ({_pct(self_retrieval.get('pass_rate'))}) |",
        f"| Self-retrieval, found in top-K | {self_retrieval.get('found_in_top_k')} / "
        f"{self_retrieval.get('n')} |",
        f"| Domain scoping respected | {scope.get('scope_respected')} / {scope.get('n')} |",
        f"| Domains with no match under scope | "
        f"{', '.join(scope.get('domains_with_no_match') or []) or 'none'} |",
        "",
        "Near-paraphrase relevance probes are logged for manual review "
        "(BENCH-003's acceptance criterion is a reviewed judgement, not an "
        "automated score):",
        "",
    ]

    near_pairs = [case for case in retrieval.get("cases", []) if case.get("family") == "near_pair"]
    if near_pairs:
        lines += ["| Case | Retrieved (top-1) | Distance | Domain | Review |", "|---|---|---|---|---|"]
        for case in near_pairs:
            top = (case.get("retrieved") or [{}])[0]
            lines.append(
                f"| `{case['case_id']}` | `{top.get('record_id', '—')}` | "
                f"{_fmt(top.get('distance'))} | {top.get('domain', '—')} | "
                f"{case.get('review_status')} |"
            )
        lines.append("")

    return lines


def _section_consistency(consistency: Optional[Mapping[str, Any]], source: Optional[str]) -> List[str]:
    lines = ["## 4. Grading consistency (BENCH-007)", ""]

    if consistency is None:
        lines += [
            f"{NOT_MEASURED} — no consistency artifact is present.",
            "",
            "Needs at least 5 live trials over a fixed batch. Reproduce with:",
            "",
            "```bash",
            "python -m benchmark.run_benchmark --mode engine --only consistency --trials 5",
            "```",
            "",
        ]
        return lines

    verdict = consistency.get("verdict", {})
    lines += [
        f"Source artifact: `{source}`  ",
        f"Mode: `{consistency.get('mode')}` · "
        f"trials: {consistency.get('trials_succeeded')}/{consistency.get('trials_requested')} · "
        f"batch size: {consistency.get('batch_size')} (held constant)",
        "",
    ]

    if not verdict.get("measurable"):
        lines += [f"{NOT_MEASURED} — {verdict.get('reason')}.", ""]
        return lines

    lines += [
        "| Metric | Value |",
        "|---|---|",
        f"| Max composite std-dev across trials | {_fmt(verdict.get('max_composite_stdev'))} |",
        f"| Mean composite std-dev | {_fmt(verdict.get('mean_composite_stdev'))} |",
        f"| Fully deterministic | {_fmt(verdict.get('deterministic'))} |",
        f"| Submissions whose band moved | "
        f"{len(verdict.get('submissions_with_unstable_band') or [])} |",
        "",
    ]
    return lines


def _section_edge_cases(edge: Optional[Mapping[str, Any]], source: Optional[str]) -> List[str]:
    lines = ["## 5. Edge cases (BENCH-008)", ""]

    if edge is None:
        lines += [
            f"{NOT_MEASURED} — no edge-case artifact is present.",
            "",
            "```bash",
            "python -m benchmark.run_benchmark --mode engine --only edge",
            "```",
            "",
        ]
        return lines

    summary = edge.get("summary", {})
    coverage = edge.get("coverage", {})

    lines += [
        f"Source artifact: `{source}`  ",
        f"Mode: `{edge.get('mode')}`",
        "",
        f"Categories covered: {_fmt(coverage.get('all_categories_covered'))} "
        f"({', '.join(coverage.get('categories_required') or [])})",
        "",
        "| Case | Category | Expected | Observed | Satisfied |",
        "|---|---|---|---|---|",
    ]

    for case in edge.get("cases", []):
        lines.append(
            f"| `{case['case_id']}` | {case['category']} | {case['expectation']} | "
            f"{case.get('observed')} | {_fmt(case.get('satisfied'))} |"
        )

    lines += [
        "",
        f"Asserted here: {summary.get('satisfied')}/{summary.get('asserted_here')} · "
        f"deferred to TEST-009: {summary.get('deferred_to_api_tests')}",
        "",
    ]

    unsatisfied = summary.get("unsatisfied") or []
    if unsatisfied:
        lines += [f"Unsatisfied cases: {', '.join(unsatisfied)}", ""]

    return lines


def _section_performance(analysis: Optional[Mapping[str, Any]]) -> List[str]:
    lines = ["## 6. Latency and token usage (BENCH-009)", ""]

    if analysis is None or not analysis.get("performance"):
        lines += [f"{NOT_MEASURED} — no run artifact to measure.", ""]
        return lines

    performance = analysis["performance"]

    if not performance.get("sla_inputs", {}).get("valid_for_sla"):
        lines += [
            "> Figures below come from a **replayed** run and carry no real "
            "latency or token cost. They are shown to demonstrate the "
            "measurement path, and must not be quoted in `docs/integration_sla.md`.",
            "",
        ]

    latency = performance.get("latency_seconds", {})
    usage = performance.get("usage_totals", {})

    lines += [
        "| Metric | Value |",
        "|---|---|",
        f"| Batches measured | {performance.get('batches_measured')} |",
        f"| Batch latency mean (s) | {_fmt(latency.get('mean'))} |",
        f"| Batch latency p50 / p95 (s) | {_fmt(latency.get('p50'))} / {_fmt(latency.get('p95'))} |",
        f"| Batch latency max (s) | {_fmt(latency.get('max'))} |",
        f"| Wall clock, all batches (s) | {_fmt(performance.get('wall_clock_seconds'))} |",
        f"| Input tokens (total) | {_fmt(usage.get('input_tokens'))} |",
        f"| Output tokens (total) | {_fmt(usage.get('output_tokens'))} |",
        f"| Cache read tokens | {_fmt(usage.get('cache_read_input_tokens'))} |",
        f"| Estimated cost (USD) | {_fmt(performance.get('cost_usd_total'))} |",
        "",
    ]

    per_size = performance.get("per_batch_size", {})
    if per_size:
        lines += [
            "### Per batch size",
            "",
            "| Batch size | Batches | Mean latency (s) | Latency / submission (s) | Input tokens / submission |",
            "|---|---|---|---|---|",
        ]
        for size, entry in per_size.items():
            lines.append(
                f"| {size} | {entry['batches']} | "
                f"{_fmt(entry['latency_seconds'].get('mean'))} | "
                f"{_fmt(entry.get('latency_per_submission_seconds'))} | "
                f"{_fmt((entry.get('tokens_per_submission') or {}).get('input_tokens'))} |"
            )
        lines.append("")

    caching = performance.get("caching", {})
    lines += [
        "### Prompt caching",
        "",
        f"Cache read tokens observed: {_fmt(caching.get('cache_read_input_tokens'))} · "
        f"hits: {_fmt(caching.get('cache_hits_observed'))}",
        "",
        f"> {caching.get('note')}",
        "",
    ]
    return lines


def render_report(
    test_set_summary: Optional[Mapping[str, Any]] = None,
    baseline: Optional[Mapping[str, Any]] = None,
    run: Optional[Mapping[str, Any]] = None,
    run_source: Optional[str] = None,
    retrieval: Optional[Mapping[str, Any]] = None,
    retrieval_source: Optional[str] = None,
    consistency: Optional[Mapping[str, Any]] = None,
    consistency_source: Optional[str] = None,
    edge: Optional[Mapping[str, Any]] = None,
    edge_source: Optional[str] = None,
    generated_note: str = "",
) -> str:
    """Render the full benchmark report markdown."""
    analysis = analyse_run(run) if run is not None else None

    lines: List[str] = [
        "# PECS Evaluation Engine — Benchmark Report",
        "",
        "Task: **BENCH-010** · Owner: Member 6 (Testing + Benchmarking + Docker "
        "+ Deployment)",
        "",
        "> Generated by `python -m benchmark.report`. Every figure below is read "
        "out of a stored artifact under `benchmark/results/`; sections whose "
        "artifact is absent say **NOT MEASURED** and name the command that "
        "produces it. No figure in this document was estimated, rounded up from "
        "a partial run, or carried over from a previous build.",
        "",
    ]

    if generated_note:
        lines += [f"> {generated_note}", ""]

    lines += ["## 1. What was measured, and against what", ""]

    if baseline is not None:
        provenance = baseline.get("reference_provenance", {})
        summary = baseline.get("summary", {})
        lines += [
            f"Baseline: `{baseline.get('baseline_id')}`  ",
            f"Source: `{baseline.get('source_file')}` "
            f"({summary.get('records')} records)  ",
            f"Domains: {', '.join(f'{k} ({v})' for k, v in (summary.get('by_domain') or {}).items())}  ",
            f"Reference bands: "
            f"{', '.join(f'{k} ({v})' for k, v in (summary.get('by_level') or {}).items())}",
            "",
            "**Reference provenance.** "
            f"{provenance.get('note', '')}",
            "",
        ]
    elif test_set_summary is not None:
        lines += [f"Test set: {json.dumps(test_set_summary)}", ""]
    else:
        lines += [f"{NOT_MEASURED} — no baseline artifact available.", ""]

    lines += _section_grading(analysis, run_source)
    lines += _section_retrieval(retrieval, retrieval_source)
    lines += _section_consistency(consistency, consistency_source)
    lines += _section_edge_cases(edge, edge_source)
    lines += _section_performance(analysis)

    lines += [
        "## 7. Traceability",
        "",
        "| Task | Artifact | Present |",
        "|---|---|---|",
        f"| BENCH-002 baseline | `benchmark/gold_baseline.json` | "
        f"{_fmt(baseline is not None)} |",
        f"| BENCH-004 grading run | `{run_source or '—'}` | {_fmt(run is not None)} |",
        f"| BENCH-003 retrieval | `{retrieval_source or '—'}` | "
        f"{_fmt(retrieval is not None)} |",
        f"| BENCH-007 consistency | `{consistency_source or '—'}` | "
        f"{_fmt(consistency is not None)} |",
        f"| BENCH-008 edge cases | `{edge_source or '—'}` | {_fmt(edge is not None)} |",
        "",
        "Regenerate this report from the artifacts on disk:",
        "",
        "```bash",
        "python -m benchmark.report",
        "```",
        "",
    ]

    return "\n".join(lines) + "\n"


def _load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _newest(
    pattern: str,
    results_dir: Path,
    required_key: Optional[str] = None,
) -> Optional[Path]:
    """Most recently modified *usable* artifact matching ``pattern``.

    A stage that could not run writes a ``{"status": "skipped", ...}`` artifact
    so the reason is recorded. Those must not be mistaken for results — hence
    the ``.skipped.json`` exclusion and the ``required_key`` shape check. Without
    them the report would try to read metrics out of a skip record and crash, or
    worse, render an empty section as though it had been measured.
    """
    candidates = sorted(
        (
            path
            for path in results_dir.glob(pattern)
            if not path.name.endswith(".skipped.json")
        ),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )

    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("status") == "skipped":
            continue
        if required_key and required_key not in payload:
            continue
        return path

    return None


def build_report(
    results_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
    run_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Assemble the report from whatever artifacts exist on disk.

    Returns:
        dict: ``{"report": path, "summary": path, "sections_present": {...}}``.
    """
    from benchmark.gold_baseline import load_baseline

    resolved_results = Path(results_dir) if results_dir is not None else RESULTS_DIR
    resolved_results.mkdir(parents=True, exist_ok=True)

    try:
        baseline: Optional[Dict[str, Any]] = load_baseline()
    except Exception:  # noqa: BLE001 - absence is a reportable state
        baseline = None

    resolved_run_path = Path(run_path) if run_path is not None else _newest(
        "grading_*.json", resolved_results, required_key="rows"
    )
    run = _load_json(resolved_run_path)

    retrieval_path = _newest("retrieval_quality*.json", resolved_results, "cases")
    consistency_path = _newest("consistency*.json", resolved_results, "per_submission")
    edge_path = _newest("edge_cases*.json", resolved_results, "cases")

    def _rel(path: Optional[Path]) -> Optional[str]:
        if path is None:
            return None
        try:
            return path.relative_to(Path(__file__).resolve().parents[1]).as_posix()
        except ValueError:  # pragma: no cover - artifact outside the repo
            return str(path)

    markdown = render_report(
        baseline=baseline,
        run=run,
        run_source=_rel(resolved_run_path),
        retrieval=_load_json(retrieval_path),
        retrieval_source=_rel(retrieval_path),
        consistency=_load_json(consistency_path),
        consistency_source=_rel(consistency_path),
        edge=_load_json(edge_path),
        edge_source=_rel(edge_path),
    )

    target = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    target.write_text(markdown, encoding="utf-8")

    summary = {
        "baseline_id": (baseline or {}).get("baseline_id"),
        "grading": analyse_run(run) if run is not None else None,
        "artifacts": {
            "grading": _rel(resolved_run_path),
            "retrieval": _rel(retrieval_path),
            "consistency": _rel(consistency_path),
            "edge_cases": _rel(edge_path),
        },
    }
    # Written beside the artifacts it summarises, so a run against a temporary
    # results directory leaves the committed one untouched.
    summary_path = resolved_results / DEFAULT_SUMMARY_PATH.name
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return {
        "report": target,
        "summary": summary_path,
        "sections_present": {
            "baseline": baseline is not None,
            "grading": run is not None,
            "retrieval": retrieval_path is not None,
            "consistency": consistency_path is not None,
            "edge_cases": edge_path is not None,
        },
    }


if __name__ == "__main__":  # pragma: no cover - CLI
    outcome = build_report()
    print(f"wrote {outcome['report']}")
    print(f"wrote {outcome['summary']}")
    print(json.dumps(outcome["sections_present"], indent=2))
