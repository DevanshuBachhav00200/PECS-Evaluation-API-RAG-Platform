"""
BENCH-009 — Latency and token-usage measurement across the benchmark batch.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Derives end-to-end latency and token/cost figures from the run artifacts the
grading harness (BENCH-004) already produces, and rolls them up *per batch size*
— which is the shape INT-004's SLA document needs, since the engine grades a
whole batch in one call and per-record latency is meaningless without the batch
size it was measured at.

Nothing here re-runs the model. Every figure traces to a stored
``benchmark/results/<run_id>.json``, which is what makes BENCH-010's "no invented
figures" requirement mechanically true rather than a promise.

Percentiles use nearest-rank on the sorted sample. At the batch counts a gold
set of this size produces (single digits), interpolated percentiles would imply
a precision the sample cannot support; nearest-rank at least names an
observation that actually happened.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from benchmark.mae import METRIC_PRECISION

#: Cost fields the RAG usage logger (RAG-020) reports.
_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "estimated_input_tokens",
)


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    """Nearest-rank percentile of a numeric sample."""
    if not values:
        return None
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    ordered = sorted(float(value) for value in values)
    rank = max(1, min(len(ordered), int(-(-fraction * len(ordered) // 1))))
    return round(ordered[rank - 1], METRIC_PRECISION)


def _latency_stats(latencies: Sequence[float]) -> Dict[str, Any]:
    if not latencies:
        return {"n": 0, "mean": None, "min": None, "max": None, "p50": None, "p95": None}

    numeric = [float(value) for value in latencies]
    return {
        "n": len(numeric),
        "mean": round(sum(numeric) / len(numeric), METRIC_PRECISION),
        "min": round(min(numeric), METRIC_PRECISION),
        "max": round(max(numeric), METRIC_PRECISION),
        "p50": percentile(numeric, 0.50),
        "p95": percentile(numeric, 0.95),
    }


def measure_run(run: Mapping[str, Any]) -> Dict[str, Any]:
    """Summarise latency, tokens and cost for one comparison run.

    Args:
        run: A run artifact from ``benchmark.grading_comparison.run_comparison``.

    Returns:
        dict: Overall figures, a per-batch-size breakdown, and the caching
              observation (RAG-012's known non-effect is a measured fact here,
              not a claim).
    """
    batches = list(run.get("batches", []))
    successful = [batch for batch in batches if batch.get("ok")]

    by_size: Dict[int, List[Mapping[str, Any]]] = {}
    for batch in successful:
        by_size.setdefault(int(batch.get("size", 0)), []).append(batch)

    per_batch_size: Dict[str, Any] = {}
    for size, group in sorted(by_size.items()):
        latencies = [float(batch.get("latency_seconds", 0.0)) for batch in group]
        tokens = {
            field: sum(
                float((batch.get("usage") or {}).get(field, 0) or 0) for batch in group
            )
            for field in _TOKEN_FIELDS
        }
        per_batch_size[str(size)] = {
            "batches": len(group),
            "submissions": size * len(group),
            "latency_seconds": _latency_stats(latencies),
            "latency_per_submission_seconds": (
                round(sum(latencies) / (size * len(group)), METRIC_PRECISION)
                if size and group
                else None
            ),
            "tokens_total": {key: round(value, 2) for key, value in tokens.items()},
            "tokens_per_submission": (
                {
                    key: round(value / (size * len(group)), 2)
                    for key, value in tokens.items()
                }
                if size and group
                else {}
            ),
        }

    all_latencies = [float(batch.get("latency_seconds", 0.0)) for batch in successful]
    usage_totals = dict(run.get("usage_totals", {}))

    cache_reads = float(usage_totals.get("cache_read_input_tokens", 0) or 0)
    caching_observations = [
        batch.get("caching") for batch in batches if batch.get("caching")
    ]

    return {
        "run_id": run.get("context", {}).get("run_id"),
        "mode": run.get("context", {}).get("mode"),
        "baseline_id": run.get("context", {}).get("baseline_id"),
        "batches_measured": len(successful),
        "batches_failed": len(batches) - len(successful),
        "latency_seconds": _latency_stats(all_latencies),
        "wall_clock_seconds": round(sum(all_latencies), METRIC_PRECISION),
        "usage_totals": usage_totals,
        "cost_usd_total": round(float(usage_totals.get("cost_usd", 0) or 0), 6),
        "per_batch_size": per_batch_size,
        "caching": {
            "cache_read_input_tokens": cache_reads,
            "cache_hits_observed": cache_reads > 0,
            "engine_diagnostics": caching_observations[:1],
            "note": (
                "RAG-012 marks the rubric block for ephemeral caching, but "
                "Claude Haiku 4.5 requires a 4,096-token minimum cacheable "
                "prefix that the condensed rubric does not reach. Zero cache "
                "reads is the expected, documented outcome — not a "
                "misconfiguration."
            ),
        },
        "sla_inputs": {
            "note": (
                "Feeds INT-004 (docs/integration_sla.md). Figures are only valid "
                "for the mode, batch size and hardware recorded on the run "
                "artifact; a replayed run carries no real latency."
            ),
            "valid_for_sla": run.get("context", {}).get("mode") in ("http", "engine"),
        },
    }


def compare_runs(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Roll several runs into one table (e.g. batch-size sweep)."""
    measured = [measure_run(run) for run in runs]
    return {
        "runs": measured,
        "by_batch_size": {
            run["run_id"]: run["per_batch_size"] for run in measured
        },
    }
