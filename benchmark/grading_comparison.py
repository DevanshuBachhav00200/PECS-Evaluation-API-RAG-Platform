"""
BENCH-004 — Grading-comparison harness.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Runs the gold test set (BENCH-001) through the evaluation engine (via any of the
adapters in ``benchmark/evaluators.py``) and compares the AI-produced C1–C6
scores against the frozen gold baseline (BENCH-002), per criterion and per
record.

Acceptance (BENCH-004): the harness produces a per-criterion, per-record
comparison table. That table is ``run["rows"]``, and ``write_run()`` persists it
as both JSON (the full artifact) and CSV (the table a reviewer opens).

Design notes worth stating:

* **Batching is part of the measurement.** The engine grades a whole batch in a
  single LLM call, so the batch a record was graded in is context that affects
  its score. Every row records ``batch_index`` and ``batch_size`` for that
  reason, and BENCH-007 holds batch composition constant between repeats.
* **A failed batch is recorded, not dropped.** Its records appear in the table
  with null AI scores and the structured error code that caused them. A
  benchmark that silently shrinks its denominator reports a better MAE than the
  system earned.
* **Every artifact carries ``baseline_id``.** That is BENCH-002's acceptance
  criterion, and it is what makes a stored run re-checkable against the exact
  reference data it was scored against.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from benchmark.evaluators import (
    BatchOutcome,
    Evaluator,
    EvaluatorUnavailable,
    normalize_evaluation,
)
from benchmark.gold_baseline import baseline_index, load_baseline
from benchmark.test_set_loader import CRITERIA, GoldRecord, load_gold_eval_set

#: Where run artifacts are written.
RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: Default submissions per LLM call. 5 keeps a batch comfortably under the
#: RAG-019 pre-flight cap (4,000 estimated input tokens) for prompts of the
#: length seen in the gold set, while still exercising the batched path.
DEFAULT_BATCH_SIZE = 5


class ComparisonError(RuntimeError):
    """Raised when a comparison run cannot be assembled."""


@dataclass(frozen=True)
class RunContext:
    """Identity of a benchmark run, stamped on the artifact it produces."""

    run_id: str
    mode: str
    baseline_id: str
    batch_size: int
    scope_retrieval: bool
    notes: str = ""


def chunk(records: Sequence[GoldRecord], size: int) -> List[List[GoldRecord]]:
    """Split records into fixed-size batches, preserving order."""
    if size <= 0:
        raise ComparisonError(f"batch size must be positive, got {size}")
    return [list(records[start : start + size]) for start in range(0, len(records), size)]


def _blank_row(
    record: GoldRecord,
    gold: Mapping[str, Any],
    batch_index: int,
    batch_size: int,
    error: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "record_id": record.record_id,
        "prompt_id": record.prompt_id,
        "domain": record.domain,
        "batch_index": batch_index,
        "batch_size": batch_size,
        "graded": False,
        "error_code": (error or {}).get("code"),
        "error_message": (error or {}).get("message"),
        "gold_composite": gold["composite"],
        "gold_level": gold.get("level"),
        "ai_composite": None,
        "ai_level": None,
        "composite_abs_error": None,
        "level_match": None,
        "calibration_used": None,
        "calibration_record_id": None,
        "model_version": None,
        "rubric_tag": None,
        "ethics_gate_triggered": None,
    }
    for criterion in CRITERIA:
        row[f"gold_{criterion}"] = gold["scores"][criterion]
        row[f"ai_{criterion}"] = None
        row[f"abs_error_{criterion}"] = None
    return row


def _graded_row(
    record: GoldRecord,
    gold: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    batch_index: int,
    batch_size: int,
) -> Dict[str, Any]:
    flat = normalize_evaluation(evaluation)

    row = _blank_row(record, gold, batch_index, batch_size, error=None)
    row["graded"] = True
    row["ai_composite"] = flat["composite"]
    row["ai_level"] = flat["skill_level"]
    row["calibration_used"] = flat["calibration_used"]
    row["calibration_record_id"] = flat["calibration_record_id"]
    row["model_version"] = flat["model_version"]
    row["rubric_tag"] = flat["rubric_tag"]
    row["ethics_gate_triggered"] = flat["ethics_gate_triggered"]

    if flat["composite"] is not None:
        row["composite_abs_error"] = round(
            abs(float(flat["composite"]) - float(gold["composite"])), 4
        )
    if flat["skill_level"] is not None and gold.get("level"):
        row["level_match"] = str(flat["skill_level"]).lower() == str(gold["level"]).lower()

    for criterion in CRITERIA:
        ai_score = flat.get(criterion)
        row[f"ai_{criterion}"] = ai_score
        if ai_score is not None:
            row[f"abs_error_{criterion}"] = round(
                abs(float(ai_score) - float(gold["scores"][criterion])), 4
            )

    return row


def run_comparison(
    evaluator: Evaluator,
    records: Optional[Sequence[GoldRecord]] = None,
    baseline: Optional[Mapping[str, Any]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    scope_retrieval: bool = False,
    run_id: str = "run",
    limit: Optional[int] = None,
    notes: str = "",
    on_batch: Optional[Any] = None,
) -> Dict[str, Any]:
    """Grade the gold set and build the per-record, per-criterion comparison.

    Args:
        evaluator:       Adapter from ``benchmark/evaluators.py``.
        records:         Gold records. Loaded from BENCH-001 when omitted.
        baseline:        Frozen baseline document. Loaded from BENCH-002 when omitted.
        batch_size:      Submissions per evaluation call.
        scope_retrieval: Pass domain/level through so retrieval is scoped (RAG-005).
        run_id:          Identifier stamped on the artifact and used as its filename.
        limit:           Grade only the first N records (smoke runs).
        notes:           Free text recorded on the artifact.
        on_batch:        Optional callback ``(batch_index, outcome)`` for progress.

    Returns:
        dict: The run artifact — ``context``, ``environment``, ``rows``,
              ``batches``, ``usage_totals`` and ``coverage``.

    Raises:
        ComparisonError:     If a gold record has no baseline entry.
        EvaluatorUnavailable: Propagated from the adapter, so the caller can
                              report a missing precondition instead of an empty run.
    """
    resolved_records = list(records) if records is not None else load_gold_eval_set()
    if limit is not None:
        resolved_records = resolved_records[:limit]

    resolved_baseline = baseline if baseline is not None else load_baseline()
    index = baseline_index(resolved_baseline)

    missing = [r.record_id for r in resolved_records if r.record_id not in index]
    if missing:
        raise ComparisonError(
            f"{len(missing)} gold record(s) have no frozen baseline entry: "
            f"{missing[:5]}{'…' if len(missing) > 5 else ''}. Regenerate the "
            "baseline (BENCH-002) after any change to the gold set."
        )

    context = RunContext(
        run_id=run_id,
        mode=evaluator.mode,
        baseline_id=str(resolved_baseline["baseline_id"]),
        batch_size=batch_size,
        scope_retrieval=scope_retrieval,
        notes=notes,
    )

    rows: List[Dict[str, Any]] = []
    batches: List[Dict[str, Any]] = []
    usage_totals: Dict[str, float] = {}

    for batch_index, batch in enumerate(chunk(resolved_records, batch_size), start=1):
        qa_pairs = [record.to_qa_pair(scope_retrieval=scope_retrieval) for record in batch]
        ids = [record.record_id for record in batch]

        outcome: BatchOutcome = evaluator.grade(qa_pairs, ids)

        if on_batch is not None:  # pragma: no cover - progress hook
            on_batch(batch_index, outcome)

        by_index = {
            entry.get("submission_index"): entry for entry in outcome.evaluations
        }

        for offset, record in enumerate(batch, start=1):
            gold = index[record.record_id]
            evaluation = by_index.get(offset)

            if evaluation is None:
                rows.append(
                    _blank_row(
                        record,
                        gold,
                        batch_index,
                        len(batch),
                        outcome.error
                        or {
                            "code": "BENCH_NO_RESULT_FOR_SUBMISSION",
                            "message": (
                                "evaluator returned no entry for submission "
                                f"{offset} of batch {batch_index}"
                            ),
                        },
                    )
                )
            else:
                rows.append(
                    _graded_row(record, gold, evaluation, batch_index, len(batch))
                )

        batches.append(
            {
                "batch_index": batch_index,
                "size": len(batch),
                "ok": outcome.ok,
                "latency_seconds": round(outcome.latency_seconds, 4),
                "usage": dict(outcome.usage),
                "error": outcome.error,
                "caching": (outcome.meta or {}).get("caching"),
            }
        )

        for key, value in (outcome.usage or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage_totals[key] = round(usage_totals.get(key, 0) + value, 6)

    graded = sum(1 for row in rows if row["graded"])

    return {
        "context": {
            "run_id": context.run_id,
            "mode": context.mode,
            "baseline_id": context.baseline_id,
            "batch_size": context.batch_size,
            "scope_retrieval": context.scope_retrieval,
            "notes": context.notes,
        },
        "environment": evaluator.describe(),
        "coverage": {
            "records_submitted": len(rows),
            "records_graded": graded,
            "records_failed": len(rows) - graded,
            "batches": len(batches),
            "batches_ok": sum(1 for batch in batches if batch["ok"]),
        },
        "usage_totals": usage_totals,
        "batches": batches,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Persistence — the artifact every reported number traces back to (BENCH-010)
# ---------------------------------------------------------------------------


def _csv_columns() -> List[str]:
    columns = [
        "record_id",
        "prompt_id",
        "domain",
        "batch_index",
        "batch_size",
        "graded",
        "error_code",
        "gold_composite",
        "ai_composite",
        "composite_abs_error",
        "gold_level",
        "ai_level",
        "level_match",
    ]
    for criterion in CRITERIA:
        columns.extend([f"gold_{criterion}", f"ai_{criterion}", f"abs_error_{criterion}"])
    columns.extend(
        [
            "calibration_used",
            "calibration_record_id",
            "model_version",
            "rubric_tag",
            "ethics_gate_triggered",
        ]
    )
    return columns


def write_run(run: Mapping[str, Any], results_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Persist a run artifact as JSON plus the reviewer-facing CSV table.

    Returns:
        dict: ``{"json": path, "csv": path}``.
    """
    target_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(run["context"]["run_id"])
    json_path = target_dir / f"{run_id}.json"
    csv_path = target_dir / f"{run_id}_comparison.csv"

    json_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")

    columns = _csv_columns()
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in run["rows"]:
            writer.writerow(row)

    return {"json": json_path, "csv": csv_path}


def load_run(path: Path) -> Dict[str, Any]:
    """Load a previously written run artifact."""
    resolved = Path(path)
    if not resolved.exists():
        raise ComparisonError(f"run artifact not found: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def graded_rows(run: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The subset of rows that actually received AI scores."""
    return [row for row in run["rows"] if row.get("graded")]


def level_agreement(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Skill-band agreement between AI and baseline over graded rows.

    Reported next to MAE because a band disagreement is what a downstream
    consumer of ``skill_level`` actually experiences — a 0.4-point composite
    error that crosses 6.99 changes the certification band, and one that does
    not, does not.
    """
    comparable = [row for row in rows if row.get("level_match") is not None]
    if not comparable:
        return {"n": 0, "agreement": None, "mismatches": []}

    matches = sum(1 for row in comparable if row["level_match"])
    return {
        "n": len(comparable),
        "agreement": round(matches / len(comparable), 4),
        "mismatches": [
            {
                "record_id": row["record_id"],
                "gold_level": row["gold_level"],
                "ai_level": row["ai_level"],
                "gold_composite": row["gold_composite"],
                "ai_composite": row["ai_composite"],
            }
            for row in comparable
            if not row["level_match"]
        ],
    }
