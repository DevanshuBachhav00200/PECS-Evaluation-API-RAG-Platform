"""
BENCH-001 — Assemble the evaluation test set.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Pulls Member 2's held-out gold evaluation set (ANN-011,
``data/processed/gold_eval_set.csv``) into a benchmark-ready format the rest of
Workstream H consumes.

The loader is deliberately strict: a record that does not carry all six
criterion scores, a usable prompt, or a stable ``record_id`` is a schema error,
not something to silently drop. A benchmark run whose test set quietly shrank is
worse than one that refuses to start — the MAE would still compute, and the
number would be wrong in a way nobody could see.

Acceptance (BENCH-001): the test set loads into the harness with zero schema
errors.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

# benchmark/test_set_loader.py -> parents[1] == repo root
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Member 2's ANN-011 held-out gold evaluation set.
DEFAULT_GOLD_EVAL_PATH = REPO_ROOT / "data" / "processed" / "gold_eval_set.csv"

#: Canonical criterion keys, matching DS-002 / VDB-004 / the evaluator output.
CRITERIA: Sequence[str] = ("c1", "c2", "c3", "c4", "c5", "c6")

#: Column in the gold CSV holding each criterion (ANN-011 export uses ``cN_score``).
_SCORE_COLUMNS = {criterion: f"{criterion}_score" for criterion in CRITERIA}

#: Valid per-criterion score range (DS-010 / EVAL-004).
SCORE_MIN = 1.0
SCORE_MAX = 10.0


class TestSetError(ValueError):
    """Raised when the gold evaluation set cannot be loaded as specified."""

    # pytest collects any imported class whose name starts with "Test"; this
    # one is an exception type, not a test case.
    __test__ = False


@dataclass(frozen=True)
class GoldRecord:
    """One benchmark-ready record from the gold evaluation set.

    Attributes:
        record_id:    Stable downstream key (Member 2's handoff note: use
                      ``record_id``, not ``prompt_id``, which is not unique).
        prompt_id:    Source identifier, retained for provenance only.
        task:         Task text presented to the evaluator.
        student_prompt: The candidate prompt being graded.
        domain:       Domain slice (CODE, EDU, …), used for retrieval scoping.
        level:        Reference skill band (weak / developing / strong).
        scores:       Reference per-criterion scores, c1…c6.
        composite:    Reference composite score on the 0–10 scale.
        source:       Provenance of the reference scores (``score_source``).
        status:       Annotation review status (``approved``, …).
    """

    record_id: str
    prompt_id: str
    task: str
    student_prompt: str
    domain: Optional[str]
    level: Optional[str]
    scores: Dict[str, float]
    composite: float
    source: str
    status: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_qa_pair(self, scope_retrieval: bool = False) -> Dict[str, Any]:
        """Render this record as an engine/API submission.

        Args:
            scope_retrieval: When True, pass ``domain``/``level`` through so
                retrieval is scoped to the same metadata slice (RAG-005). Off by
                default: the graded run should see the same unscoped retrieval a
                downstream caller gets unless the benchmark is explicitly
                measuring the scoped path.
        """
        pair: Dict[str, Any] = {
            "task": self.task,
            "student_prompt": self.student_prompt,
        }
        if scope_retrieval:
            if self.domain:
                pair["domain"] = self.domain
            if self.level:
                pair["level"] = self.level
        return pair


def _require(row: Mapping[str, Any], column: str, line: int) -> str:
    value = row.get(column)
    if value is None or not str(value).strip():
        raise TestSetError(
            f"row {line}: required column {column!r} is missing or empty"
        )
    return str(value).strip()


def _parse_score(raw: Any, column: str, line: int) -> float:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise TestSetError(
            f"row {line}: column {column!r} is not numeric ({raw!r})"
        ) from exc

    if not SCORE_MIN <= value <= SCORE_MAX:
        raise TestSetError(
            f"row {line}: column {column!r} = {value} is outside the valid "
            f"{SCORE_MIN:g}-{SCORE_MAX:g} range"
        )
    return value


def _row_to_record(row: Mapping[str, Any], line: int) -> GoldRecord:
    record_id = _require(row, "record_id", line)
    prompt_text = _require(row, "prompt_text", line)

    scores = {
        criterion: _parse_score(row.get(column), column, line)
        for criterion, column in _SCORE_COLUMNS.items()
    }

    composite_raw = row.get("final_score")
    if composite_raw is None or not str(composite_raw).strip():
        raise TestSetError(f"row {line}: required column 'final_score' is empty")
    try:
        composite = float(str(composite_raw).strip())
    except ValueError as exc:
        raise TestSetError(
            f"row {line}: 'final_score' is not numeric ({composite_raw!r})"
        ) from exc

    if not 0.0 <= composite <= 10.0:
        raise TestSetError(
            f"row {line}: 'final_score' = {composite} is outside the 0-10 scale. "
            "A value near 76.5 means the ×10 scoring error (EVAL-006) is present "
            "in the reference data."
        )

    domain = (row.get("domain") or "").strip() or None
    level = (row.get("skill_level") or "").strip() or None

    return GoldRecord(
        record_id=record_id,
        prompt_id=(row.get("prompt_id") or "").strip(),
        # The gold export carries the candidate prompt only; the evaluator still
        # needs a task string, so the prompt doubles as its own task statement.
        # Documented rather than invented: see docs/member6/IMPLEMENTATION.md.
        task=(row.get("task") or "").strip() or prompt_text,
        student_prompt=prompt_text,
        domain=domain,
        level=level,
        scores=scores,
        composite=composite,
        source=(row.get("score_source") or "").strip() or "unspecified",
        status=(row.get("status") or "").strip() or "unspecified",
        extra={
            "dataset_split": (row.get("dataset_split") or "").strip(),
            "source_type": (row.get("source_type") or "").strip(),
            "kappa_flag": (row.get("kappa_flag") or "").strip(),
            "needs_review": (row.get("needs_review") or "").strip(),
        },
    )


def load_gold_eval_set(
    path: Optional[Path] = None,
    require_split: Optional[str] = "test",
) -> List[GoldRecord]:
    """Load the gold evaluation set into benchmark-ready records.

    Args:
        path:          Override the default ``data/processed/gold_eval_set.csv``.
        require_split: Assert every record carries this ``dataset_split`` value.
                       ANN-011's acceptance criterion is zero overlap with the
                       training split, so a stray ``train`` row here means
                       leakage into the benchmark. Pass ``None`` to skip.

    Returns:
        list[GoldRecord]: One record per row, in file order.

    Raises:
        TestSetError: On a missing file, an empty set, a duplicate ``record_id``,
                      a malformed score, or a split violation.
    """
    resolved = Path(path) if path is not None else DEFAULT_GOLD_EVAL_PATH

    if not resolved.exists():
        raise TestSetError(
            f"gold evaluation set not found at {resolved}. It is produced by "
            "Member 2's ANN-011 pipeline (scripts/run_member2_pipeline.py)."
        )

    records: List[GoldRecord] = []
    seen: Dict[str, int] = {}

    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise TestSetError(f"{resolved} has no header row")

        missing_columns = sorted(
            {"record_id", "prompt_text", "final_score", *_SCORE_COLUMNS.values()}
            - set(reader.fieldnames)
        )
        if missing_columns:
            raise TestSetError(
                f"{resolved} is missing required column(s): {missing_columns}"
            )

        for line, row in enumerate(reader, start=2):
            record = _row_to_record(row, line)

            if record.record_id in seen:
                raise TestSetError(
                    f"row {line}: duplicate record_id {record.record_id!r} "
                    f"(first seen on row {seen[record.record_id]})"
                )
            seen[record.record_id] = line

            if require_split:
                split = record.extra.get("dataset_split", "")
                if split and split != require_split:
                    raise TestSetError(
                        f"row {line}: record {record.record_id!r} has "
                        f"dataset_split={split!r}, expected {require_split!r}. "
                        "A non-test row in the gold set means split leakage "
                        "(DS-014 / ANN-011)."
                    )

            records.append(record)

    if not records:
        raise TestSetError(f"{resolved} contains no records")

    return records


def summarize(records: Sequence[GoldRecord]) -> Dict[str, Any]:
    """Describe a loaded test set — counts by domain, level and score source.

    Used by the benchmark report (BENCH-010) so every reported figure states the
    population it was measured over.
    """
    by_domain: Dict[str, int] = {}
    by_level: Dict[str, int] = {}
    by_source: Dict[str, int] = {}

    for record in records:
        by_domain[record.domain or "unspecified"] = (
            by_domain.get(record.domain or "unspecified", 0) + 1
        )
        by_level[record.level or "unspecified"] = (
            by_level.get(record.level or "unspecified", 0) + 1
        )
        by_source[record.source] = by_source.get(record.source, 0) + 1

    return {
        "records": len(records),
        "by_domain": dict(sorted(by_domain.items())),
        "by_level": dict(sorted(by_level.items())),
        "by_score_source": dict(sorted(by_source.items())),
    }


if __name__ == "__main__":  # pragma: no cover - operator convenience
    import json

    loaded = load_gold_eval_set()
    print(json.dumps(summarize(loaded), indent=2))
