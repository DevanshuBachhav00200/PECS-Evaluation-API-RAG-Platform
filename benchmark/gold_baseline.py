"""
BENCH-002 — Establish reference/gold scores as the comparison baseline.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Freezes Member 2's gold evaluation scores into ``benchmark/gold_baseline.json``
— the immutable comparison baseline every MAE (BENCH-005), correlation
(BENCH-006) and consistency (BENCH-007) figure is measured against.

"Immutable" is enforced, not asserted. The baseline carries a ``baseline_id``
that is a SHA-256 fingerprint of the reference records themselves, and
``load_baseline()`` recomputes it on every read. If anyone edits a score in the
file, the fingerprint stops matching and the load fails loudly rather than
silently rebasing months of comparisons.

Acceptance (BENCH-002): the baseline file is version-locked and referenced by ID
in every benchmark run — see ``benchmark/grading_comparison.py``, which stamps
``baseline_id`` onto every run artifact it writes.

Regenerate with::

    python -m benchmark.gold_baseline --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from benchmark.test_set_loader import (
    CRITERIA,
    DEFAULT_GOLD_EVAL_PATH,
    GoldRecord,
    REPO_ROOT,
    load_gold_eval_set,
    summarize,
)

#: Frozen baseline location. Committed — it is the reference, not an output.
DEFAULT_BASELINE_PATH = Path(__file__).resolve().parent / "gold_baseline.json"

#: Bumped only when the *shape* of the baseline file changes.
BASELINE_SCHEMA_VERSION = "1.0"


class BaselineError(RuntimeError):
    """Raised when the frozen baseline is missing, malformed, or altered."""


def _fingerprint(records: Sequence[Dict[str, Any]]) -> str:
    """SHA-256 over the reference records, insensitive to key ordering.

    Deliberately covers only the reference data — not the generation metadata —
    so regenerating the baseline from an unchanged gold set reproduces the same
    ``baseline_id`` byte for byte.
    """
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _to_baseline_record(record: GoldRecord) -> Dict[str, Any]:
    return {
        "record_id": record.record_id,
        "prompt_id": record.prompt_id,
        "domain": record.domain,
        "level": record.level,
        "scores": {criterion: record.scores[criterion] for criterion in CRITERIA},
        "composite": record.composite,
        "score_source": record.source,
        "status": record.status,
    }


def build_baseline(
    records: Optional[Sequence[GoldRecord]] = None,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build the baseline payload from the gold evaluation set.

    Args:
        records:     Pre-loaded records. Loaded from ``source_path`` when omitted.
        source_path: Gold-set CSV to freeze. Defaults to Member 2's ANN-011 export.

    Returns:
        dict: The baseline document, ready to write.
    """
    resolved_source = source_path or DEFAULT_GOLD_EVAL_PATH
    loaded = list(records) if records is not None else load_gold_eval_set(resolved_source)

    baseline_records = [_to_baseline_record(record) for record in loaded]
    baseline_records.sort(key=lambda entry: entry["record_id"])

    score_sources = sorted({entry["score_source"] for entry in baseline_records})

    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "baseline_id": _fingerprint(baseline_records),
        "source_file": str(
            Path(resolved_source).resolve().relative_to(REPO_ROOT).as_posix()
        ),
        "produced_by": "ANN-011 (Member 2) — held-out gold evaluation set",
        "frozen_by": "BENCH-002 (Member 6)",
        "criteria": list(CRITERIA),
        "score_scale": {"min": 1.0, "max": 10.0, "composite_scale": "0-10"},
        # Stated, not glossed: these are 3-judge auto-mean scores, not SME human
        # reference scores. BENCH-010 reports correlation against this baseline
        # as model-vs-judge-consensus agreement, and says so.
        "reference_provenance": {
            "score_sources": score_sources,
            "human_sme_signoff": "pending",
            "note": (
                "Reference scores are the 3-judge consensus mean recorded by "
                "ANN-010/ANN-011, not SME human reference scores. Treat "
                "agreement figures as model-vs-judge-consensus until Team Beta "
                "signs off (see ANN-007, data/reports/quality_summary.json "
                "human_review_status)."
            ),
        },
        "summary": summarize(loaded),
        "records": baseline_records,
    }


def write_baseline(
    path: Optional[Path] = None,
    source_path: Optional[Path] = None,
) -> Path:
    """Write the frozen baseline to disk and return its path."""
    target = Path(path) if path is not None else DEFAULT_BASELINE_PATH
    payload = build_baseline(source_path=source_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return target


def load_baseline(path: Optional[Path] = None, verify: bool = True) -> Dict[str, Any]:
    """Load the frozen baseline, verifying it has not been altered.

    Args:
        path:   Baseline file. Defaults to ``benchmark/gold_baseline.json``.
        verify: Recompute the fingerprint and reject a mismatch (default True).

    Returns:
        dict: The baseline document.

    Raises:
        BaselineError: If the file is missing, malformed, or its records no
                       longer hash to the recorded ``baseline_id``.
    """
    resolved = Path(path) if path is not None else DEFAULT_BASELINE_PATH

    if not resolved.exists():
        raise BaselineError(
            f"frozen baseline not found at {resolved}. Regenerate with "
            "`python -m benchmark.gold_baseline --write`."
        )

    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineError(f"{resolved} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict) or "records" not in payload:
        raise BaselineError(f"{resolved} is not a baseline document")

    if verify:
        recomputed = _fingerprint(payload["records"])
        recorded = payload.get("baseline_id")
        if recomputed != recorded:
            raise BaselineError(
                f"baseline integrity check failed for {resolved}: records hash to "
                f"{recomputed}, but the file records baseline_id {recorded}. The "
                "baseline is version-locked (BENCH-002) — regenerate it "
                "deliberately rather than editing scores in place."
            )

    return payload


def baseline_index(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Index a baseline document by ``record_id`` for per-record comparison."""
    return {entry["record_id"]: entry for entry in payload["records"]}


def _main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="BENCH-002 gold baseline tool")
    parser.add_argument(
        "--write", action="store_true", help="regenerate benchmark/gold_baseline.json"
    )
    parser.add_argument("--path", type=Path, default=None, help="baseline file path")
    args = parser.parse_args(argv)

    if args.write:
        written = write_baseline(args.path)
        payload = load_baseline(written)
        print(f"wrote {written} ({len(payload['records'])} records)")
        print(f"baseline_id: {payload['baseline_id']}")
        return 0

    payload = load_baseline(args.path)
    print(f"baseline_id: {payload['baseline_id']}")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
