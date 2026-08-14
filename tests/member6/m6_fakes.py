"""
Shared fakes and builders for Member 6 (testing, benchmarking, deployment) tests.

The whole suite runs offline: no Anthropic key, no network, and no dependence on
whatever happens to be in the production ChromaDB directory. That is a hard
requirement rather than a convenience — TEST-006's acceptance criterion is
literally "runs offline with zero real Anthropic API calls", and a benchmark
harness whose own tests need a paid API call could never run in CI.

Basename is unique across the repository (as ``tests/member4/m4_fakes.py``
already is), so ``from m6_fakes import ...`` can only resolve here regardless of
which ``conftest.py`` files exist alongside it.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from benchmark.evaluators import BatchOutcome, Evaluator
from benchmark.test_set_loader import CRITERIA, GoldRecord

# Re-exported so Member 6 tests can drive the real engine with Member 4's
# scripted client rather than maintaining a second copy of it.
from m4_fakes import (  # noqa: F401  (re-export)
    FakeClient,
    FakeCollection,
    FakeEmbedder,
    FakeResponse,
    FakeUsage,
    calibration_document,
    calibration_metadata,
    evaluator_payload,
    evaluator_response,
    query_result,
)


def gold_record(
    record_id: str = "rec-001",
    scores: Optional[Dict[str, float]] = None,
    composite: float = 7.05,
    domain: str = "CODE",
    level: str = "strong",
    prompt: str = "Review this endpoint and list issues by severity.",
) -> GoldRecord:
    """Build a GoldRecord without touching the CSV on disk."""
    resolved = scores or {"c1": 8.0, "c2": 7.0, "c3": 6.0, "c4": 7.0, "c5": 9.0, "c6": 5.0}
    return GoldRecord(
        record_id=record_id,
        prompt_id=f"PELS-{record_id}",
        task=prompt,
        student_prompt=prompt,
        domain=domain,
        level=level,
        scores=resolved,
        composite=composite,
        source="auto_mean",
        status="approved",
        extra={"dataset_split": "test"},
    )


def evaluation_entry(
    submission_index: int = 1,
    scores: Optional[Mapping[str, float]] = None,
    composite: float = 7.05,
    skill_level: str = "strong",
    record_id: Optional[str] = "rec-001",
) -> Dict[str, Any]:
    """One per-submission evaluation object, shaped like the engine's output."""
    resolved = dict(scores or {"c1": 8.0, "c2": 7.0, "c3": 6.0, "c4": 7.0, "c5": 9.0, "c6": 5.0})
    return {
        "submission_index": submission_index,
        "task": "Review this endpoint and list issues by severity.",
        "scores": resolved,
        "composite_score": composite,
        "skill_level": skill_level,
        "ethics_gate": {
            "mode": "weighted_only",
            "triggered": False,
            "c5_score": resolved["c5"],
            "threshold": 3.0,
            "capped_score": None,
            "pre_gate_composite": composite,
            "policy_confirmation": "pending_alliance_compliance_confirmation",
        },
        "justifications": {
            criterion: f"{criterion.upper()} rationale." for criterion in CRITERIA
        },
        "justification_truncated": [],
        "prompt_truncated": False,
        "calibration_used": True,
        "calibration_record_id": record_id,
        "domain": "CODE",
        "level": "strong",
        "confidence": None,
        "confidence_label": None,
        "confidence_status": "not_implemented_in_mvp",
        "model_version": "claude-haiku-4-5-20251001",
        "rubric_version": "pecs-rubric-v1.0",
        "rubric_tag": "pecs-rubric-v1.0+abcdef123456",
    }


class ScriptedEvaluator(Evaluator):
    """Benchmark adapter that returns pre-built outcomes, in order.

    Lets the harness be tested end to end — batching, per-criterion comparison,
    failure rows, usage aggregation — without a model, a store, or a key. A
    scripted entry may be a ``BatchOutcome`` (used as-is), a list of evaluation
    dicts, or an error dict.
    """

    mode = "scripted"

    def __init__(self, script: Sequence[Any], latency: float = 0.25) -> None:
        self.script: List[Any] = list(script)
        self.calls: List[Dict[str, Any]] = []
        self.latency = latency

    def describe(self) -> Dict[str, Any]:
        return {"mode": self.mode, "scripted_batches": len(self.script)}

    def grade(
        self,
        qa_pairs: Sequence[Mapping[str, Any]],
        record_ids: Sequence[str],
    ) -> BatchOutcome:
        self.calls.append(
            {"qa_pairs": [dict(pair) for pair in qa_pairs], "record_ids": list(record_ids)}
        )

        if not self.script:
            raise AssertionError("ScriptedEvaluator ran out of scripted batches")

        nxt = self.script.pop(0)

        if isinstance(nxt, BatchOutcome):
            return nxt
        if isinstance(nxt, dict) and "error" in nxt:
            return BatchOutcome(
                evaluations=[], latency_seconds=self.latency, error=nxt["error"]
            )

        return BatchOutcome(
            evaluations=list(nxt),
            usage={
                "input_tokens": 1000,
                "output_tokens": 200,
                "cache_read_input_tokens": 0,
                "cost_usd": 0.002,
            },
            meta={"caching": {"requested": True, "expected_effective": False}},
            latency_seconds=self.latency,
        )


def perfect_script(records: Sequence[GoldRecord], batch_size: int) -> List[List[Dict[str, Any]]]:
    """A script that echoes each record's own gold scores back.

    Used to prove the metric layer bottoms out correctly: an evaluator that
    reproduces the baseline exactly must yield MAE 0 and correlation 1. If that
    does not hold, the metric code is wrong, and no amount of live data would
    have revealed it.
    """
    batches: List[List[Dict[str, Any]]] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        batches.append(
            [
                evaluation_entry(
                    submission_index=offset,
                    scores=record.scores,
                    composite=record.composite,
                    skill_level=record.level or "strong",
                    record_id=record.record_id,
                )
                for offset, record in enumerate(chunk, start=1)
            ]
        )
    return batches


def offset_script(
    records: Sequence[GoldRecord],
    batch_size: int,
    delta: float = 1.0,
) -> List[List[Dict[str, Any]]]:
    """A script that grades every criterion ``delta`` points above gold.

    Gives the metric layer a known-answer case: MAE must equal ``delta`` and
    bias must equal ``+delta``, both hand-checkable.
    """
    batches: List[List[Dict[str, Any]]] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        entries = []
        for offset, record in enumerate(chunk, start=1):
            shifted = {
                criterion: min(10.0, record.scores[criterion] + delta)
                for criterion in CRITERIA
            }
            entries.append(
                evaluation_entry(
                    submission_index=offset,
                    scores=shifted,
                    composite=round(record.composite + delta, 2),
                    skill_level=record.level or "strong",
                    record_id=record.record_id,
                )
            )
        batches.append(entries)
    return batches


def recording_from_records(
    records: Sequence[GoldRecord],
    label: str = "synthetic-selftest",
) -> Dict[str, Any]:
    """A replay recording keyed by ``record_id``.

    Explicitly labelled synthetic. The replay adapter stamps the label onto the
    run artifact, and the report renders a warning banner for any replayed run,
    so a self-test recording can never be mistaken for a measured result.
    """
    return {
        "label": label,
        "records": {
            record.record_id: evaluation_entry(
                scores=record.scores,
                composite=record.composite,
                skill_level=record.level or "strong",
                record_id=record.record_id,
            )
            for record in records
        },
    }


def write_recording(path: Any, records: Sequence[GoldRecord]) -> Any:
    """Write a replay recording to ``path`` and return it."""
    payload = recording_from_records(records)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
