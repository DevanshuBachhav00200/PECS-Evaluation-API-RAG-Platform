"""
Evaluator adapters for the benchmark harness (supports BENCH-004…BENCH-009).

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

BENCH-004's acceptance criterion is "run the gold test set through /evaluate".
There are three legitimate ways to reach that engine, and the harness supports
all three behind one interface so a benchmark run is reproducible regardless of
which infrastructure happens to be available:

``http``    POST to a running ``/evaluate`` with ``x-api-key``. This is the
            literal reading of BENCH-004 and the one that also exercises
            Member 5's API layer, auth and error mapping. Needs a live service.

``engine``  Call ``evaluate_batch()`` in-process. Same evaluation code path,
            one HTTP hop shorter. Needs ``ANTHROPIC_API_KEY`` and a populated
            vector store, but no running server.

``replay``  Replay a previously recorded run artifact. No API key, no network,
            no cost, fully deterministic. This is what makes the metric layer
            (BENCH-005/006/007) testable in CI, and what lets a published
            report be re-derived months later from its stored artifact.

Every adapter returns the same normalised shape, and every adapter records
``mode`` on the run so a report can never present a replayed run as a live
measurement.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from benchmark.test_set_loader import CRITERIA

#: Adapter identifiers recorded on every run artifact.
MODE_HTTP = "http"
MODE_ENGINE = "engine"
MODE_REPLAY = "replay"


class EvaluatorUnavailable(RuntimeError):
    """Raised when an adapter cannot run in the current environment.

    Carries the specific missing precondition (no API key, no server, no
    recorded artifact) so the harness can report *why* a benchmark did not run
    instead of emitting an empty result table.
    """


@dataclass
class BatchOutcome:
    """Result of grading one batch, normalised across adapters."""

    evaluations: List[Dict[str, Any]]
    usage: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0
    error: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Evaluator:
    """Interface every adapter implements."""

    mode: str = "abstract"

    def describe(self) -> Dict[str, Any]:
        """Environment description recorded on the run artifact."""
        return {"mode": self.mode}

    def grade(
        self,
        qa_pairs: Sequence[Mapping[str, Any]],
        record_ids: Sequence[str],
    ) -> BatchOutcome:
        """Grade one batch.

        Args:
            qa_pairs:   Submission payloads, exactly as a downstream caller
                        would send them — no benchmark-only keys smuggled in.
            record_ids: Gold ``record_id`` per submission, in the same order.
                        Carried alongside rather than inside the payload so the
                        graded request is byte-identical to a production one.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# In-process engine adapter
# ---------------------------------------------------------------------------


class EngineEvaluator(Evaluator):
    """Grade through ``src.evaluation.engine.evaluate_batch_safe`` in-process."""

    mode = MODE_ENGINE

    def __init__(
        self,
        retrieve_calibration: bool = True,
        client: Optional[Any] = None,
        require_api_key: bool = True,
    ) -> None:
        self.retrieve_calibration = retrieve_calibration
        self.client = client
        self._require_api_key = require_api_key

    def _check(self) -> None:
        if self.client is not None or not self._require_api_key:
            return
        if not os.getenv("ANTHROPIC_API_KEY", "").strip():
            raise EvaluatorUnavailable(
                "ANTHROPIC_API_KEY is not set — the in-process engine cannot "
                "reach the evaluator model. Set it in .env, or run the harness "
                "in --mode replay."
            )

    def describe(self) -> Dict[str, Any]:
        from src.rag.config import snapshot

        return {
            "mode": self.mode,
            "retrieve_calibration": self.retrieve_calibration,
            "config": snapshot(),
        }

    def grade(
        self,
        qa_pairs: Sequence[Mapping[str, Any]],
        record_ids: Sequence[str],
    ) -> BatchOutcome:
        self._check()

        from src.evaluation.engine import evaluate_batch_safe

        started = time.perf_counter()
        result = evaluate_batch_safe(
            list(qa_pairs),
            client=self.client,
            retrieve_calibration=self.retrieve_calibration,
        )
        elapsed = time.perf_counter() - started

        if "error" in result:
            return BatchOutcome(
                evaluations=[], latency_seconds=elapsed, error=result["error"]
            )

        return BatchOutcome(
            evaluations=list(result.get("evaluations", [])),
            usage=dict(result.get("usage", {})),
            meta=dict(result.get("meta", {})),
            latency_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# HTTP adapter — the literal BENCH-004 path
# ---------------------------------------------------------------------------


class HttpEvaluator(Evaluator):
    """Grade by POSTing to a running ``/evaluate`` endpoint."""

    mode = MODE_HTTP

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("PECS_API_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("API_KEY", "")
        self.timeout = timeout

    def describe(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout,
            # Never the key itself — this dict is written to a committed artifact.
            "api_key_configured": bool(self.api_key),
        }

    def _client(self) -> Any:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise EvaluatorUnavailable(
                "httpx is required for --mode http (pip install -r requirements.txt)"
            ) from exc
        return httpx

    def grade(
        self,
        qa_pairs: Sequence[Mapping[str, Any]],
        record_ids: Sequence[str],
    ) -> BatchOutcome:
        if not self.base_url:
            raise EvaluatorUnavailable(
                "no API base URL — set PECS_API_URL or pass --base-url "
                "(e.g. http://localhost:9000)"
            )
        if not self.api_key:
            raise EvaluatorUnavailable(
                "API_KEY is not set — /evaluate requires the x-api-key header "
                "(API-007)"
            )

        httpx = self._client()
        payload = {"submissions": [dict(pair) for pair in qa_pairs]}

        started = time.perf_counter()
        response = httpx.post(
            f"{self.base_url}/evaluate",
            json=payload,
            headers={"x-api-key": self.api_key},
            timeout=self.timeout,
        )
        elapsed = time.perf_counter() - started

        try:
            body = response.json()
        except ValueError:
            return BatchOutcome(
                evaluations=[],
                latency_seconds=elapsed,
                error={
                    "code": "BENCH_NON_JSON_RESPONSE",
                    "layer": "benchmark",
                    "message": (
                        f"HTTP {response.status_code} body was not JSON "
                        f"({response.text[:200]!r})"
                    ),
                    "details": {"status_code": response.status_code},
                },
            )

        if response.status_code != 200:
            error = body.get("error") if isinstance(body, dict) else None
            return BatchOutcome(
                evaluations=[],
                latency_seconds=elapsed,
                error=error
                or {
                    "code": f"BENCH_HTTP_{response.status_code}",
                    "layer": "benchmark",
                    "message": f"HTTP {response.status_code}",
                    "details": {"body": body},
                },
            )

        return BatchOutcome(
            evaluations=list(body.get("evaluations", [])),
            usage=dict(body.get("usage", {})),
            meta=dict(body.get("meta", {})),
            latency_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Replay adapter — offline, deterministic, free
# ---------------------------------------------------------------------------


class ReplayEvaluator(Evaluator):
    """Replay recorded evaluator outputs keyed by ``record_id``.

    The recording is a JSON file mapping ``record_id`` to the per-submission
    evaluation object produced by a previous live run (or to a synthetic one for
    harness self-tests). Missing keys are reported as an unusable record rather
    than substituted with a default score — an invented score is exactly the
    failure mode BENCH-010 forbids.
    """

    mode = MODE_REPLAY

    def __init__(
        self,
        recording: Optional[Mapping[str, Mapping[str, Any]]] = None,
        path: Optional[Path] = None,
        label: str = "unlabelled-recording",
    ) -> None:
        if recording is None and path is None:
            raise EvaluatorUnavailable(
                "replay mode needs either a recording mapping or a --replay path"
            )

        if recording is None:
            resolved = Path(path)  # type: ignore[arg-type]
            if not resolved.exists():
                raise EvaluatorUnavailable(f"replay recording not found: {resolved}")
            loaded = json.loads(resolved.read_text(encoding="utf-8"))
            recording = loaded.get("records", loaded)
            label = loaded.get("label", resolved.name) if isinstance(loaded, dict) else resolved.name

        self.recording: Dict[str, Dict[str, Any]] = {
            key: dict(value) for key, value in recording.items()  # type: ignore[union-attr]
        }
        self.label = label
        self.path = Path(path) if path is not None else None

    def describe(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "label": self.label,
            "source": str(self.path) if self.path else "in-memory",
            "records": len(self.recording),
            "note": (
                "Replayed from a recording. Scores were not produced by a live "
                "model call in this run."
            ),
        }

    def grade(
        self,
        qa_pairs: Sequence[Mapping[str, Any]],
        record_ids: Sequence[str],
    ) -> BatchOutcome:
        evaluations: List[Dict[str, Any]] = []
        missing: List[str] = []

        started = time.perf_counter()
        for offset, record_id in enumerate(record_ids, start=1):
            entry = self.recording.get(str(record_id))
            if entry is None:
                missing.append(str(record_id))
                continue
            evaluations.append({**entry, "submission_index": offset})
        elapsed = time.perf_counter() - started

        if missing and not evaluations:
            return BatchOutcome(
                evaluations=[],
                latency_seconds=elapsed,
                error={
                    "code": "BENCH_REPLAY_MISS",
                    "layer": "benchmark",
                    "message": (
                        f"recording {self.label!r} has no entry for any record in "
                        "this batch"
                    ),
                    "details": {"missing_record_ids": missing},
                },
            )

        return BatchOutcome(
            evaluations=evaluations,
            usage={"replayed": True},
            meta={"replay_label": self.label, "missing_record_ids": missing},
            latency_seconds=elapsed,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_evaluator(
    mode: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    replay_path: Optional[Path] = None,
    retrieve_calibration: bool = True,
) -> Evaluator:
    """Construct the adapter named by ``mode``."""
    normalized = (mode or "").strip().lower()

    if normalized == MODE_HTTP:
        return HttpEvaluator(base_url=base_url, api_key=api_key)
    if normalized == MODE_ENGINE:
        return EngineEvaluator(retrieve_calibration=retrieve_calibration)
    if normalized == MODE_REPLAY:
        return ReplayEvaluator(path=replay_path)

    raise ValueError(
        f"unknown evaluator mode {mode!r}; expected one of "
        f"{MODE_HTTP}, {MODE_ENGINE}, {MODE_REPLAY}"
    )


def normalize_evaluation(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten one evaluation object into the fields the metrics layer reads."""
    scores = entry.get("scores") or {}
    flattened: Dict[str, Any] = {
        "submission_index": entry.get("submission_index"),
        "composite": entry.get("composite_score"),
        "skill_level": entry.get("skill_level"),
        "calibration_used": entry.get("calibration_used"),
        "calibration_record_id": entry.get("calibration_record_id"),
        "model_version": entry.get("model_version"),
        "rubric_tag": entry.get("rubric_tag"),
        "ethics_gate_triggered": (entry.get("ethics_gate") or {}).get("triggered"),
    }
    for criterion in CRITERIA:
        flattened[criterion] = scores.get(criterion)
    return flattened
