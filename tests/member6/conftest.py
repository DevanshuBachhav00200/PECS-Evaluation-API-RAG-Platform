"""
Pytest fixtures for the Member 6 suite (testing, benchmarking, deployment).

Only fixtures live here; the fakes they are built from are in ``m6_fakes.py``,
whose basename is unique across the repository. ``tests/member4/`` uses the same
split for the same reason — ``conftest.py`` is a name every test directory is
entitled to, so importing shared helpers from it is ambiguous.

The ``rootdir``-level ``pytest.ini`` puts the repo root on ``sys.path``; the two
``sys.path`` insertions below add the sibling fake modules, which live in test
directories rather than in an importable package.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]
for candidate in (TESTS_ROOT / "member4", TESTS_ROOT / "member6"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from m6_fakes import (  # noqa: E402
    FakeClient,
    FakeCollection,
    FakeEmbedder,
    ScriptedEvaluator,
    evaluation_entry,
    evaluator_response,
    gold_record,
    query_result,
)


@pytest.fixture
def gold_records() -> List[Any]:
    """A small in-memory gold set spanning all three skill bands."""
    return [
        gold_record(
            record_id="rec-strong",
            scores={"c1": 9.0, "c2": 8.0, "c3": 8.0, "c4": 9.0, "c5": 9.0, "c6": 7.0},
            composite=8.40,
            level="strong",
            prompt="You are a reviewer. List defects by severity with a fix for each.",
        ),
        gold_record(
            record_id="rec-developing",
            scores={"c1": 5.0, "c2": 6.0, "c3": 5.0, "c4": 6.0, "c5": 7.0, "c6": 4.0},
            composite=5.60,
            level="developing",
            prompt="Review the code and tell me what could be better.",
        ),
        gold_record(
            record_id="rec-weak",
            scores={"c1": 2.0, "c2": 2.0, "c3": 3.0, "c4": 3.0, "c5": 4.0, "c6": 2.0},
            composite=2.70,
            level="weak",
            prompt="check this",
        ),
    ]


@pytest.fixture
def baseline_document(gold_records: List[Any]) -> Dict[str, Any]:
    """A frozen baseline built from ``gold_records`` (not read from disk)."""
    from benchmark.gold_baseline import build_baseline

    return build_baseline(records=gold_records)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def populated_collection() -> FakeCollection:
    return FakeCollection(query_result())


@pytest.fixture
def scripted_evaluator():
    """Factory for a ``ScriptedEvaluator`` over the supplied batches."""

    def _factory(script, latency: float = 0.25) -> ScriptedEvaluator:
        return ScriptedEvaluator(script, latency=latency)

    return _factory


@pytest.fixture
def offline_llm(monkeypatch: pytest.MonkeyPatch):
    """Install a scripted Anthropic client into the RAG layer.

    Patches the module-level cached client rather than the SDK, so the whole
    real pipeline runs — request assembly, caching markers, parsing, retries,
    scoring — and only the network hop is faked. That is what makes TEST-007 an
    end-to-end test rather than a mock of the thing under test.
    """
    from src.rag import llm_call

    def _install(responses) -> FakeClient:
        client = FakeClient(list(responses))
        monkeypatch.setattr(llm_call, "_client", client)
        return client

    yield _install

    llm_call.reset_client()


@pytest.fixture
def api_client(monkeypatch: pytest.MonkeyPatch):
    """A FastAPI TestClient with auth configured and rate limiting reset."""
    from fastapi.testclient import TestClient

    from src.api import app, global_rate_limiter, request_counter

    monkeypatch.setenv("API_KEY", "member6-test-key")
    global_rate_limiter.reset()
    request_counter.reset(start=1)

    with TestClient(app) as client:
        yield client

    global_rate_limiter.reset()
    request_counter.reset(start=1)


@pytest.fixture
def auth_headers() -> Dict[str, str]:
    return {"x-api-key": "member6-test-key"}
