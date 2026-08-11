"""
Pytest fixtures for Member 4 (RAG + evaluation engine) tests.

Only fixtures live here — pytest discovers them automatically by filename, so
this file must keep its name. The fakes and payload builders they are built from
live in ``m4_fakes.py``, whose basename is unique across the repository; tests
import those directly from there rather than from ``conftest``, so a sibling
``tests/member5/conftest.py`` can never shadow them.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from m4_fakes import (
    FakeCollection,
    FakeEmbedder,
    calibration_metadata,
    empty_query_result,
    query_result,
)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def populated_collection() -> FakeCollection:
    return FakeCollection(query_result())


@pytest.fixture
def empty_collection() -> FakeCollection:
    return FakeCollection(empty_query_result())


@pytest.fixture
def qa_pairs() -> List[Dict[str, str]]:
    return [
        {
            "task": "Write a prompt that asks for a REST endpoint review.",
            "student_prompt": "Review this endpoint and list issues by severity.",
        }
    ]


@pytest.fixture
def stub_retriever():
    """Retriever callable that returns a fixed calibration without a store."""

    def _retriever(prompt, domain, level, collection, embedder):  # noqa: ANN001
        return {
            "available": True,
            "calibration": (
                "Scores: C1:8 | C2:7 | C3:9 | C4:6 | C5:10 | C6:5 "
                "→ Justification: reference example."
            ),
            "record_id": "rec-001",
            "distance": 0.12,
            "metadata": calibration_metadata(),
            "examples": [],
        }

    return _retriever
