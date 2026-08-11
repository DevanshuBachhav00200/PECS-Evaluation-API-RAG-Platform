"""
Shared fakes and payload builders for Member 4 (RAG + evaluation engine) tests.

The whole Member 4 stack is dependency-injected, so these tests need neither a
populated ChromaDB nor a real Anthropic API key. That is deliberate: Member 6's
suite runs on every CI checkout, and requiring a live vector store or a paid API
call to test prompt assembly would make that impossible.

Why this module exists separately from ``conftest.py``:
``conftest.py`` is a filename every test directory is entitled to use, so
``from conftest import ...`` is ambiguous once a sibling directory (e.g.
``tests/member5/``) adds its own. This module's basename is unique across the
repository, so ``from m4_fakes import ...`` can only ever resolve here.

``conftest.py`` keeps the pytest fixtures — pytest discovers those automatically
by filename, and they must stay there.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

CRITERIA = ("c1", "c2", "c3", "c4", "c5", "c6")


# ---------------------------------------------------------------------------
# Retrieval fakes
# ---------------------------------------------------------------------------


class FakeCollection:
    """Minimal stand-in for a ChromaDB collection.

    Returns whatever ``result`` it was constructed with and records the kwargs it
    was queried with, so tests can assert on the where-clause and n_results.
    """

    def __init__(self, result: Optional[Dict[str, Any]] = None) -> None:
        self.result = result if result is not None else empty_query_result()
        self.calls: List[Dict[str, Any]] = []

    def query(self, **kwargs: Any) -> Dict[str, Any]:
        self.calls.append(kwargs)
        return self.result


class FakeEmbedder:
    """Deterministic 384-dimensional embedder. Records what it encoded."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.encoded: List[str] = []

    def encode(self, text: str) -> List[float]:
        self.encoded.append(text)
        seed = (sum(ord(char) for char in text) % 97) + 1
        return [(seed % (index + 7)) / 100.0 for index in range(self.dim)]


def calibration_document(
    scores: Optional[Dict[str, float]] = None,
    justification: str = "Clear task framing, weak verification step.",
) -> str:
    """Build a VDB-003-shaped document string."""
    resolved = scores or {"c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5}
    return json.dumps(
        {"scores": resolved, "justification": justification}, sort_keys=True
    )


def calibration_metadata(
    domain: str = "CODE",
    level: str = "strong",
    scores: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build a VDB-004-shaped metadata dict."""
    resolved = scores or {"c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5}
    return {
        **resolved,
        "domain": domain,
        "level": level,
        "source": "train",
        "version": "pecs-v1.0",
    }


def query_result(
    ids: Optional[List[str]] = None,
    documents: Optional[List[str]] = None,
    metadatas: Optional[List[Dict[str, Any]]] = None,
    distances: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Build a ChromaDB-shaped query result for a single query."""
    resolved_ids = ids if ids is not None else ["rec-001"]
    return {
        "ids": [resolved_ids],
        "documents": [
            documents if documents is not None else [calibration_document()]
        ],
        "metadatas": [
            metadatas if metadatas is not None else [calibration_metadata()]
        ],
        "distances": [distances if distances is not None else [0.12]],
    }


def empty_query_result() -> Dict[str, Any]:
    """A ChromaDB miss (empty collection or zero-match filter)."""
    return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


# ---------------------------------------------------------------------------
# LLM fakes
# ---------------------------------------------------------------------------


class FakeUsage:
    def __init__(
        self,
        input_tokens: int = 1200,
        output_tokens: int = 300,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class FakeResponse:
    """Shaped like an ``anthropic.types.Message`` for the fields we read."""

    def __init__(
        self,
        text: str,
        model: str = "claude-haiku-4-5-20251001",
        stop_reason: str = "end_turn",
        usage: Optional[FakeUsage] = None,
    ) -> None:
        self.content = [FakeTextBlock(text)]
        self.model = model
        self.stop_reason = stop_reason
        self.usage = usage or FakeUsage()


class FakeMessages:
    def __init__(self, owner: "FakeClient") -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        self._owner.requests.append(kwargs)
        if not self._owner.responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        nxt = self._owner.responses.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt


class FakeClient:
    """Scripted client. Each entry in ``responses`` is returned or raised."""

    def __init__(self, responses: Optional[List[Any]] = None) -> None:
        self.responses: List[Any] = list(responses or [])
        self.requests: List[Dict[str, Any]] = []
        self.messages = FakeMessages(self)

    @property
    def call_count(self) -> int:
        return len(self.requests)


def evaluator_payload(
    count: int = 1,
    scores: Optional[Dict[str, float]] = None,
    include_summary: bool = True,
    drop_criterion: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a well-formed (or deliberately broken) evaluator JSON payload."""
    base = scores or {"c1": 8, "c2": 7, "c3": 6, "c4": 7, "c5": 9, "c6": 5}

    evaluations = []
    for index in range(1, count + 1):
        entry_scores = dict(base)
        entry_justifications = {
            criterion: f"Submission {index} {criterion.upper()} rationale."
            for criterion in CRITERIA
        }
        if drop_criterion:
            entry_scores.pop(drop_criterion, None)
            entry_justifications.pop(drop_criterion, None)
        evaluations.append(
            {
                "submission_index": index,
                "scores": entry_scores,
                "justifications": entry_justifications,
            }
        )

    payload: Dict[str, Any] = {"evaluations": evaluations}

    if include_summary:
        payload["overall_summary"] = {
            "strength": "Tasks are framed clearly.",
            "weakness": "Verification steps are thin.",
            "improvement": "Ask for an explicit self-check.",
        }

    return payload


def evaluator_response(
    count: int = 1,
    fenced: bool = False,
    **payload_kwargs: Any,
) -> FakeResponse:
    """Build a FakeResponse carrying a valid evaluator payload."""
    text = json.dumps(evaluator_payload(count=count, **payload_kwargs))
    if fenced:
        text = f"```json\n{text}\n```"
    return FakeResponse(text)
