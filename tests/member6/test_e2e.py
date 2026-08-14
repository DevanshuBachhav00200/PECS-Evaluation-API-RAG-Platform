"""
TEST-007 — End-to-end pipeline tests.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Covers the full path: candidate prompt → query preprocessing → embedding →
ChromaDB retrieval → calibration formatting → batch assembly → LLM call →
parsing → validation → scoring → skill band → API response.

Acceptance (TEST-007): a single test invokes the real pipeline against a known
prompt and asserts a valid structured response.
``test_single_prompt_travels_the_whole_pipeline`` is that test.

What is faked, and why only that: the **network hop to Anthropic**, and nothing
else. The scripted client is installed at ``llm_call._client``, so request
assembly, cache markers, fence stripping, schema validation, retries, the score
guard, the composite, the ethics gate and the response model all execute for
real. Mocking ``evaluate_batch_safe`` instead — as the inherited API tests do,
correctly, for their own purpose — would make this a test of the mock.

The vector store is injected as an in-memory collection seeded with one known
calibration record, so the test asserts real retrieval without depending on
whatever happens to be in the developer's ``Database/`` directory.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List

import pytest

pytest.importorskip("chromadb", reason="chromadb is not installed")
pytest.importorskip("sentence_transformers", reason="sentence-transformers is not installed")

import chromadb  # noqa: E402

from m4_fakes import FakeClient, FakeResponse, evaluator_payload, evaluator_response  # noqa: E402
from src.embeddings.batch_embed import embed_texts  # noqa: E402
from src.evaluation.engine import evaluate_batch, evaluate_batch_safe  # noqa: E402
from src.vectorstore.document_format import serialize_document  # noqa: E402
from src.vectorstore.metadata_schema import build_metadata  # noqa: E402
from src.vectorstore.upsert import upsert_record  # noqa: E402

KNOWN_PROMPT = (
    "You are a senior API reviewer. Review the endpoint below and list every "
    "defect by severity, with a concrete fix for each and the assumption it "
    "rests on."
)
NEIGHBOUR_PROMPT = (
    "Act as an API reviewer. Go through this endpoint and report the problems "
    "in order of severity, suggesting a fix for each one."
)

CALIBRATION_SCORES = {"c1": 8.0, "c2": 7.0, "c3": 9.0, "c4": 6.0, "c5": 10.0, "c6": 5.0}
CALIBRATION_JUSTIFICATION = "Clear role and severity ordering; verification step is thin."


@pytest.fixture
def seeded_store() -> Iterator[Any]:
    """An in-memory collection holding one real, correctly shaped record."""
    client = chromadb.EphemeralClient()
    name = f"m6_e2e_{id(client):x}"
    collection = client.create_collection(name=name)

    upsert_record(
        collection,
        "calib-review-001",
        embed_texts([KNOWN_PROMPT])[0].tolist(),
        serialize_document(CALIBRATION_SCORES, CALIBRATION_JUSTIFICATION),
        build_metadata(
            domain="CODE",
            level="strong",
            scores=CALIBRATION_SCORES,
            source="train",
            version="pecs-v1.0",
        ),
    )

    yield collection

    client.delete_collection(name=name)


def _qa_pair(prompt: str = NEIGHBOUR_PROMPT) -> Dict[str, str]:
    return {
        "task": "Ask the model to review an API endpoint.",
        "student_prompt": prompt,
    }


# ---------------------------------------------------------------------------
# The acceptance test
# ---------------------------------------------------------------------------


def test_single_prompt_travels_the_whole_pipeline(seeded_store: Any) -> None:
    """One prompt in, one schema-valid structured evaluation out."""
    client = FakeClient([evaluator_response(count=1)])

    result = evaluate_batch([_qa_pair()], collection=seeded_store, client=client)

    assert client.call_count == 1

    evaluation = result["evaluations"][0]
    assert evaluation["submission_index"] == 1
    assert set(evaluation["scores"]) == {"c1", "c2", "c3", "c4", "c5", "c6"}
    assert 0.0 <= evaluation["composite_score"] <= 10.0
    assert evaluation["skill_level"] in {"weak", "developing", "strong"}
    assert evaluation["model_version"]
    assert evaluation["rubric_version"]

    # Retrieval really ran: the seeded record was found and formatted in.
    assert evaluation["calibration_used"] is True
    assert evaluation["calibration_record_id"] == "calib-review-001"


def test_retrieved_calibration_reaches_the_model_request(seeded_store: Any) -> None:
    """The formatted calibration line is present in the request actually sent.

    Without this, "calibration_used: true" could be a bookkeeping flag on a
    request that never carried the example.
    """
    client = FakeClient([evaluator_response(count=1)])
    evaluate_batch([_qa_pair()], collection=seeded_store, client=client)

    sent = json.dumps(client.requests[0])
    assert "Scores: C1:8" in sent
    assert CALIBRATION_JUSTIFICATION in sent


def test_batch_of_three_is_one_call_with_three_results(seeded_store: Any) -> None:
    client = FakeClient([evaluator_response(count=3)])

    result = evaluate_batch(
        [_qa_pair(), _qa_pair("Review the endpoint."), _qa_pair("check the api")],
        collection=seeded_store,
        client=client,
    )

    assert client.call_count == 1
    assert [entry["submission_index"] for entry in result["evaluations"]] == [1, 2, 3]
    assert result["meta"]["submissions"] == 3


def test_scores_are_the_ones_the_evaluator_returned(seeded_store: Any) -> None:
    """The pipeline must not quietly rescale or substitute the model's scores."""
    scores = {"c1": 9, "c2": 8, "c3": 8, "c4": 9, "c5": 9, "c6": 7}
    client = FakeClient([evaluator_response(count=1, scores=scores)])

    evaluation = evaluate_batch(
        [_qa_pair()], collection=seeded_store, client=client
    )["evaluations"][0]

    assert evaluation["scores"] == {key: float(value) for key, value in scores.items()}
    # 1.35+1.60+1.60+1.80+1.35+0.70 = 8.40, hand-computed.
    assert evaluation["composite_score"] == 8.40
    assert evaluation["skill_level"] == "strong"


def test_empty_store_produces_an_uncalibrated_result_not_a_crash() -> None:
    """RAG-006: no calibration available is a documented state, not a failure."""
    client_db = chromadb.EphemeralClient()
    name = f"m6_e2e_empty_{id(client_db):x}"
    empty = client_db.create_collection(name=name)

    try:
        result = evaluate_batch(
            [_qa_pair()],
            collection=empty,
            client=FakeClient([evaluator_response(count=1)]),
        )
        evaluation = result["evaluations"][0]
        assert evaluation["calibration_used"] is False
        assert evaluation["composite_score"] is not None
    finally:
        client_db.delete_collection(name=name)


def test_pipeline_failure_returns_a_structured_error(seeded_store: Any) -> None:
    """``evaluate_batch_safe`` is the API's entry point; it must never raise."""
    client = FakeClient([FakeResponse("this is not json")] * 5)

    result = evaluate_batch_safe(
        [_qa_pair()], collection=seeded_store, client=client, sleep=lambda _s: None
    )

    assert "error" in result
    assert result["error"]["code"]
    assert "Traceback" not in result["error"]["message"]


# ---------------------------------------------------------------------------
# End-to-end through the HTTP layer (API-005 + the whole engine beneath it)
# ---------------------------------------------------------------------------


def test_evaluate_endpoint_runs_the_real_engine(
    api_client: Any,
    auth_headers: Dict[str, str],
    seeded_store: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /evaluate → real RAG + evaluation stack → schema-valid 200 body.

    The engine is called through the endpoint, not mocked at it: only the
    Anthropic client and the vector-store collection are substituted.
    """
    from src.api import main as api_main
    from src.evaluation import engine as engine_module
    from src.rag import llm_call

    monkeypatch.setattr(llm_call, "_client", FakeClient([evaluator_response(count=1)]))

    def _with_store(qa_pairs: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        kwargs.setdefault("collection", seeded_store)
        return engine_module.evaluate_batch_safe(qa_pairs, **kwargs)

    monkeypatch.setattr(api_main, "evaluate_batch_safe", _with_store)

    response = api_client.post(
        "/evaluate",
        json={
            "submissions": [
                {
                    "task": "Ask the model to review an API endpoint.",
                    "student_prompt": NEIGHBOUR_PROMPT,
                    "domain": "CODE",
                    "level": "strong",
                }
            ]
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()

    from src.api.schemas import EvaluationResponse

    validated = EvaluationResponse.model_validate(body)
    assert len(validated.evaluations) == 1
    assert validated.evaluations[0].calibration_record_id == "calib-review-001"
    assert validated.meta.submissions == 1
    assert validated.rubric_version


def test_health_endpoint_reports_the_warm_embedder(api_client: Any) -> None:
    """Gate 6's container health check depends on this staying honest."""
    response = api_client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "healthy"
    assert body["embedder_loaded"] is True
    assert body["version"]
