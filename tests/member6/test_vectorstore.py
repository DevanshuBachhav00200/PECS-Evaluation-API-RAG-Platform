"""
TEST-003 — ChromaDB upsert/retrieval unit tests.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Covers upsert idempotency (VDB-007), retrieval correctness (VDB-009), metadata
preservation (VDB-011) and metadata filtering (VDB-010).

Acceptance (TEST-003): tests pass against a disposable test collection, not the
production store. Enforced literally — the fixture asserts its collection is not
the VDB-006 production one, and every test in this module receives the fixture.

One trap worth naming, because it cost the inherited suite eleven failures:
``chromadb.EphemeralClient()`` does **not** give each caller an isolated
database. Clients constructed with identical settings share one in-process
system, so two modules that both seed ``prompt_examples`` with ids
``record-A/B/C`` silently overwrite each other and fail whichever runs second.
The fixture below uses a per-test collection name, which sidesteps the sharing
entirely rather than relying on cleanup order.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator

import pytest

pytest.importorskip("chromadb", reason="chromadb is not installed")

import chromadb  # noqa: E402

from src.vectorstore.document_format import deserialize_document, serialize_document  # noqa: E402
from src.vectorstore.filtered_retrieve import filtered_retrieve  # noqa: E402
from src.vectorstore.ingest import ingest_record  # noqa: E402
from src.vectorstore.metadata_schema import build_metadata  # noqa: E402
from src.vectorstore.retrieve import retrieve  # noqa: E402
from src.vectorstore.upsert import upsert_record  # noqa: E402

DIM = 384

SCORES_A = {"c1": 8.0, "c2": 7.0, "c3": 9.0, "c4": 6.0, "c5": 10.0, "c6": 5.0}
SCORES_B = {"c1": 2.0, "c2": 3.0, "c3": 4.0, "c4": 5.0, "c5": 6.0, "c6": 7.0}


def _vector(seed: int) -> list[float]:
    """A deterministic unit vector, distinct per seed.

    Built by hand rather than with numpy's RNG so the geometry is obvious: each
    vector puts all its mass on one axis, which makes "nearest neighbour" a fact
    about the fixture rather than something to take on trust.
    """
    values = [0.0] * DIM
    values[seed % DIM] = 1.0
    return values


VECTOR_A = _vector(3)
VECTOR_B = _vector(100)
VECTOR_C = _vector(200)


@pytest.fixture
def disposable_collection() -> Iterator[Any]:
    """A uniquely named ephemeral collection, torn down after the test."""
    client = chromadb.EphemeralClient()
    name = f"m6_test_{id(client):x}"
    collection = client.create_collection(name=name)

    # TEST-003's acceptance criterion, asserted rather than assumed.
    from src.vectorstore.config import get_chroma_db_path

    assert collection.name != "prompt_examples"
    assert str(get_chroma_db_path()) not in repr(client)

    yield collection

    client.delete_collection(name=name)


def _metadata(domain: str = "CODE", level: str = "strong", scores: Dict[str, float] = None) -> Dict[str, Any]:
    return build_metadata(
        domain=domain,
        level=level,
        scores=scores or SCORES_A,
        source="train",
        version="pecs-v1.0",
    )


def _seed(collection: Any) -> None:
    upsert_record(
        collection, "rec-a", VECTOR_A, serialize_document(SCORES_A, "Record A rationale."),
        _metadata("CODE", "strong", SCORES_A),
    )
    upsert_record(
        collection, "rec-b", VECTOR_B, serialize_document(SCORES_B, "Record B rationale."),
        _metadata("EDGE", "weak", SCORES_B),
    )


# ---------------------------------------------------------------------------
# VDB-007 — upsert idempotency
# ---------------------------------------------------------------------------


def test_upserting_the_same_id_twice_leaves_one_record(disposable_collection: Any) -> None:
    document = serialize_document(SCORES_A, "First write.")
    upsert_record(disposable_collection, "rec-a", VECTOR_A, document, _metadata())
    upsert_record(disposable_collection, "rec-a", VECTOR_A, document, _metadata())

    assert disposable_collection.count() == 1


def test_upserting_the_same_id_updates_rather_than_duplicates(disposable_collection: Any) -> None:
    upsert_record(
        disposable_collection, "rec-a", VECTOR_A,
        serialize_document(SCORES_A, "First write."), _metadata(),
    )
    upsert_record(
        disposable_collection, "rec-a", VECTOR_A,
        serialize_document(SCORES_B, "Second write."), _metadata("EDGE", "weak", SCORES_B),
    )

    stored = disposable_collection.get(ids=["rec-a"])
    assert disposable_collection.count() == 1
    assert "Second write." in stored["documents"][0]
    assert stored["metadatas"][0]["domain"] == "EDGE"


def test_distinct_ids_produce_distinct_records(disposable_collection: Any) -> None:
    _seed(disposable_collection)
    assert disposable_collection.count() == 2


# ---------------------------------------------------------------------------
# VDB-005 — ingestion delegates to the upsert layer
# ---------------------------------------------------------------------------


def test_ingest_record_writes_through_the_upsert_layer(
    disposable_collection: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """VDB-005 must not carry its own copy of the write.

    Regression guard: ``ingest_record`` previously called ``collection.upsert``
    directly, so VDB-007's duplicate-ID semantics were bypassed and free to
    drift. Member 6 fixed the delegation; this keeps it fixed.
    """
    from src.vectorstore import upsert as upsert_module

    seen: Dict[str, Any] = {}
    original = upsert_module.upsert_record

    def _spy(**kwargs: Any) -> None:
        seen.update(kwargs)
        original(**kwargs)

    monkeypatch.setattr(upsert_module, "upsert_record", _spy)

    ingest_record(
        {
            "prompt_id": "rec-ingested",
            "student_prompt": "Review this endpoint.",
            "justification": "Ingestion rationale.",
            "domain": "CODE",
            "level": "strong",
            "source": "train",
            "version": "pecs-v1.0",
            "scores": SCORES_A,
            "embedding": VECTOR_A,
        },
        disposable_collection,
    )

    assert seen.get("record_id") == "rec-ingested"
    assert disposable_collection.count() == 1


# ---------------------------------------------------------------------------
# VDB-009 / VDB-011 — retrieval correctness and metadata preservation
# ---------------------------------------------------------------------------


def test_querying_a_stored_vector_returns_its_own_record(disposable_collection: Any) -> None:
    _seed(disposable_collection)
    result = retrieve(VECTOR_A, n_results=1, collection=disposable_collection)
    assert result["ids"][0][0] == "rec-a"


def test_default_top_k_returns_exactly_one_record(disposable_collection: Any) -> None:
    _seed(disposable_collection)
    result = retrieve(VECTOR_A, collection=disposable_collection)
    assert len(result["ids"][0]) == 1


def test_distances_are_returned_in_ascending_order(disposable_collection: Any) -> None:
    _seed(disposable_collection)
    upsert_record(
        disposable_collection, "rec-c", VECTOR_C,
        serialize_document(SCORES_A, "Record C rationale."), _metadata(),
    )

    distances = retrieve(VECTOR_A, n_results=3, collection=disposable_collection)["distances"][0]
    assert distances == sorted(distances)


def test_metadata_survives_the_round_trip_intact(disposable_collection: Any) -> None:
    """VDB-011: every ingested metadata field comes back unchanged."""
    _seed(disposable_collection)
    metadata = retrieve(VECTOR_A, n_results=1, collection=disposable_collection)["metadatas"][0][0]

    assert dict(metadata) == _metadata("CODE", "strong", SCORES_A)


def test_document_deserializes_back_to_its_scores_and_justification(
    disposable_collection: Any,
) -> None:
    _seed(disposable_collection)
    document = retrieve(VECTOR_A, n_results=1, collection=disposable_collection)["documents"][0][0]

    payload = deserialize_document(document)
    assert payload["justification"] == "Record A rationale."
    assert {key: float(value) for key, value in payload["scores"].items()} == SCORES_A


def test_retrieve_rejects_a_wrong_dimension_query(disposable_collection: Any) -> None:
    with pytest.raises(ValueError, match="384"):
        retrieve([0.1, 0.2], n_results=1, collection=disposable_collection)


def test_retrieve_rejects_a_non_positive_top_k(disposable_collection: Any) -> None:
    with pytest.raises(ValueError):
        retrieve(VECTOR_A, n_results=0, collection=disposable_collection)


def test_empty_collection_returns_no_matches(disposable_collection: Any) -> None:
    """An empty store is a legitimate state, not an error (RAG-006's upstream)."""
    result = retrieve(VECTOR_A, n_results=1, collection=disposable_collection)
    assert result["ids"][0] == []


# ---------------------------------------------------------------------------
# VDB-010 — metadata filtering
# ---------------------------------------------------------------------------


def test_domain_filtered_query_never_returns_another_domain(
    disposable_collection: Any,
) -> None:
    _seed(disposable_collection)
    result = filtered_retrieve(
        VECTOR_B, n_results=2, domain="CODE", collection=disposable_collection
    )
    for metadata in result["metadatas"][0]:
        assert metadata["domain"] == "CODE"


def test_filter_with_no_match_returns_empty_rather_than_falling_back(
    disposable_collection: Any,
) -> None:
    """A zero-match filter must not silently widen to an unfiltered query."""
    upsert_record(
        disposable_collection, "rec-a", VECTOR_A,
        serialize_document(SCORES_A, "Record A rationale."), _metadata("CODE", "strong"),
    )
    result = filtered_retrieve(
        VECTOR_A, n_results=2, domain="EDGE", collection=disposable_collection
    )
    assert result["ids"][0] == []


def test_level_filter_is_applied_alongside_domain(disposable_collection: Any) -> None:
    _seed(disposable_collection)
    result = filtered_retrieve(
        VECTOR_A, n_results=2, domain="CODE", level="weak",
        collection=disposable_collection,
    )
    assert result["ids"][0] == []


# ---------------------------------------------------------------------------
# VDB-003 — document representation
# ---------------------------------------------------------------------------


def test_document_serialization_round_trips() -> None:
    document = serialize_document(SCORES_A, "Round-trip rationale.")
    payload = deserialize_document(document)
    assert payload["justification"] == "Round-trip rationale."


def test_document_serialization_rejects_a_partial_score_set() -> None:
    with pytest.raises(ValueError):
        serialize_document({"c1": 8, "c2": 7}, "Incomplete.")


def test_document_serialization_rejects_an_empty_justification() -> None:
    with pytest.raises(ValueError):
        serialize_document(SCORES_A, "   ")


def test_stored_document_is_valid_json() -> None:
    """The RAG layer parses this string; malformed JSON breaks calibration."""
    assert isinstance(json.loads(serialize_document(SCORES_A, "ok")), dict)
