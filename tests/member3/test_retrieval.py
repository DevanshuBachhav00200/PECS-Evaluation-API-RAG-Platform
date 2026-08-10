"""
Unit tests for VDB-009 — Top-K Retrieval Function.
"""

from __future__ import annotations

from typing import Any, Dict
import pytest
import chromadb
import numpy as np

from src.embeddings import embedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedding(seed: int) -> list[float]:
    """Return a deterministic 384-dimensional unit-normalised embedding."""
    rng = np.random.default_rng(seed)
    vec = rng.random(384).astype(np.float64)
    # L2-normalise so cosine and L2 distances agree on ordering
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


# Three clearly separated embeddings
EMBEDDING_A = _make_embedding(1)
EMBEDDING_B = _make_embedding(999)
EMBEDDING_C = _make_embedding(77777)

META_A: Dict[str, Any] = {
    "domain": "CODE", "level": "strong",
    "c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5,
    "source": "train", "version": "pecs-v1.0",
}
META_B: Dict[str, Any] = {
    "domain": "EDGE", "level": "weak",
    "c1": 2, "c2": 3, "c3": 4, "c4": 5, "c5": 6, "c6": 7,
    "source": "validation", "version": "pecs-v1.0",
}
META_C: Dict[str, Any] = {
    "domain": "CODE", "level": "developing",
    "c1": 5, "c2": 5, "c3": 5, "c4": 5, "c5": 5, "c6": 5,
    "source": "train", "version": "pecs-v2.0",
}

DOC_A = '{"scores": {"c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5}, "justification": "Record A"}'
DOC_B = '{"scores": {"c1": 2, "c2": 3, "c3": 4, "c4": 5, "c5": 6, "c6": 7}, "justification": "Record B"}'
DOC_C = '{"scores": {"c1": 5, "c2": 5, "c3": 5, "c4": 5, "c5": 5, "c6": 5}, "justification": "Record C"}'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def seeded_collection() -> chromadb.Collection:
    """Provides an isolated in-memory collection pre-seeded with 3 records."""
    from src.vectorstore import init, upsert
    client = chromadb.EphemeralClient()
    collection = init.get_or_create_collection(client)

    upsert.upsert_record(collection, "record-A", EMBEDDING_A, DOC_A, META_A)
    upsert.upsert_record(collection, "record-B", EMBEDDING_B, DOC_B, META_B)
    upsert.upsert_record(collection, "record-C", EMBEDDING_C, DOC_C, META_C)

    return collection


# ---------------------------------------------------------------------------
# Core nearest-neighbour tests
# ---------------------------------------------------------------------------

def test_query_own_vector_returns_itself_as_top(seeded_collection: chromadb.Collection) -> None:
    """Querying with embedding_A must return record-A as the top (closest) match."""
    from src.vectorstore.retrieve import retrieve

    result = retrieve(EMBEDDING_A, n_results=1, collection=seeded_collection)
    assert result["ids"][0][0] == "record-A"


def test_query_b_returns_record_b_as_top(seeded_collection: chromadb.Collection) -> None:
    """Querying with embedding_B must return record-B as the top match."""
    from src.vectorstore.retrieve import retrieve

    result = retrieve(EMBEDDING_B, n_results=1, collection=seeded_collection)
    assert result["ids"][0][0] == "record-B"


def test_query_c_returns_record_c_as_top(seeded_collection: chromadb.Collection) -> None:
    """Querying with embedding_C must return record-C as the top match."""
    from src.vectorstore.retrieve import retrieve

    result = retrieve(EMBEDDING_C, n_results=1, collection=seeded_collection)
    assert result["ids"][0][0] == "record-C"


# ---------------------------------------------------------------------------
# n_results behavior
# ---------------------------------------------------------------------------

def test_default_n_results_is_1(seeded_collection: chromadb.Collection) -> None:
    """Default n_results must return exactly 1 record."""
    from src.vectorstore.retrieve import retrieve

    result = retrieve(EMBEDDING_A, collection=seeded_collection)
    assert len(result["ids"][0]) == 1


def test_n_results_2_returns_two(seeded_collection: chromadb.Collection) -> None:
    """n_results=2 must return exactly 2 records."""
    from src.vectorstore.retrieve import retrieve

    result = retrieve(EMBEDDING_A, n_results=2, collection=seeded_collection)
    assert len(result["ids"][0]) == 2


def test_n_results_3_returns_three(seeded_collection: chromadb.Collection) -> None:
    """n_results=3 must return exactly 3 records."""
    from src.vectorstore.retrieve import retrieve

    result = retrieve(EMBEDDING_A, n_results=3, collection=seeded_collection)
    assert len(result["ids"][0]) == 3


# ---------------------------------------------------------------------------
# Result structure tests
# ---------------------------------------------------------------------------

def test_returned_documents_correspond_to_correct_ids(seeded_collection: chromadb.Collection) -> None:
    """Documents returned in a top-3 query must match the expected IDs."""
    from src.vectorstore.retrieve import retrieve

    result = retrieve(EMBEDDING_A, n_results=3, collection=seeded_collection)
    ids = result["ids"][0]
    docs = result["documents"][0]

    doc_map = dict(zip(ids, docs))
    assert doc_map["record-A"] == DOC_A
    assert doc_map["record-B"] == DOC_B
    assert doc_map["record-C"] == DOC_C


def test_metadata_is_preserved(seeded_collection: chromadb.Collection) -> None:
    """Metadata returned for record-A must exactly match what was stored."""
    from src.vectorstore.retrieve import retrieve

    result = retrieve(EMBEDDING_A, n_results=1, collection=seeded_collection)
    assert result["metadatas"][0][0] == META_A


def test_distances_are_present(seeded_collection: chromadb.Collection) -> None:
    """Distances must be returned and be non-negative floats."""
    from src.vectorstore.retrieve import retrieve

    result = retrieve(EMBEDDING_A, n_results=1, collection=seeded_collection)
    assert "distances" in result
    assert result["distances"] is not None
    dist = result["distances"][0][0]
    assert isinstance(dist, float)
    assert dist >= 0.0


def test_own_vector_distance_is_smallest(seeded_collection: chromadb.Collection) -> None:
    """Querying embedding_A: distance to record-A must be smaller than to B and C."""
    from src.vectorstore.retrieve import retrieve

    result = retrieve(EMBEDDING_A, n_results=3, collection=seeded_collection)
    ids = result["ids"][0]
    distances = result["distances"][0]  # type: ignore[index]

    dist_map = dict(zip(ids, distances))
    assert dist_map["record-A"] < dist_map["record-B"]
    assert dist_map["record-A"] < dist_map["record-C"]


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

def test_none_embedding_raises_value_error(seeded_collection: chromadb.Collection) -> None:
    from src.vectorstore.retrieve import retrieve
    with pytest.raises(ValueError):
        retrieve(None, collection=seeded_collection)  # type: ignore


def test_empty_embedding_raises_value_error(seeded_collection: chromadb.Collection) -> None:
    from src.vectorstore.retrieve import retrieve
    with pytest.raises(ValueError):
        retrieve([], collection=seeded_collection)


def test_non_numeric_embedding_raises_type_error(seeded_collection: chromadb.Collection) -> None:
    from src.vectorstore.retrieve import retrieve
    bad = ["not", "a", "number"] + [0.0] * 381
    with pytest.raises(TypeError):
        retrieve(bad, collection=seeded_collection)  # type: ignore


def test_wrong_dimension_raises_value_error(seeded_collection: chromadb.Collection) -> None:
    from src.vectorstore.retrieve import retrieve
    with pytest.raises(ValueError, match="384"):
        retrieve([0.1] * 100, collection=seeded_collection)


def test_n_results_zero_raises_value_error(seeded_collection: chromadb.Collection) -> None:
    from src.vectorstore.retrieve import retrieve
    with pytest.raises(ValueError):
        retrieve(EMBEDDING_A, n_results=0, collection=seeded_collection)


def test_n_results_negative_raises_value_error(seeded_collection: chromadb.Collection) -> None:
    from src.vectorstore.retrieve import retrieve
    with pytest.raises(ValueError):
        retrieve(EMBEDDING_A, n_results=-1, collection=seeded_collection)


def test_n_results_float_raises_type_error(seeded_collection: chromadb.Collection) -> None:
    from src.vectorstore.retrieve import retrieve
    with pytest.raises(TypeError):
        retrieve(EMBEDDING_A, n_results=1.5, collection=seeded_collection)  # type: ignore


def test_n_results_string_raises_type_error(seeded_collection: chromadb.Collection) -> None:
    from src.vectorstore.retrieve import retrieve
    with pytest.raises(TypeError):
        retrieve(EMBEDDING_A, n_results="1", collection=seeded_collection)  # type: ignore


# ---------------------------------------------------------------------------
# Architecture boundary tests
# ---------------------------------------------------------------------------

def test_retrieve_does_not_instantiate_persistent_client(
    seeded_collection: chromadb.Collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """retrieve() must not create a new PersistentClient when a collection is supplied."""
    from src.vectorstore.retrieve import retrieve

    instantiated = False

    def mock_persistent_client(*args: Any, **kwargs: Any) -> Any:
        nonlocal instantiated
        instantiated = True
        return chromadb.EphemeralClient()

    monkeypatch.setattr(chromadb, "PersistentClient", mock_persistent_client)

    retrieve(EMBEDDING_A, collection=seeded_collection)
    assert not instantiated, "retrieve() must not instantiate PersistentClient"


def test_retrieve_does_not_load_sentence_transformer(
    seeded_collection: chromadb.Collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """retrieve() must not load SentenceTransformer."""
    from src.vectorstore.retrieve import retrieve

    loaded = False

    def mock_st(*args: Any, **kwargs: Any) -> Any:
        nonlocal loaded
        loaded = True
        return embedder.get_embedder()

    monkeypatch.setattr(embedder, "SentenceTransformer", mock_st)

    retrieve(EMBEDDING_A, collection=seeded_collection)
    assert not loaded, "retrieve() must not load SentenceTransformer"


def test_production_db_not_modified(
    seeded_collection: chromadb.Collection,
) -> None:
    """All queries run against the isolated EphemeralClient; production DB is untouched."""
    from src.vectorstore.retrieve import retrieve

    # Passes collection explicitly — the production PersistentClient is never touched
    result = retrieve(EMBEDDING_A, n_results=1, collection=seeded_collection)
    assert result["ids"][0] == ["record-A"]
