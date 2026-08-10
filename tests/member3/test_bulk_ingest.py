"""
Unit tests for VDB-012 — Bulk Ingestion from Serialised Embeddings.
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
    """Deterministic 384-dimensional unit-normalised embedding."""
    rng = np.random.default_rng(seed)
    vec = rng.random(384).astype(np.float64)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


EMB_A = _make_embedding(1)
EMB_B = _make_embedding(999)
EMB_C = _make_embedding(77777)

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

DOC_A = '{"scores": {"c1": 8}, "justification": "Record A"}'
DOC_B = '{"scores": {"c1": 2}, "justification": "Record B"}'
DOC_C = '{"scores": {"c1": 5}, "justification": "Record C"}'

VALID_PAYLOAD: Dict[str, Any] = {
    "ids":        ["rec-A", "rec-B", "rec-C"],
    "embeddings": [EMB_A, EMB_B, EMB_C],
    "metadatas":  [META_A, META_B, META_C],
    "documents":  [DOC_A, DOC_B, DOC_C],
}


@pytest.fixture
def col() -> chromadb.Collection:
    """Isolated in-memory collection."""
    from src.vectorstore import init
    client = chromadb.EphemeralClient()
    return init.get_or_create_collection(client)


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------

def test_bulk_ingest_returns_record_count(col: chromadb.Collection) -> None:
    """bulk_ingest_from_serialized must return the number of upserted records."""
    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized

    count = bulk_ingest_from_serialized(VALID_PAYLOAD, collection=col)
    assert count == 3


def test_bulk_ingest_all_records_present(col: chromadb.Collection) -> None:
    """All seeded records must be retrievable after bulk ingest."""
    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized

    bulk_ingest_from_serialized(VALID_PAYLOAD, collection=col)

    result = col.get(ids=["rec-A", "rec-B", "rec-C"])
    assert set(result["ids"]) == {"rec-A", "rec-B", "rec-C"}


def test_bulk_ingest_documents_preserved(col: chromadb.Collection) -> None:
    """Documents must be stored verbatim."""
    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized

    bulk_ingest_from_serialized(VALID_PAYLOAD, collection=col)

    result = col.get(ids=["rec-A", "rec-B", "rec-C"], include=["documents"])
    doc_map = dict(zip(result["ids"], result["documents"]))
    assert doc_map["rec-A"] == DOC_A
    assert doc_map["rec-B"] == DOC_B
    assert doc_map["rec-C"] == DOC_C


def test_bulk_ingest_metadata_preserved(col: chromadb.Collection) -> None:
    """All ten VDB-004 metadata fields must be stored intact for each record."""
    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized

    bulk_ingest_from_serialized(VALID_PAYLOAD, collection=col)

    result = col.get(ids=["rec-A"], include=["metadatas"])
    assert result["metadatas"][0] == META_A


def test_bulk_ingest_embeddings_stored(col: chromadb.Collection) -> None:
    """Embeddings must be stored and retrievable."""
    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized

    bulk_ingest_from_serialized(VALID_PAYLOAD, collection=col)

    result = col.get(ids=["rec-A"], include=["embeddings"])  # type: ignore
    assert result["embeddings"] is not None
    assert result["embeddings"][0] == pytest.approx(EMB_A)


def test_bulk_ingest_is_idempotent(col: chromadb.Collection) -> None:
    """Calling bulk_ingest_from_serialized twice on the same payload must not duplicate records."""
    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized

    bulk_ingest_from_serialized(VALID_PAYLOAD, collection=col)
    bulk_ingest_from_serialized(VALID_PAYLOAD, collection=col)

    all_ids = col.get()["ids"]
    assert len(all_ids) == 3
    assert set(all_ids) == {"rec-A", "rec-B", "rec-C"}


def test_bulk_ingest_second_call_updates_record(col: chromadb.Collection) -> None:
    """A second ingest of the same ID with different content must update, not duplicate."""
    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized

    bulk_ingest_from_serialized(VALID_PAYLOAD, collection=col)

    updated_payload: Dict[str, Any] = {
        "ids":        ["rec-A"],
        "embeddings": [EMB_C],   # different embedding
        "metadatas":  [META_C],  # different metadata
        "documents":  [DOC_C],   # different document
    }
    bulk_ingest_from_serialized(updated_payload, collection=col)

    all_ids = col.get()["ids"]
    assert all_ids.count("rec-A") == 1  # still exactly one

    result = col.get(ids=["rec-A"], include=["documents", "metadatas"])
    assert result["documents"][0] == DOC_C
    assert result["metadatas"][0] == META_C


def test_bulk_ingest_empty_payload_returns_zero(col: chromadb.Collection) -> None:
    """An empty payload must return 0 without error and must not modify the collection."""
    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized

    count_before = len(col.get()["ids"])

    count = bulk_ingest_from_serialized(
        {"ids": [], "embeddings": [], "metadatas": [], "documents": []},
        collection=col,
    )
    assert count == 0
    # Collection must be unchanged
    count_after = len(col.get()["ids"])
    assert count_after == count_before


def test_bulk_ingest_nearest_retrieval_correct(col: chromadb.Collection) -> None:
    """After bulk ingest, retrieve() must return the correct nearest record."""
    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized
    from src.vectorstore.retrieve import retrieve

    bulk_ingest_from_serialized(VALID_PAYLOAD, collection=col)

    result = retrieve(EMB_B, n_results=1, collection=col)
    assert result["ids"][0][0] == "rec-B"


# ---------------------------------------------------------------------------
# Payload validation
# ---------------------------------------------------------------------------

def test_non_dict_payload_raises_type_error(col: chromadb.Collection) -> None:
    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized
    with pytest.raises(TypeError):
        bulk_ingest_from_serialized(["ids", "embeddings"], collection=col)  # type: ignore


def test_missing_key_raises_value_error(col: chromadb.Collection) -> None:
    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized
    bad = {"ids": ["x"], "embeddings": [EMB_A], "metadatas": [META_A]}  # missing documents
    with pytest.raises(ValueError, match="documents"):
        bulk_ingest_from_serialized(bad, collection=col)


def test_mismatched_list_lengths_raises_value_error(col: chromadb.Collection) -> None:
    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized
    bad = {
        "ids":        ["rec-A", "rec-B"],
        "embeddings": [EMB_A],           # length 1 vs ids length 2
        "metadatas":  [META_A, META_B],
        "documents":  [DOC_A, DOC_B],
    }
    with pytest.raises(ValueError):
        bulk_ingest_from_serialized(bad, collection=col)


def test_empty_id_raises_value_error(col: chromadb.Collection) -> None:
    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized
    bad = {
        "ids":        [""],
        "embeddings": [EMB_A],
        "metadatas":  [META_A],
        "documents":  [DOC_A],
    }
    with pytest.raises(ValueError):
        bulk_ingest_from_serialized(bad, collection=col)


def test_wrong_embedding_dimension_raises_value_error(col: chromadb.Collection) -> None:
    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized
    bad = {
        "ids":        ["rec-X"],
        "embeddings": [[0.1] * 100],     # wrong dimension
        "metadatas":  [META_A],
        "documents":  [DOC_A],
    }
    with pytest.raises(ValueError, match="384"):
        bulk_ingest_from_serialized(bad, collection=col)


def test_invalid_metadata_raises_value_error(col: chromadb.Collection) -> None:
    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized
    bad = {
        "ids":        ["rec-X"],
        "embeddings": [EMB_A],
        "metadatas":  [{"domain": "CODE"}],  # incomplete metadata
        "documents":  [DOC_A],
    }
    with pytest.raises(ValueError):
        bulk_ingest_from_serialized(bad, collection=col)


def test_non_numeric_embedding_raises_type_error(col: chromadb.Collection) -> None:
    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized
    bad_emb = ["x"] * 384
    bad = {
        "ids":        ["rec-X"],
        "embeddings": [bad_emb],
        "metadatas":  [META_A],
        "documents":  [DOC_A],
    }
    with pytest.raises(TypeError):
        bulk_ingest_from_serialized(bad, collection=col)


# ---------------------------------------------------------------------------
# Architecture boundary
# ---------------------------------------------------------------------------

def test_no_persistent_client_created(
    col: chromadb.Collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bulk_ingest_from_serialized must not instantiate PersistentClient when collection supplied."""
    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized

    created = False

    def mock_pc(*args: Any, **kwargs: Any) -> Any:
        nonlocal created
        created = True
        return chromadb.EphemeralClient()

    monkeypatch.setattr(chromadb, "PersistentClient", mock_pc)
    bulk_ingest_from_serialized(VALID_PAYLOAD, collection=col)
    assert not created


def test_no_sentence_transformer_loaded(
    col: chromadb.Collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bulk_ingest_from_serialized must not load SentenceTransformer."""
    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized

    loaded = False

    def mock_st(*args: Any, **kwargs: Any) -> Any:
        nonlocal loaded
        loaded = True
        return embedder.get_embedder()

    monkeypatch.setattr(embedder, "SentenceTransformer", mock_st)
    bulk_ingest_from_serialized(VALID_PAYLOAD, collection=col)
    assert not loaded
