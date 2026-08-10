"""
Unit tests for VDB-007 document upsert and duplicate-ID handling.
"""

from __future__ import annotations

from typing import Any, Dict, List
import sys
import pytest
import chromadb
import numpy as np

from src.embeddings import batch_embed, embedder


@pytest.fixture
def test_collection() -> chromadb.Collection:
    """Provides a fresh isolated in-memory test collection."""
    from src.vectorstore import init
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection(name="prompt_examples")
    except Exception:
        pass
    return init.get_or_create_collection(client)


@pytest.fixture
def valid_metadata() -> dict[str, Any]:
    return {
        "domain": "CODE",
        "level": "strong",
        "c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5,
        "source": "train",
        "version": "pecs-v1.0"
    }


@pytest.fixture
def valid_embedding() -> list[float]:
    # Generate a deterministic 384-dimensional vector using numpy without sentence_transformers
    rng = np.random.default_rng(42)
    vec = rng.random(384).astype(np.float32)
    return vec.tolist()


def test_first_upsert_creates_exactly_one_record(
    test_collection: chromadb.Collection,
    valid_embedding: list[float],
    valid_metadata: dict[str, Any]
) -> None:
    """Verify that a single upsert creates exactly one record in the collection."""
    from src.vectorstore import upsert, document_format

    doc_str = document_format.serialize_document(
        {"c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5},
        "Test justification."
    )

    upsert.upsert_record(
        collection=test_collection,
        record_id="PECS-TEST-ID-001",
        embedding=valid_embedding,
        document=doc_str,
        metadata=valid_metadata
    )

    results = test_collection.get(ids=["PECS-TEST-ID-001"], include=["embeddings", "documents", "metadatas"])  # type: ignore
    assert results["ids"] == ["PECS-TEST-ID-001"]
    assert len(results["ids"]) == 1
    assert results["documents"] == [doc_str]
    assert results["embeddings"] is not None
    assert results["embeddings"][0] == pytest.approx(valid_embedding)
    assert results["metadatas"] == [valid_metadata]


def test_idempotent_duplicate_upsert_replaces_and_updates(
    test_collection: chromadb.Collection,
    valid_embedding: list[float],
    valid_metadata: dict[str, Any]
) -> None:
    """Verify that upserting the same ID with different contents leaves exactly one updated record."""
    from src.vectorstore import upsert, document_format

    # 1. First upsert with content A
    doc_a = document_format.serialize_document(
        {"c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5},
        "Justification A"
    )
    upsert.upsert_record(
        collection=test_collection,
        record_id="PECS-DUP-ID-999",
        embedding=valid_embedding,
        document=doc_a,
        metadata=valid_metadata
    )

    # 2. Second upsert of the same ID with content B (different embedding, doc, metadata)
    new_embedding = [0.9] * 384
    doc_b = document_format.serialize_document(
        {"c1": 1, "c2": 2, "c3": 3, "c4": 4, "c5": 5, "c6": 6},
        "Justification B"
    )
    new_metadata = {
        "domain": "EDGE",
        "level": "weak",
        "c1": 1, "c2": 2, "c3": 3, "c4": 4, "c5": 5, "c6": 6,
        "source": "validation",
        "version": "pecs-v2.0"
    }

    upsert.upsert_record(
        collection=test_collection,
        record_id="PECS-DUP-ID-999",
        embedding=new_embedding,
        document=doc_b,
        metadata=new_metadata
    )

    # Verify that there is exactly one record in the collection
    all_records = test_collection.get()
    assert all_records["ids"] == ["PECS-DUP-ID-999"]
    assert len(all_records["ids"]) == 1

    # Verify that the record contains the NEW contents (B)
    results = test_collection.get(ids=["PECS-DUP-ID-999"], include=["embeddings", "documents", "metadatas"])  # type: ignore
    assert results["documents"] == [doc_b]
    embeddings_list = results["embeddings"]
    assert embeddings_list is not None
    assert embeddings_list[0] == pytest.approx(new_embedding)
    assert results["metadatas"] == [new_metadata]


def test_multiple_distinct_ids(
    test_collection: chromadb.Collection,
    valid_embedding: list[float],
    valid_metadata: dict[str, Any]
) -> None:
    """Verify that multiple distinct IDs create multiple records correctly."""
    from src.vectorstore import upsert

    doc = "serialized doc"
    upsert.upsert_record(test_collection, "ID-1", valid_embedding, doc, valid_metadata)
    upsert.upsert_record(test_collection, "ID-2", valid_embedding, doc, valid_metadata)

    all_ids = test_collection.get()["ids"]
    assert len(all_ids) == 2
    assert "ID-1" in all_ids
    assert "ID-2" in all_ids


def test_batch_upsert_support(
    test_collection: chromadb.Collection,
    valid_embedding: list[float],
    valid_metadata: dict[str, Any]
) -> None:
    """Verify that upsert_record supports batch upserts when lists are passed."""
    from src.vectorstore import upsert

    ids = ["BATCH-1", "BATCH-2"]
    embeddings = [valid_embedding, [0.5] * 384]
    documents = ["doc 1", "doc 2"]
    metadatas = [valid_metadata, valid_metadata]

    upsert.upsert_record(test_collection, ids, embeddings, documents, metadatas)

    all_ids = test_collection.get()["ids"]
    assert len(all_ids) == 2
    assert "BATCH-1" in all_ids
    assert "BATCH-2" in all_ids


def test_missing_or_empty_id(
    test_collection: chromadb.Collection,
    valid_embedding: list[float],
    valid_metadata: dict[str, Any]
) -> None:
    """Missing or empty ID must fail clearly."""
    from src.vectorstore import upsert

    doc = "serialized doc"
    # None ID
    with pytest.raises(TypeError):
        upsert.upsert_record(test_collection, None, valid_embedding, doc, valid_metadata)  # type: ignore

    # Empty string ID
    with pytest.raises(ValueError, match="ID"):
        upsert.upsert_record(test_collection, "", valid_embedding, doc, valid_metadata)

    # Whitespace ID
    with pytest.raises(ValueError, match="ID"):
        upsert.upsert_record(test_collection, "   ", valid_embedding, doc, valid_metadata)


def test_invalid_embedding_dimension(
    test_collection: chromadb.Collection,
    valid_metadata: dict[str, Any]
) -> None:
    """Invalid embedding dimension must fail validation clearly."""
    from src.vectorstore import upsert

    doc = "serialized doc"
    bad_emb = [0.1] * 100  # Wrong dimension (100 instead of 384)
    with pytest.raises(ValueError, match="dimensions"):
        upsert.upsert_record(test_collection, "ID-1", bad_emb, doc, valid_metadata)


def test_invalid_metadata_rejected(
    test_collection: chromadb.Collection,
    valid_embedding: list[float]
) -> None:
    """Invalid metadata must be rejected."""
    from src.vectorstore import upsert

    doc = "serialized doc"
    # Metadata missing keys
    bad_meta = {"domain": "CODE"}
    with pytest.raises(ValueError):
        upsert.upsert_record(test_collection, "ID-1", valid_embedding, doc, bad_meta)


def test_no_model_or_client_instantiation(
    test_collection: chromadb.Collection,
    valid_embedding: list[float],
    valid_metadata: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that upsert_record does not instantiate a PersistentClient or SentenceTransformer."""
    from src.vectorstore import upsert

    model_reinstantiated = False
    client_reinstantiated = False

    def mock_model_init(*args: Any, **kwargs: Any) -> Any:
        nonlocal model_reinstantiated
        model_reinstantiated = True
        return embedder.get_embedder()

    def mock_client_init(*args: Any, **kwargs: Any) -> Any:
        nonlocal client_reinstantiated
        client_reinstantiated = True
        return chromadb.EphemeralClient()

    monkeypatch.setattr(embedder, "SentenceTransformer", mock_model_init)
    monkeypatch.setattr(chromadb, "PersistentClient", mock_client_init)

    upsert.upsert_record(test_collection, "TEST-ID", valid_embedding, "doc", valid_metadata)

    assert not model_reinstantiated, "Upsert must not load SentenceTransformer"
    assert not client_reinstantiated, "Upsert must not instantiate PersistentClient"


def test_ingest_delegates_to_upsert(
    test_collection: chromadb.Collection,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that ingest_record correctly delegates its write operation to upsert_record."""
    from src.vectorstore import ingest, upsert

    called_upsert = False
    original_upsert = upsert.upsert_record

    def mock_upsert_record(
        collection: Any,
        record_id: str | List[str],
        embedding: List[float] | List[List[float]],
        document: str | List[str],
        metadata: Dict[str, Any] | List[Dict[str, Any]],
    ) -> None:
        nonlocal called_upsert
        called_upsert = True
        # Delegate to original upsert so collection still works
        original_upsert(collection, record_id, embedding, document, metadata)

    monkeypatch.setattr(upsert, "upsert_record", mock_upsert_record)

    record = {
        "prompt_id": "DELEGATION-ID",
        "student_prompt": "Prompt text",
        "domain": "CODE",
        "level": "strong",
        "justification": "justification",
        "source": "train",
        "version": "pecs-v1.0",
        "scores": {"c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5},
    }

    ingest.ingest_record(record, test_collection)
    assert called_upsert, "ingest_record must delegate write to upsert_record"
