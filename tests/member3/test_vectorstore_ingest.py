"""
Unit tests for VDB-005 vector-store embedding ingestion logic.
"""

from __future__ import annotations

from typing import Any
import sys
import pytest
import chromadb

from src.vectorstore import init, ingest, document_format, metadata_schema
from src.embeddings import batch_embed, embedder


@pytest.fixture
def test_collection() -> chromadb.Collection:
    """Provides a fresh isolated in-memory test collection."""
    client = chromadb.EphemeralClient()
    # Ensure collection starts clean
    try:
        client.delete_collection(name="prompt_examples")
    except Exception:
        pass
    return init.get_or_create_collection(client)


@pytest.fixture
def valid_record() -> dict[str, Any]:
    return {
        "prompt_id": "PELS-L1-CODE-TEST-999",
        "student_prompt": "Write a python function to check if a number is prime.",
        "domain": "CODE",
        "level": "strong",
        "justification": "This is a prime check verification example.",
        "source": "train",
        "version": "pecs-v1.0",
        "scores": {"c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5},
    }


def test_valid_record_ingestion_success(
    test_collection: chromadb.Collection, valid_record: dict[str, Any]
) -> None:
    """Verify that a valid record embeds and ingests successfully with expected fields."""
    ingest.ingest_record(valid_record, test_collection)

    # Fetch it from collection
    results = test_collection.get(
        ids=[valid_record["prompt_id"]],
        include=["embeddings", "documents", "metadatas"],  # type: ignore
    )

    # 1. Deterministic ID match
    assert results["ids"] == [valid_record["prompt_id"]]

    # 2. Embedding dimension verification
    embeddings = results["embeddings"]
    assert embeddings is not None
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384

    # 3. Document deserialization validation
    documents = results["documents"]
    assert documents is not None
    assert len(documents) == 1
    parsed_doc = document_format.deserialize_document(documents[0])
    assert parsed_doc["justification"] == valid_record["justification"]
    assert parsed_doc["scores"] == valid_record["scores"]

    # 4. Metadata schema validation
    metadatas = results["metadatas"]
    assert metadatas is not None
    assert len(metadatas) == 1
    metadata_schema.validate_metadata(metadatas[0])
    assert metadatas[0]["domain"] == valid_record["domain"]
    assert metadatas[0]["level"] == valid_record["level"]
    assert metadatas[0]["source"] == valid_record["source"]
    assert metadatas[0]["version"] == valid_record["version"]


def test_idempotence_and_upsert_update(
    test_collection: chromadb.Collection, valid_record: dict[str, Any]
) -> None:
    """Verify that ingesting twice does not duplicate the entry, and re-ingesting updates the fields."""
    # 1. Ingest first time
    ingest.ingest_record(valid_record, test_collection)
    assert len(test_collection.get()["ids"]) == 1

    # 2. Ingest second time (exact same data)
    ingest.ingest_record(valid_record, test_collection)
    assert len(test_collection.get()["ids"]) == 1

    # 3. Ingest with modified content (verify update/upsert)
    modified_record = valid_record.copy()
    modified_record["justification"] = "Updated explanation."
    modified_record["scores"] = {"c1": 1, "c2": 2, "c3": 3, "c4": 4, "c5": 5, "c6": 6}

    ingest.ingest_record(modified_record, test_collection)
    assert len(test_collection.get()["ids"]) == 1

    # Check updated values
    results = test_collection.get(
        ids=[valid_record["prompt_id"]],
        include=["documents"],  # type: ignore
    )
    documents = results["documents"]
    assert documents is not None
    parsed_doc = document_format.deserialize_document(documents[0])
    assert parsed_doc["justification"] == "Updated explanation."
    assert parsed_doc["scores"] == modified_record["scores"]


def test_missing_prompt_id(
    test_collection: chromadb.Collection, valid_record: dict[str, Any]
) -> None:
    """Missing prompt_id must fail validation clearly."""
    bad_record = valid_record.copy()
    del bad_record["prompt_id"]
    with pytest.raises(ValueError, match="prompt_id"):
        ingest.ingest_record(bad_record, test_collection)


def test_missing_student_prompt(
    test_collection: chromadb.Collection, valid_record: dict[str, Any]
) -> None:
    """Missing student_prompt must fail validation clearly."""
    bad_record = valid_record.copy()
    del bad_record["student_prompt"]
    with pytest.raises(ValueError, match="student_prompt"):
        ingest.ingest_record(bad_record, test_collection)


def test_invalid_scores_rejected(
    test_collection: chromadb.Collection, valid_record: dict[str, Any]
) -> None:
    """Invalid scores (out of range) must be rejected through VDB-004 validation."""
    bad_record = valid_record.copy()
    bad_record["scores"] = {"c1": 11, "c2": 2, "c3": 3, "c4": 4, "c5": 5, "c6": 6}
    with pytest.raises(ValueError):
        ingest.ingest_record(bad_record, test_collection)


def test_empty_or_invalid_prompt_follows_emb007(
    test_collection: chromadb.Collection, valid_record: dict[str, Any]
) -> None:
    """Empty or whitespace-only prompt must follow EMB-007 behavior and raise an exception."""
    bad_record = valid_record.copy()
    bad_record["student_prompt"] = "    "
    with pytest.raises(ValueError, match="validation failed"):
        ingest.ingest_record(bad_record, test_collection)


def test_model_and_client_reuse(
    test_collection: chromadb.Collection, valid_record: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that ingest_record does not reinstantiate a new SentenceTransformer or PersistentClient."""
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

    ingest.ingest_record(valid_record, test_collection)

    assert not model_reinstantiated, "Ingestion must not load a new SentenceTransformer model"
    assert not client_reinstantiated, "Ingestion must not instantiate a ChromaDB client"


def test_pre_computed_embedding_reuse(
    test_collection: chromadb.Collection, valid_record: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that if an embedding list is pre-supplied in the record, it is validated and used directly."""
    # Seed a custom 384-dimensional list to make sure it's the one stored
    custom_emb = [0.1] * 384
    record_with_emb = valid_record.copy()
    record_with_emb["embedding"] = custom_emb

    # Mock embed_texts to guarantee it is not executed
    embed_texts_called = False

    def mock_embed_texts(*args: Any, **kwargs: Any) -> Any:
        nonlocal embed_texts_called
        embed_texts_called = True
        return [[0.0] * 384]

    monkeypatch.setattr(batch_embed, "embed_texts", mock_embed_texts)

    ingest.ingest_record(record_with_emb, test_collection)

    # Verify custom embedding is exactly the one stored
    results = test_collection.get(
        ids=[valid_record["prompt_id"]],
        include=["embeddings"],  # type: ignore
    )
    embeddings = results["embeddings"]
    assert embeddings is not None
    assert len(embeddings) == 1
    assert embeddings[0] == pytest.approx(custom_emb)
    assert not embed_texts_called, "embed_texts must not be called when embedding is pre-supplied"


def test_pre_computed_embedding_invalid_dimension(
    test_collection: chromadb.Collection, valid_record: dict[str, Any]
) -> None:
    """Invalid dimension in pre-computed embedding must raise a ValueError."""
    bad_emb_rec = valid_record.copy()
    bad_emb_rec["embedding"] = [0.1] * 100  # Wrong dimension (100 instead of 384)
    with pytest.raises(ValueError, match="embedding dimension"):
        ingest.ingest_record(bad_emb_rec, test_collection)
