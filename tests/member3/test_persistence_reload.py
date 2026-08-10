"""
Unit tests for VDB-008 — Reload-After-Restart Verification.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
import pytest
import chromadb

from src.vectorstore import document_format, metadata_schema


@pytest.fixture
def test_db_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Sets up an isolated temporary persistence path for testing process restarts."""
    path = tmp_path / "test_pels_reload_db"
    monkeypatch.setenv("CHROMA_DB_PATH", str(path))

    # Clear cached vectorstore modules to force clean module loading
    for k in list(sys.modules.keys()):
        if k.startswith("src.vectorstore"):
            sys.modules.pop(k, None)

    return path


def test_persistence_survives_restart_without_reingestion(test_db_path: Path) -> None:
    """
    Verify that records ingested into the persistent ChromaDB store survive
    a complete module/client restart without re-ingesting data.
    """
    # 1. Phase 1: Pre-restart setup and ingestion
    from src.vectorstore import client as client_phase1
    from src.vectorstore import init as init_phase1
    from src.vectorstore import upsert as upsert_phase1

    c1 = client_phase1.get_client()
    collection1 = init_phase1.get_or_create_collection(c1)

    record_id = "PECS-RELOAD-TEST-001"
    doc_str = document_format.serialize_document(
        {"c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5},
        "Justification prior to restart."
    )
    meta = metadata_schema.build_metadata(
        domain="CODE",
        level="strong",
        scores={"c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5},
        source="train",
        version="pecs-v1.0"
    )
    dummy_embedding = [0.123] * 384

    # Upsert the record before restart
    upsert_phase1.upsert_record(
        collection=collection1,
        record_id=record_id,
        embedding=dummy_embedding,
        document=doc_str,
        metadata=meta
    )

    # 2. Confirm record exists before restart
    pre_restart_res = collection1.get(ids=[record_id], include=["embeddings", "documents", "metadatas"])  # type: ignore
    assert pre_restart_res["ids"] == [record_id]
    assert pre_restart_res["documents"] == [doc_str]

    # 3. Phase 2: Simulate process/client restart
    # Clear client and vectorstore modules from sys.modules
    for k in list(sys.modules.keys()):
        if k.startswith("src.vectorstore"):
            sys.modules.pop(k, None)

    # Re-import client and init modules (simulates process restart)
    from src.vectorstore import client as client_phase2
    from src.vectorstore import init as init_phase2

    c2 = client_phase2.get_client()

    # Ensure c1 and c2 are separate client objects
    assert c1 is not c2

    # Obtain collection again without calling get_or_create_collection creation logic or ingestion
    collection2 = c2.get_collection(name=init_phase2.COLLECTION_NAME)

    # 4. Post-restart verification (WITHOUT calling ingest_record or upsert_record)
    post_restart_res = collection2.get(ids=[record_id], include=["embeddings", "documents", "metadatas"])  # type: ignore

    # Verify ID
    assert post_restart_res["ids"] == [record_id]

    # Verify document
    assert post_restart_res["documents"] == [doc_str]
    doc_deserialized = document_format.deserialize_document(post_restart_res["documents"][0])
    assert doc_deserialized["justification"] == "Justification prior to restart."

    # Verify embedding
    assert post_restart_res["embeddings"] is not None
    assert post_restart_res["embeddings"][0] == pytest.approx(dummy_embedding)

    # Verify metadata
    assert post_restart_res["metadatas"] == [meta]


def test_direct_persistent_client_reload(test_db_path: Path) -> None:
    """
    Verify that creating a completely new PersistentClient directly against the same directory
    reloads the persisted data without re-ingestion.
    """
    from src.vectorstore import client, init, upsert

    c1 = client.get_client()
    collection1 = init.get_or_create_collection(c1)

    record_id = "PECS-DIRECT-RELOAD-002"
    doc_str = document_format.serialize_document(
        {"c1": 5, "c2": 5, "c3": 5, "c4": 5, "c5": 5, "c6": 5},
        "Direct client reload test justification."
    )
    meta = metadata_schema.build_metadata(
        domain="EDGE",
        level="weak",
        scores={"c1": 5, "c2": 5, "c3": 5, "c4": 5, "c5": 5, "c6": 5},
        source="validation",
        version="pecs-v1.0"
    )
    dummy_embedding = [0.789] * 384

    upsert.upsert_record(
        collection=collection1,
        record_id=record_id,
        embedding=dummy_embedding,
        document=doc_str,
        metadata=meta
    )

    # Create a fresh PersistentClient directly pointing to test_db_path
    c_fresh = chromadb.PersistentClient(path=str(test_db_path))
    assert c_fresh is not c1

    # Fetch collection from fresh client
    collection_fresh = c_fresh.get_collection(name=init.COLLECTION_NAME)

    # Retrieve record without any re-ingestion call
    res = collection_fresh.get(ids=[record_id], include=["embeddings", "documents", "metadatas"])  # type: ignore
    assert res["ids"] == [record_id]
    assert res["documents"] == [doc_str]
    assert res["embeddings"] is not None
    assert res["embeddings"][0] == pytest.approx(dummy_embedding)
    assert res["metadatas"] == [meta]
