"""
Unit tests for VDB-006 persistent ChromaDB client initialization.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
import pytest
import chromadb

from src.vectorstore import init


@pytest.fixture(autouse=True)
def isolated_client_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Overriding environment variables and evicting cached modules to isolate
    the persistent client to a temporary directory in every test.
    """
    # Create an isolated temporary db path
    test_db_path = tmp_path / "test_pels_vector_db"
    monkeypatch.setenv("CHROMA_DB_PATH", str(test_db_path))

    # Force client.py and config.py to re-initialize and consume the overridden env variable
    for k in list(sys.modules.keys()):
        if k.startswith("src.vectorstore"):
            sys.modules.pop(k, None)

    return test_db_path


def test_get_client_returns_persistent_client(isolated_client_env: Path) -> None:
    """Verify that get_client() returns a chromadb.ClientAPI instance."""
    from src.vectorstore import client

    c = client.get_client()
    assert isinstance(c, chromadb.ClientAPI)


def test_client_singleton_behavior(isolated_client_env: Path) -> None:
    """Verify that repeated calls to get_client() return the same client instance."""
    from src.vectorstore import client

    c1 = client.get_client()
    c2 = client.get_client()
    assert c1 is c2  # Singleton/module-level caching check


def test_client_respects_configured_path(isolated_client_env: Path) -> None:
    """Verify that the PersistentClient uses the path from get_chroma_db_path()."""
    from src.vectorstore import client, config

    c = client.get_client()
    # Confirm it matches the get_chroma_db_path() configuration
    assert Path(c.get_settings().persist_directory).resolve() == config.get_chroma_db_path()
    assert config.get_chroma_db_path() == isolated_client_env.resolve()


def test_client_access_collection_via_init(isolated_client_env: Path) -> None:
    """Verify that the client can create and access the collection through the VDB-002 routine."""
    from src.vectorstore import client

    c = client.get_client()

    # Pre-condition: clean test db
    assert len(c.list_collections()) == 0

    # Call collection initialization
    collection = init.get_or_create_collection(c)
    assert collection.name == "prompt_examples"
    assert len(c.list_collections()) == 1


def test_process_restart_persistence(isolated_client_env: Path) -> None:
    """
    Test reconnecting and retrieving collection data after simulated process restart.
    - Saves a record on the module-level client.
    - Evicts the client module from cache (sys.modules) to simulate process shutdown.
    - Re-imports/re-initializes the client module.
    - Connects to the same path and retrieves the pre-existing collection and record.
    """
    # 1. First "session"
    from src.vectorstore import client as client_v1

    c1 = client_v1.get_client()
    collection1 = init.get_or_create_collection(c1)

    # Add a calibration vector
    collection1.add(
        ids=["restart-test-id"],
        embeddings=[[0.5] * 384],
        documents=["restart test doc"],
        metadatas=[{
            "domain": "CODE",
            "level": "strong",
            "c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5,
            "source": "train",
            "version": "pecs-v1.0"
        }]
    )

    # 2. Simulate process exit by evicting client from cache
    for k in list(sys.modules.keys()):
        if k.startswith("src.vectorstore"):
            sys.modules.pop(k, None)

    # 3. Second "session" after restart
    from src.vectorstore import client as client_v2

    c2 = client_v2.get_client()

    # Assert that c1 and c2 are different objects (module reload worked)
    assert c1 is not c2

    # Access the same collection
    collection2 = c2.get_collection(name=init.COLLECTION_NAME)

    # Retrieve the previously added record
    results = collection2.get(ids=["restart-test-id"])
    assert results["ids"] == ["restart-test-id"]
    assert results["documents"] == ["restart test doc"]


def test_no_production_database_modified(isolated_client_env: Path) -> None:
    """Verify that the test client path is not the production database path."""
    from src.vectorstore import client
    c = client.get_client()
    prod_path = (Path(__file__).resolve().parents[2] / "Database" / "pels_vector_db").resolve()
    assert Path(c.get_settings().persist_directory).resolve() != prod_path
