"""
Unit tests for VDB-002 vector-store collection initialization and naming.
"""

from __future__ import annotations

from typing import Any
import sys
import pytest
import chromadb

from src.vectorstore import init


def cleanup_collection(client: Any) -> None:
    """Helper to delete the prompt_examples collection if it already exists."""
    try:
        client.delete_collection(name="prompt_examples")
    except Exception:
        pass


def test_collection_name_constant() -> None:
    """Verify that COLLECTION_NAME is exactly 'prompt_examples'."""
    assert init.COLLECTION_NAME == "prompt_examples"


def test_collection_creation_first_run() -> None:
    """Verify that the collection is created on the first initialization call on a clean client."""
    client = chromadb.EphemeralClient()
    cleanup_collection(client)

    # Pre-condition: no collections exist
    assert len(client.list_collections()) == 0

    collection = init.get_or_create_collection(client)
    assert collection.name == "prompt_examples"

    # Post-condition: exactly one collection exists and matches
    collections = client.list_collections()
    assert len(collections) == 1
    assert collections[0].name == "prompt_examples"


def test_collection_initialization_idempotence() -> None:
    """Verify that calling the initialization twice does not error, duplicate, or fail."""
    client = chromadb.EphemeralClient()
    cleanup_collection(client)

    col1 = init.get_or_create_collection(client)
    col2 = init.get_or_create_collection(client)

    # Both calls must refer to the exact same collection
    assert col1.id == col2.id
    assert len(client.list_collections()) == 1


def test_existing_collection_returned_correctly() -> None:
    """Verify that if the collection was manually pre-created, it is returned correctly."""
    client = chromadb.EphemeralClient()
    cleanup_collection(client)

    # Pre-create the collection manually
    col_manual = client.create_collection(name="prompt_examples")

    # Get it through the init routine
    col_retrieved = init.get_or_create_collection(client)

    assert col_retrieved.id == col_manual.id
    assert len(client.list_collections()) == 1


def test_does_not_instantiate_chromadb_client_internally(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that the initialization logic does not instantiate a Client itself."""
    instantiated = False

    # Store original constructors to avoid infinite recursion
    orig_ephemeral = chromadb.EphemeralClient
    orig_persistent = chromadb.PersistentClient

    def mock_client_init(*args: Any, **kwargs: Any) -> chromadb.EphemeralClient:
        nonlocal instantiated
        instantiated = True
        return orig_ephemeral()

    monkeypatch.setattr(chromadb, "PersistentClient", mock_client_init)
    monkeypatch.setattr(chromadb, "EphemeralClient", mock_client_init)

    # Manually instantiate for the test using original class
    client = orig_ephemeral()
    instantiated = False  # Reset flag

    # Run get_or_create_collection
    init.get_or_create_collection(client)

    # Verify that get_or_create_collection did not trigger any client instantiation
    assert not instantiated, "get_or_create_collection must not instantiate a client internally"
