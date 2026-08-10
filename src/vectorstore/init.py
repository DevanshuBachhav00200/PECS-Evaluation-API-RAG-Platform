"""
VDB-002 — Define collection name & initialization routine.

Exposes the default collection name for prompt examples and provides
an idempotent routine to get or create the collection.
"""

from __future__ import annotations

from typing import Any
import chromadb

# The default collection name for PECS prompt examples
COLLECTION_NAME = "prompt_examples"


def get_or_create_collection(client: Any) -> Any:
    """
    Get or create the default ChromaDB collection for prompt examples.

    This routine is idempotent:
    - First call creates the collection.
    - Subsequent calls return the same existing collection.
    - No duplicate collections are created.
    - No exception is raised simply because the collection already exists.

    This function does not instantiate any persistent client internally, adhering
    to the architectural boundary where VDB-006 owns client creation.

    Args:
        client: An initialized ChromaDB client instance (e.g. EphemeralClient or PersistentClient).

    Returns:
        chromadb.Collection: The resolved default collection.
    """
    return client.get_or_create_collection(name=COLLECTION_NAME)
