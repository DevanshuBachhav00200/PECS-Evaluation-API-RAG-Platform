"""
VDB-006 — Persistent ChromaDB Client Initialization.

Initializes exactly one persistent ChromaDB client at module load time
reusing the path configured in VDB-001.
"""

from __future__ import annotations

import chromadb
from src.vectorstore.config import get_chroma_db_path

# Resolve the absolute ChromaDB storage directory
CHROMA_DB_PATH = get_chroma_db_path()

# Initialize the persistent client exactly once at module load
_client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))


def get_client() -> chromadb.PersistentClient:
    """
    Return the module-level warm persistent ChromaDB client.

    This ensures that get_client() returns the same persistent client instance
    on repeated calls and does not initialize a new client inside every function call.

    Returns:
        chromadb.PersistentClient: The shared persistent client instance.
    """
    return _client
