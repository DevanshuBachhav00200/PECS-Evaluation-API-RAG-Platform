"""
VDB-006 — Persistent ChromaDB Client Initialization.

Initializes exactly one persistent ChromaDB client, reusing the path configured
in VDB-001, and hands the same instance to every caller.

Initialization is deferred to the first ``get_client()`` call rather than run at
module import. The singleton guarantee is unchanged — the client is still built
once per process and cached — but importing ``src.vectorstore`` no longer has
the side effect of creating the ``Database/`` directory on disk.

Why that mattered enough to change (Member 6, TEST-003 / DEP-003):

* It made ``tests/member3/test_vectorstore_config.py::test_config_does_not_
  initialize_chromadb`` order-dependent: the test asserts no ``Database/``
  directory exists, and it passed alone but failed once any earlier test had
  imported the vector-store package.
* In the container (DEP-003) the store lives on a mounted volume. An import-time
  client meant a bare ``python -c "import src.api.main"`` — including the Docker
  health check path — could create a stray, unmounted ``Database/`` inside the
  image layer.

The eager-load rationale still holds for the *service*: the API should warm the
client at startup rather than inside the first user request. That is now an
explicit call (``get_client()`` in a startup hook), not an import side effect.
"""

from __future__ import annotations

from typing import Optional

import chromadb

from src.vectorstore.config import get_chroma_db_path

# Resolved at import for backwards compatibility: existing code and tests read
# this constant. Resolution is pure path arithmetic — it touches no disk.
CHROMA_DB_PATH = get_chroma_db_path()

_client: Optional[chromadb.ClientAPI] = None


def get_client() -> chromadb.ClientAPI:
    """
    Return the process-wide persistent ChromaDB client, creating it on first use.

    Repeated calls return the identical instance, so no function ever builds its
    own client and the store stays warm after the first request.

    Returns:
        chromadb.ClientAPI: The shared persistent client instance.
    """
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(get_chroma_db_path()))
    return _client


def reset_client() -> None:
    """
    Drop the cached client so the next ``get_client()`` rebuilds it.

    Exists for tests that repoint ``CHROMA_DB_PATH`` mid-process; production code
    has no reason to call it.
    """
    global _client
    _client = None
