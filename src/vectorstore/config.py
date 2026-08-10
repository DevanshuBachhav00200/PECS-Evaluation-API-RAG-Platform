"""
VDB-001 — Define vector-store directory structure.

Defines the on-disk vector-store layout and CHROMA_DB_PATH convention.
Provides path resolution relative to the repository root.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve repository root relative to this file
# src/vectorstore/config.py -> parents[2] == repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from .env file if it exists at the repo root
load_dotenv(REPO_ROOT / ".env")


def get_chroma_db_path() -> Path:
    """
    Resolve and return the configured path to the ChromaDB vector database.

    Resolution rules:
    - If CHROMA_DB_PATH environment variable is unset or empty:
        Returns `<repo_root>/Database/pels_vector_db`
    - If CHROMA_DB_PATH is a relative path:
        Resolves it relative to the repository root.
    - If CHROMA_DB_PATH is an absolute path:
        Preserves and returns it as the configured absolute location.

    Returns:
        Path: The resolved path to the ChromaDB persist directory.
    """
    db_path_str = os.getenv("CHROMA_DB_PATH")

    if not db_path_str or not db_path_str.strip():
        # Fallback to default path: Database/pels_vector_db relative to repo root
        return (REPO_ROOT / "Database" / "pels_vector_db").resolve()

    path = Path(db_path_str)

    if path.is_absolute():
        # Preserve absolute path
        return path.resolve()
    else:
        # Resolve relative to repo root
        return (REPO_ROOT / path).resolve()
