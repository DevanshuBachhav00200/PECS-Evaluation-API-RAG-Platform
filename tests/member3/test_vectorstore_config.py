"""
Unit tests for VDB-001 vector-store configuration path resolution.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

from src.vectorstore import config


def test_get_chroma_db_path_returns_path_object() -> None:
    """Verify that get_chroma_db_path() returns a pathlib.Path instance."""
    path = config.get_chroma_db_path()
    assert isinstance(path, Path)


def test_default_path_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """When CHROMA_DB_PATH is unset, it should default to `<repo_root>/Database/pels_vector_db`."""
    monkeypatch.delenv("CHROMA_DB_PATH", raising=False)
    path = config.get_chroma_db_path()
    expected = (config.REPO_ROOT / "Database" / "pels_vector_db").resolve()
    assert path == expected


def test_relative_path_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """When CHROMA_DB_PATH is a relative path, it must resolve relative to the repository root."""
    monkeypatch.setenv("CHROMA_DB_PATH", "custom/vector_db")
    path = config.get_chroma_db_path()
    expected = (config.REPO_ROOT / "custom" / "vector_db").resolve()
    assert path == expected


def test_absolute_path_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """When CHROMA_DB_PATH is an absolute path, the absolute location should be preserved."""
    abs_path_str = str(Path("/my_absolute_vector_db_path").resolve())
    monkeypatch.setenv("CHROMA_DB_PATH", abs_path_str)
    path = config.get_chroma_db_path()
    assert str(path) == abs_path_str


def test_empty_string_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """When CHROMA_DB_PATH is an empty string, it should fall back to the default path."""
    monkeypatch.setenv("CHROMA_DB_PATH", "  ")
    path = config.get_chroma_db_path()
    expected = (config.REPO_ROOT / "Database" / "pels_vector_db").resolve()
    assert path == expected


def test_config_does_not_initialize_chromadb() -> None:
    """
    Verify that ChromaDB is not imported or initialized, and no directories
    are created during configuration resolution.
    """
    # 1. Verify ChromaDB is not imported by config.py using AST parsing
    import ast
    config_file = Path(config.__file__)
    tree = ast.parse(config_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                assert "chromadb" not in name.name, "config.py must not import chromadb"
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            assert "chromadb" not in node.module, "config.py must not import chromadb"

    # 2. Verify that get_chroma_db_path() creates no directory.
    #
    # Scoped to "resolution creates nothing", not "no Database/ exists anywhere":
    # a populated store on disk is the normal state of a working checkout (it is
    # what `python -m scripts.build_vector_store` produces, and what the
    # container mounts), so the absolute form made this test fail for anyone who
    # had actually built the store. Comparing before/after is the property
    # VDB-001 promises. Amended by Member 6 (TEST-003).
    db_dir = config.REPO_ROOT / "Database"
    existed_before = db_dir.exists()

    resolved = config.get_chroma_db_path()

    assert resolved.name == "pels_vector_db" or resolved.is_absolute()
    assert db_dir.exists() == existed_before, (
        "Database directory must not be created during path resolution"
    )
    assert not (db_dir / "created_by_config_resolution").exists()
