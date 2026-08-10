"""
Unit tests for EMB-010 semantic near-duplicate audit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from scripts import run_semantic_dedupe


# ---------------------------------------------------------------------------
# Core math tests
# ---------------------------------------------------------------------------

def test_cosine_similarity_identical() -> None:
    """Identical vectors must produce a cosine similarity of approximately 1.0."""
    v1 = np.random.rand(384)
    v2 = v1.copy()

    # Calculate cosine similarity using normalized dot product
    norm_v1 = v1 / np.linalg.norm(v1)
    norm_v2 = v2 / np.linalg.norm(v2)
    sim = float(np.dot(norm_v1, norm_v2))

    assert pytest.approx(sim, abs=1e-5) == 1.0


def test_cosine_similarity_dissimilar() -> None:
    """Clearly different vectors should produce similarity below 0.92."""
    v1 = np.zeros(384)
    v1[0] = 1.0

    v2 = np.zeros(384)
    v2[1] = 1.0  # Orthogonal vector

    norm_v1 = v1 / np.linalg.norm(v1)
    norm_v2 = v2 / np.linalg.norm(v2)
    sim = float(np.dot(norm_v1, norm_v2))

    assert sim == 0.0
    assert sim < 0.92


# ---------------------------------------------------------------------------
# Mocks & Integration tests for run_audit
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_audit_env(monkeypatch: pytest.MonkeyPatch) -> dict:
    """
    Setup a mock audit environment where dataset reading, report writing,
    and embedding loading/saving are handled in memory.
    """
    # In-memory storage for test verification
    in_memory_files = {}

    def mock_read_csv(filepath: Any, **kwargs: Any) -> pd.DataFrame:
        if "corpus" in in_memory_files:
            return in_memory_files["corpus"]
        raise FileNotFoundError(f"Mock file not found: {filepath}")

    def mock_load_serialized(filepath: Any) -> Dict[str, Any]:
        if "embeddings" in in_memory_files:
            return in_memory_files["embeddings"]
        raise FileNotFoundError(f"Mock file not found: {filepath}")

    def mock_save_serialized(data: Dict[str, Any], filepath: Any) -> None:
        in_memory_files["embeddings_saved"] = data

    def mock_write_report(
        total_records: int,
        lexical_pairs_count: int,
        semantic_pairs_count: int,
        semantic_only_count: int,
        lexical_only_count: int,
        findings: list[dict[str, Any]],
    ) -> None:
        in_memory_files["report_data"] = {
            "total_records": total_records,
            "lexical_pairs_count": lexical_pairs_count,
            "semantic_pairs_count": semantic_pairs_count,
            "semantic_only_count": semantic_only_count,
            "lexical_only_count": lexical_only_count,
            "findings": findings,
        }

    # Apply patches
    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    monkeypatch.setattr(run_semantic_dedupe, "load_serialized", mock_load_serialized)
    monkeypatch.setattr(run_semantic_dedupe, "save_serialized", mock_save_serialized)
    monkeypatch.setattr(run_semantic_dedupe, "write_report", mock_write_report)

    # Force paths to exist
    monkeypatch.setattr(Path, "exists", lambda self: True)

    return in_memory_files


def test_run_audit_empty_corpus(mock_audit_env: dict) -> None:
    """An empty corpus must be handled safely and return zero pairs."""
    mock_audit_env["corpus"] = pd.DataFrame(columns=["record_id", "prompt_text"])

    results = run_semantic_dedupe.run_audit()

    assert results["total_records"] == 0
    assert results["lexical_pairs_count"] == 0
    assert results["semantic_pairs_count"] == 0
    assert len(results["findings"]) == 0


def test_run_audit_single_record_corpus(mock_audit_env: dict) -> None:
    """A single record corpus must produce zero pairs and run safely."""
    mock_audit_env["corpus"] = pd.DataFrame([{
        "record_id": "REC001",
        "prompt_id": "PELS-001",
        "prompt_text": "Only one prompt in the corpus.",
        "domain": "CODE",
        "skill_level": "developing",
    }])

    # Mock corresponding embeddings (N=1)
    mock_audit_env["embeddings"] = {
        "ids": ["REC001"],
        "embeddings": [np.ones(384).tolist()],
        "metadatas": [{"prompt_id": "PELS-001", "domain": "CODE", "skill_level": "developing"}],
        "documents": ["Only one prompt in the corpus."],
    }

    results = run_semantic_dedupe.run_audit()

    assert results["total_records"] == 1
    assert results["lexical_pairs_count"] == 0
    assert results["semantic_pairs_count"] == 0
    assert len(results["findings"]) == 0


def test_run_audit_categorization_and_metadata_preservation(mock_audit_env: dict) -> None:
    """
    Test that duplicate pairs are classified correctly and metadata is preserved.

    Pairs to seed:
    - REC001 & REC002: Identical prompts (Lexical + Semantic)
      Prompt 1: "Write a function to add two numbers."
      Prompt 2: "Write a function to add two numbers."
      Semantic similarity: 1.0 (>= 0.92)
      Lexical similarity: 1.0 (>= 0.90)
      Expected category: "lexical + semantic"

    - REC001 & REC003: Semantically similar but lexically different (Semantic-Only)
      Prompt 1: "Write a function to add two numbers."
      Prompt 3: "Create a Python subroutine to compute the summation of two integers."
      Semantic similarity: 0.95 (>= 0.92)
      Lexical similarity: 0.35 (< 0.90)
      Expected category: "semantic-only"

    - REC001 & REC004: Distinct prompts (No match)
      Prompt 4: "Suggest ways to forecast energy consumption."
      Semantic similarity: 0.10 (< 0.92)
      Lexical similarity: 0.10 (< 0.90)
      Expected category: Excluded
    """
    mock_audit_env["corpus"] = pd.DataFrame([
        {"record_id": "REC001", "prompt_id": "PELS-001", "prompt_text": "Write a function to add two numbers.", "domain": "CODE", "skill_level": "weak"},
        {"record_id": "REC002", "prompt_id": "PELS-002", "prompt_text": "Write a function to add two numbers.", "domain": "CODE", "skill_level": "weak"},
        {"record_id": "REC003", "prompt_id": "PELS-003", "prompt_text": "Create a Python subroutine to compute the summation of two integers.", "domain": "CODE", "skill_level": "developing"},
        {"record_id": "REC004", "prompt_id": "PELS-004", "prompt_text": "Suggest ways to forecast energy consumption.", "domain": "CODE", "skill_level": "developing"},
    ])

    # Seed orthogonal base vectors
    base1 = np.zeros(384)
    base1[0] = 1.0
    base2 = np.zeros(384)
    base2[1] = 1.0

    # Build vectors matching the test description:
    # v1 & v2: identical (cosine = 1.0)
    v1 = base1.copy()
    v2 = base1.copy()
    # v1 & v3: highly similar (cosine = 0.95)
    # v3 = 0.95 * v1 + sqrt(1 - 0.95^2) * base2
    v3 = 0.95 * base1 + np.sqrt(1 - 0.95**2) * base2
    # v1 & v4: orthogonal/dissimilar (cosine = 0.0)
    v4 = base2.copy()

    mock_audit_env["embeddings"] = {
        "ids": ["REC001", "REC002", "REC003", "REC004"],
        "embeddings": [v1.tolist(), v2.tolist(), v3.tolist(), v4.tolist()],
        "metadatas": [
            {"prompt_id": "PELS-001", "domain": "CODE", "skill_level": "weak"},
            {"prompt_id": "PELS-002", "domain": "CODE", "skill_level": "weak"},
            {"prompt_id": "PELS-003", "domain": "CODE", "skill_level": "developing"},
            {"prompt_id": "PELS-004", "domain": "CODE", "skill_level": "developing"},
        ],
        "documents": [
            "Write a function to add two numbers.",
            "Write a function to add two numbers.",
            "Create a Python subroutine to compute the summation of two integers.",
            "Suggest ways to forecast energy consumption.",
        ],
    }

    results = run_semantic_dedupe.run_audit()

    assert results["total_records"] == 4
    assert results["semantic_pairs_count"] == 3
    assert results["semantic_only_count"] == 2  # (REC001, REC003) and (REC002, REC003)

    findings = results["findings"]
    # Should contain exactly 3 findings: REC001-REC002, REC001-REC003, and REC002-REC003
    assert len(findings) == 3

    # Check finding 1: Lexical + Semantic overlap
    f_overlap = next(f for f in findings if f["id_2"] == "REC002")
    assert f_overlap["id_1"] == "REC001"
    assert f_overlap["category"] == "lexical + semantic"
    assert pytest.approx(f_overlap["cosine_similarity"]) == 1.0
    assert pytest.approx(f_overlap["lexical_similarity"]) == 1.0
    assert f_overlap["domain_1"] == "CODE"
    assert f_overlap["level_1"] == "weak"

    # Check finding 2: Semantic-only (Missed by DS-007 lexical pass)
    f_semantic_only = next(f for f in findings if f["id_2"] == "REC003")
    assert f_semantic_only["id_1"] == "REC001"
    assert f_semantic_only["category"] == "semantic-only"
    assert pytest.approx(f_semantic_only["cosine_similarity"]) == 0.95
    assert f_semantic_only["lexical_similarity"] < 0.90
    assert f_semantic_only["domain_2"] == "CODE"
    assert f_semantic_only["level_2"] == "developing"


def test_self_pairs_excluded_and_order_uniqueness(mock_audit_env: dict) -> None:
    """
    Self-pairs must be excluded, and reversed duplicates must not create duplicate findings.
    With N=2 similar prompts, there should be exactly 1 finding, not 2 (REC001-REC002, but no REC002-REC001 or REC001-REC001).
    """
    mock_audit_env["corpus"] = pd.DataFrame([
        {"record_id": "REC001", "prompt_id": "PELS-001", "prompt_text": "Write a function to add two numbers.", "domain": "CODE", "skill_level": "weak"},
        {"record_id": "REC002", "prompt_id": "PELS-002", "prompt_text": "Write a function to add two numbers.", "domain": "CODE", "skill_level": "weak"},
    ])

    mock_audit_env["embeddings"] = {
        "ids": ["REC001", "REC002"],
        "embeddings": [np.ones(384).tolist(), np.ones(384).tolist()],
        "metadatas": [
            {"prompt_id": "PELS-001", "domain": "CODE", "skill_level": "weak"},
            {"prompt_id": "PELS-002", "domain": "CODE", "skill_level": "weak"},
        ],
        "documents": [
            "Write a function to add two numbers.",
            "Write a function to add two numbers.",
        ],
    }

    results = run_semantic_dedupe.run_audit()
    findings = results["findings"]

    assert len(findings) == 1
    assert findings[0]["id_1"] == "REC001"
    assert findings[0]["id_2"] == "REC002"
