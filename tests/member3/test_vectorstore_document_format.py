"""
Unit tests for VDB-003 vector-store document representation formatting.
"""

from __future__ import annotations

import json
import sys
import pytest

from src.vectorstore import document_format


@pytest.fixture
def valid_scores_c1_c6() -> dict[str, float | int]:
    return {"c1": 8, "c2": 7.5, "c3": 9, "c4": 6, "c5": 10, "c6": 5}


@pytest.fixture
def valid_scores_c1_c6_score() -> dict[str, float | int]:
    return {
        "c1_score": 8,
        "c2_score": 7.5,
        "c3_score": 9.0,
        "c4_score": 6,
        "c5_score": 10,
        "c6_score": 5,
    }


@pytest.fixture
def valid_justification() -> str:
    return "This is a detailed justification demonstrating prompt quality."


def test_serialization_produces_valid_json(
    valid_scores_c1_c6: dict[str, float | int], valid_justification: str
) -> None:
    """Verify that serialization produces a valid JSON string."""
    doc_str = document_format.serialize_document(valid_scores_c1_c6, valid_justification)
    assert isinstance(doc_str, str)

    # Must parse successfully back to python dict using standard json library
    parsed = json.loads(doc_str)
    assert isinstance(parsed, dict)
    assert "scores" in parsed
    assert "justification" in parsed


def test_round_trip_reconstruction_c1_c6(
    valid_scores_c1_c6: dict[str, float | int], valid_justification: str
) -> None:
    """Verify that serialization followed by deserialization preserves the exact structure."""
    doc_str = document_format.serialize_document(valid_scores_c1_c6, valid_justification)
    reconstructed = document_format.deserialize_document(doc_str)

    assert reconstructed["justification"] == valid_justification

    # Numeric types and values are preserved exactly
    for k in valid_scores_c1_c6:
        original_val = valid_scores_c1_c6[k]
        recon_val = reconstructed["scores"][k]
        assert original_val == recon_val
        assert type(original_val) is type(recon_val)


def test_round_trip_reconstruction_c1_c6_score(
    valid_scores_c1_c6_score: dict[str, float | int], valid_justification: str
) -> None:
    """Verify that the c1_score..c6_score key convention is also supported and preserved."""
    doc_str = document_format.serialize_document(valid_scores_c1_c6_score, valid_justification)
    reconstructed = document_format.deserialize_document(doc_str)

    assert reconstructed["justification"] == valid_justification

    # Numeric values and key conventions are preserved
    for k in valid_scores_c1_c6_score:
        original_val = valid_scores_c1_c6_score[k]
        recon_val = reconstructed["scores"][k]
        assert original_val == recon_val
        assert type(original_val) is type(recon_val)


def test_missing_or_empty_justification(valid_scores_c1_c6: dict[str, float | int]) -> None:
    """Serialization must reject empty or whitespace-only justifications."""
    # Non-string justifications must raise TypeError
    with pytest.raises(TypeError):
        document_format.serialize_document(valid_scores_c1_c6, None)  # type: ignore

    with pytest.raises(TypeError):
        document_format.serialize_document(valid_scores_c1_c6, 123)  # type: ignore

    # Empty/whitespace justifications must raise ValueError
    with pytest.raises(ValueError):
        document_format.serialize_document(valid_scores_c1_c6, "")

    with pytest.raises(ValueError):
        document_format.serialize_document(valid_scores_c1_c6, "   ")


def test_incomplete_or_invalid_scores(valid_justification: str) -> None:
    """Serialization must reject scores dictionaries missing any C1-C6/C1_score-C6_score keys."""
    # Not a dictionary
    with pytest.raises(TypeError):
        document_format.serialize_document([8, 7, 9, 6, 10, 5], valid_justification)  # type: ignore

    # Missing c6 key
    incomplete_scores = {"c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10}
    with pytest.raises(ValueError):
        document_format.serialize_document(incomplete_scores, valid_justification)


def test_out_of_range_scores(
    valid_scores_c1_c6: dict[str, float | int], valid_justification: str
) -> None:
    """Scores outside the 1..10 range must raise a ValueError."""
    # Score too low
    bad_scores_low = valid_scores_c1_c6.copy()
    bad_scores_low["c3"] = 0.5
    with pytest.raises(ValueError):
        document_format.serialize_document(bad_scores_low, valid_justification)

    # Score too high
    bad_scores_high = valid_scores_c1_c6.copy()
    bad_scores_high["c4"] = 11
    with pytest.raises(ValueError):
        document_format.serialize_document(bad_scores_high, valid_justification)


def test_invalid_score_types(
    valid_scores_c1_c6: dict[str, float | int], valid_justification: str
) -> None:
    """Non-numeric score types (including boolean which is a subclass of int) must raise TypeError."""
    bad_scores_type = valid_scores_c1_c6.copy()
    bad_scores_type["c1"] = "8"
    with pytest.raises(TypeError):
        document_format.serialize_document(bad_scores_type, valid_justification)

    bad_scores_bool = valid_scores_c1_c6.copy()
    bad_scores_bool["c2"] = True
    with pytest.raises(TypeError):
        document_format.serialize_document(bad_scores_bool, valid_justification)


def test_malformed_json_deserialization() -> None:
    """Deserialization of malformed JSON strings must raise a ValueError."""
    with pytest.raises(TypeError):
        document_format.deserialize_document(123)  # type: ignore

    with pytest.raises(ValueError):
        document_format.deserialize_document("Not a JSON string")

    with pytest.raises(ValueError):
        # JSON is valid syntactically but missing keys
        document_format.deserialize_document('{"justification": "text"}')


def test_serialization_determinism(
    valid_scores_c1_c6: dict[str, float | int], valid_justification: str
) -> None:
    """Verify that repeated serialization yields the exact same string (deterministic sorting)."""
    # Insert keys in different dictionary insertion order
    scores_order_a = {"c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5}
    scores_order_b = {"c6": 5, "c5": 10, "c4": 6, "c3": 9, "c2": 7, "c1": 8}

    doc_a = document_format.serialize_document(scores_order_a, valid_justification)
    doc_b = document_format.serialize_document(scores_order_b, valid_justification)

    assert doc_a == doc_b


def test_no_chromadb_instantiated() -> None:
    """Confirm that no ChromaDB dependencies are imported by document_format.py."""
    import ast
    from pathlib import Path
    config_file = Path(document_format.__file__)
    tree = ast.parse(config_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                assert "chromadb" not in name.name, "document_format.py must not import chromadb"
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            assert "chromadb" not in node.module, "document_format.py must not import chromadb"
