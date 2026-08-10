"""
Unit tests for VDB-004 metadata schema creation and validation.
"""

from __future__ import annotations

import sys
import pytest

from src.vectorstore import metadata_schema


@pytest.fixture
def valid_scores() -> dict[str, float | int]:
    return {"c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5}


@pytest.fixture
def valid_scores_score_keys() -> dict[str, float | int]:
    return {
        "c1_score": 8,
        "c2_score": 7,
        "c3_score": 9,
        "c4_score": 6,
        "c5_score": 10,
        "c6_score": 5,
    }


def test_valid_metadata_build_and_validation(valid_scores: dict[str, float | int]) -> None:
    """Verify that building metadata with valid inputs constructs a dictionary that passes validation."""
    meta = metadata_schema.build_metadata(
        domain="CODE",
        level="developing",
        scores=valid_scores,
        source="train",
        version="pecs-v1.0",
    )

    # Validate that validate_metadata runs without raising errors
    metadata_schema.validate_metadata(meta)

    # Verify all expected values are present and preserved
    assert meta["domain"] == "CODE"
    assert meta["level"] == "developing"
    assert meta["source"] == "train"
    assert meta["version"] == "pecs-v1.0"
    for i in range(1, 7):
        assert meta[f"c{i}"] == valid_scores[f"c{i}"]
        assert type(meta[f"c{i}"]) is type(valid_scores[f"c{i}"])


def test_alternative_score_keys_mapping(valid_scores_score_keys: dict[str, float | int]) -> None:
    """Verify that c1_score..c6_score keys are mapped correctly to c1..c6 in the built metadata."""
    meta = metadata_schema.build_metadata(
        domain="EDGE",
        level="weak",
        scores=valid_scores_score_keys,
        source="validation",
        version="pecs-v1.0",
    )

    metadata_schema.validate_metadata(meta)

    # Verify keys are converted to c1..c6
    for i in range(1, 7):
        assert f"c{i}" in meta
        assert f"c{i}_score" not in meta
        assert meta[f"c{i}"] == valid_scores_score_keys[f"c{i}_score"]


def test_exact_required_keys(valid_scores: dict[str, float | int]) -> None:
    """Verify that built metadata contains exactly the 10 required keys (no more, no less)."""
    meta = metadata_schema.build_metadata(
        domain="CODE",
        level="strong",
        scores=valid_scores,
        source="test",
        version="pecs-v1.0",
    )

    assert set(meta.keys()) == metadata_schema.REQUIRED_KEYS
    assert len(meta) == 10


def test_chromadb_value_compatibility(valid_scores: dict[str, float | int]) -> None:
    """ChromaDB metadata values must be only str, int, float, or bool. Verify compatibility."""
    meta = metadata_schema.build_metadata(
        domain="CODE",
        level="developing",
        scores=valid_scores,
        source="train",
        version="pecs-v1.0",
    )

    allowed_types = (str, int, float, bool)
    for k, v in meta.items():
        assert isinstance(v, allowed_types), f"Value for key '{k}' must be ChromaDB-compatible type"


def test_invalid_or_missing_domain(valid_scores: dict[str, float | int]) -> None:
    """Invalid or missing domain values must fail validation."""
    # Build a metadata dict directly to test validation boundaries
    meta_valid = metadata_schema.build_metadata(
        domain="CODE", level="developing", scores=valid_scores, source="train", version="pecs-v1.0"
    )

    # Bad domain string
    meta_bad_str = meta_valid.copy()
    meta_bad_str["domain"] = "INVALID"
    with pytest.raises(ValueError):
        metadata_schema.validate_metadata(meta_bad_str)

    # Bad domain type
    meta_bad_type = meta_valid.copy()
    meta_bad_type["domain"] = 123
    with pytest.raises(TypeError):
        metadata_schema.validate_metadata(meta_bad_type)


def test_invalid_or_missing_level(valid_scores: dict[str, float | int]) -> None:
    """Invalid or missing level values must fail validation."""
    meta_valid = metadata_schema.build_metadata(
        domain="EDGE", level="developing", scores=valid_scores, source="train", version="pecs-v1.0"
    )

    # Bad level string
    meta_bad_str = meta_valid.copy()
    meta_bad_str["level"] = "advanced"
    with pytest.raises(ValueError):
        metadata_schema.validate_metadata(meta_bad_str)

    # Bad level type
    meta_bad_type = meta_valid.copy()
    meta_bad_type["level"] = None
    with pytest.raises(TypeError):
        metadata_schema.validate_metadata(meta_bad_type)


def test_missing_or_out_of_range_scores(valid_scores: dict[str, float | int]) -> None:
    """Missing score fields or scores outside the 1..10 range must fail validation."""
    meta_valid = metadata_schema.build_metadata(
        domain="CODE", level="developing", scores=valid_scores, source="train", version="pecs-v1.0"
    )

    # Score below 1
    meta_low = meta_valid.copy()
    meta_low["c3"] = 0.9
    with pytest.raises(ValueError):
        metadata_schema.validate_metadata(meta_low)

    # Score above 10
    meta_high = meta_valid.copy()
    meta_high["c4"] = 10.1
    with pytest.raises(ValueError):
        metadata_schema.validate_metadata(meta_high)

    # Missing score key
    meta_missing = meta_valid.copy()
    del meta_missing["c6"]
    with pytest.raises(ValueError):
        metadata_schema.validate_metadata(meta_missing)

    # Non-numeric score type
    meta_type = meta_valid.copy()
    meta_type["c1"] = "8"
    with pytest.raises(TypeError):
        metadata_schema.validate_metadata(meta_type)


def test_missing_source_or_version(valid_scores: dict[str, float | int]) -> None:
    """Missing or empty/whitespace source/version strings must fail validation."""
    meta_valid = metadata_schema.build_metadata(
        domain="CODE", level="developing", scores=valid_scores, source="train", version="pecs-v1.0"
    )

    # Empty source
    meta_empty_src = meta_valid.copy()
    meta_empty_src["source"] = ""
    with pytest.raises(ValueError):
        metadata_schema.validate_metadata(meta_empty_src)

    # Whitespace version
    meta_ws_ver = meta_valid.copy()
    meta_ws_ver["version"] = "   "
    with pytest.raises(ValueError):
        metadata_schema.validate_metadata(meta_ws_ver)


def test_extra_metadata_keys_rejected(valid_scores: dict[str, float | int]) -> None:
    """Unexpected/extra metadata keys must be rejected during validation."""
    meta_valid = metadata_schema.build_metadata(
        domain="CODE", level="developing", scores=valid_scores, source="train", version="pecs-v1.0"
    )

    # Add extra key (e.g. prompt_id or justification)
    meta_extra = meta_valid.copy()
    meta_extra["prompt_id"] = "PELS-001"
    with pytest.raises(ValueError):
        metadata_schema.validate_metadata(meta_extra)


def test_no_chromadb_instantiated() -> None:
    """Confirm that no ChromaDB dependencies are imported by metadata_schema.py."""
    import ast
    from pathlib import Path
    config_file = Path(metadata_schema.__file__)
    tree = ast.parse(config_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                assert "chromadb" not in name.name, "metadata_schema.py must not import chromadb"
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            assert "chromadb" not in node.module, "metadata_schema.py must not import chromadb"
