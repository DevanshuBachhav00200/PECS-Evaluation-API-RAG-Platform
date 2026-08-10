"""
VDB-004 — Define metadata schema.

Provides construction and validation routines for ChromaDB metadata,
ensuring alignment with the canonical DS-002 requirements.
"""

from __future__ import annotations

from typing import Any, Dict, Set

# The exact 10 required metadata keys
REQUIRED_KEYS: Set[str] = {
    "domain",
    "level",
    "c1",
    "c2",
    "c3",
    "c4",
    "c5",
    "c6",
    "source",
    "version",
}

# Canonical domains and skill levels from the corpus
VALID_DOMAINS: Set[str] = {"CODE", "EDGE"}
VALID_LEVELS: Set[str] = {"weak", "developing", "strong"}


def validate_metadata(metadata: Dict[str, Any]) -> None:
    """
    Validate that the metadata dictionary conforms exactly to the REQUIRED_KEYS schema
    and contains compatible types and values.

    Args:
        metadata: The metadata dictionary to validate.

    Raises:
        TypeError: If types of dictionary values are incorrect or not ChromaDB compatible.
        ValueError: If required keys are missing, extra keys exist, or values are out of bounds.
    """
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")

    # Verify keys set matches REQUIRED_KEYS exactly
    current_keys = set(metadata.keys())
    missing = REQUIRED_KEYS - current_keys
    extra = current_keys - REQUIRED_KEYS

    if missing:
        raise ValueError(f"Missing required metadata keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected/extra metadata keys: {sorted(extra)}")

    # 1. Validate domain
    domain = metadata["domain"]
    if not isinstance(domain, str):
        raise TypeError(f"domain must be a string, got {type(domain).__name__}")
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Invalid domain '{domain}'; must be one of {sorted(VALID_DOMAINS)}")

    # 2. Validate level
    level = metadata["level"]
    if not isinstance(level, str):
        raise TypeError(f"level must be a string, got {type(level).__name__}")
    if level not in VALID_LEVELS:
        raise ValueError(f"Invalid level '{level}'; must be one of {sorted(VALID_LEVELS)}")

    # 3. Validate C1-C6 scores
    for i in range(1, 7):
        key = f"c{i}"
        val = metadata[key]
        if val is None:
            raise ValueError(f"Score for '{key}' cannot be None")
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise TypeError(f"Score for '{key}' must be a number, got {type(val).__name__}")
        if not (1 <= val <= 10):
            raise ValueError(f"Score for '{key}' must be between 1 and 10 inclusive, got {val}")

    # 4. Validate source
    source = metadata["source"]
    if not isinstance(source, str):
        raise TypeError(f"source must be a string, got {type(source).__name__}")
    if not source.strip():
        raise ValueError("source cannot be empty or whitespace only")

    # 5. Validate version
    version = metadata["version"]
    if not isinstance(version, str):
        raise TypeError(f"version must be a string, got {type(version).__name__}")
    if not version.strip():
        raise ValueError("version cannot be empty or whitespace only")


def build_metadata(
    domain: str,
    level: str,
    scores: Dict[str, Any],
    source: str,
    version: str,
) -> Dict[str, Any]:
    """
    Helper function to build and validate metadata.

    Supports scores dictionaries with keys 'c1'..'c6' or 'c1_score'..'c6_score'.

    Args:
        domain: Canonical domain (e.g. 'CODE' or 'EDGE').
        level: Canonical level (e.g. 'weak', 'developing', 'strong').
        scores: Dict containing calibration scores.
        source: Source dataset split or identifier.
        version: Version string.

    Returns:
        dict: A validated metadata dictionary containing exactly the required keys.
    """
    if not isinstance(scores, dict):
        raise TypeError("scores must be a dictionary")

    # Extract score values (c1..c6 or c1_score..c6_score)
    expected_c1_c6 = {f"c{i}" for i in range(1, 7)}
    expected_c1_c6_score = {f"c{i}_score" for i in range(1, 7)}

    keys = set(scores.keys())
    score_mapping = {}

    if expected_c1_c6.issubset(keys):
        score_mapping = {k: scores[k] for k in expected_c1_c6}
    elif expected_c1_c6_score.issubset(keys):
        score_mapping = {f"c{i}": scores[f"c{i}_score"] for i in range(1, 7)}
    else:
        raise ValueError(
            "scores dictionary must contain all 6 calibration keys (c1..c6 or c1_score..c6_score)"
        )

    metadata = {
        "domain": domain,
        "level": level,
        "source": source,
        "version": version,
    }
    # Merge extracted score mappings
    metadata.update(score_mapping)

    # Perform validation before returning
    validate_metadata(metadata)

    return metadata
