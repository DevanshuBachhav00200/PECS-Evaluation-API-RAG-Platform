"""
VDB-003 — Define document representation.

Provides serialization and deserialization helpers for calibration examples
to store scores and justification in ChromaDB's document field as a JSON string.
"""

from __future__ import annotations

import json
from typing import Any, Dict


def serialize_document(scores: Dict[str, Any], justification: str) -> str:
    """
    Serialize a scores dictionary and justification string into a JSON document.

    Validation rules:
    - `justification` must be a non-empty, non-whitespace-only string.
    - `scores` must be a dictionary.
    - `scores` must contain all 6 calibration scores (keys must be either 'c1'..'c6'
      or 'c1_score'..'c6_score').
    - Each score must be a number (int or float) and fall between 1 and 10 (inclusive).

    Args:
        scores: Dict containing C1-C6 scores.
        justification: The string justification for the scores.

    Returns:
        str: A deterministic, JSON-encoded string representation of the data.
    """
    if not isinstance(justification, str):
        raise TypeError("justification must be a string")

    if not justification.strip():
        raise ValueError("justification cannot be empty or whitespace only")

    if not isinstance(scores, dict):
        raise TypeError("scores must be a dictionary")

    # Determine which set of keys are present
    expected_c1_c6 = {f"c{i}" for i in range(1, 7)}
    expected_c1_c6_score = {f"c{i}_score" for i in range(1, 7)}

    keys = set(scores.keys())
    if expected_c1_c6.issubset(keys):
        target_keys = expected_c1_c6
    elif expected_c1_c6_score.issubset(keys):
        target_keys = expected_c1_c6_score
    else:
        raise ValueError(
            "scores dictionary must contain all 6 calibration keys (c1..c6 or c1_score..c6_score)"
        )

    # Validate score values
    validated_scores = {}
    for k in target_keys:
        val = scores[k]
        if val is None:
            raise ValueError(f"Score for key '{k}' cannot be None")
        # Ensure it is a number and not a boolean (bool inherits from int in Python)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise TypeError(f"Score for key '{k}' must be a number, got {type(val).__name__}")
        if not (1 <= val <= 10):
            raise ValueError(f"Score for key '{k}' must be between 1 and 10 inclusive, got {val}")
        validated_scores[k] = val

    payload = {
        "scores": validated_scores,
        "justification": justification,
    }

    # Use sort_keys=True for deterministic JSON serialization
    return json.dumps(payload, sort_keys=True)


def deserialize_document(document: str) -> Dict[str, Any]:
    """
    Deserialize a JSON-encoded document string back into its original Python structure.

    Validation is performed to ensure the deserialized document is structurally valid.

    Args:
        document: The JSON string document payload from ChromaDB.

    Returns:
        dict: A dictionary containing the keys:
            - 'scores': A dictionary containing C1-C6 scores
            - 'justification': The justification string
    """
    if not isinstance(document, str):
        raise TypeError("document must be a string")

    try:
        data = json.loads(document)
    except json.JSONDecodeError as e:
        raise ValueError(f"document is not a valid JSON string: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Deserialized document payload must be a JSON object (dict)")

    if "scores" not in data or "justification" not in data:
        raise ValueError("Document payload is missing required 'scores' or 'justification' fields")

    scores = data["scores"]
    justification = data["justification"]

    if not isinstance(scores, dict):
        raise ValueError("'scores' field must be a dictionary")

    if not isinstance(justification, str):
        raise ValueError("'justification' field must be a string")

    if not justification.strip():
        raise ValueError("'justification' field cannot be empty or whitespace only")

    # Determine which set of keys are present in scores
    expected_c1_c6 = {f"c{i}" for i in range(1, 7)}
    expected_c1_c6_score = {f"c{i}_score" for i in range(1, 7)}

    keys = set(scores.keys())
    if expected_c1_c6.issubset(keys):
        target_keys = expected_c1_c6
    elif expected_c1_c6_score.issubset(keys):
        target_keys = expected_c1_c6_score
    else:
        raise ValueError(
            "Deserialized scores must contain all 6 calibration keys (c1..c6 or c1_score..c6_score)"
        )

    # Validate score values
    for k in target_keys:
        val = scores[k]
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise TypeError(f"Deserialized score for key '{k}' must be a number, got {type(val).__name__}")
        if not (1 <= val <= 10):
            raise ValueError(f"Deserialized score for key '{k}' must be between 1 and 10 inclusive, got {val}")

    return {
        "scores": {k: scores[k] for k in target_keys},
        "justification": justification,
    }
