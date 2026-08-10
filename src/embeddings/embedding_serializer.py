"""
EMB-006 — Embedding persistence for ChromaDB ingestion.

Purpose:
    Define the serialization boundary between the embedding generation
    layer (EMB-001/002/003) and the future ChromaDB ingestion layer
    (VDB-xxx). This module does NOT talk to ChromaDB -- it only produces
    and persists a plain, ingestion-ready structure that a future
    `collection.upsert(**loaded)` call can consume directly.

Input representation:
    A sequence of EmbeddingRecord objects, one per calibration example:
        - id:        unique string identifier (e.g. prompt_id)
        - embedding: a 384-dimensional numeric vector (np.ndarray or
                      any array-like), matching the EMB-003/EMB-004
                      contract (all-MiniLM-L6-v2, dimension 384)
        - metadata:  a plain dict of scalar fields to store alongside
                      the vector (e.g. domain, c1..c6, source)
        - document:  optional original text the vector was computed
                      from (e.g. prompt_text), stored for traceability

Serialized representation (JSON-safe, on disk):
    {
        "ids": [...],
        "embeddings": [[...], [...], ...],   # plain floats, not np.ndarray
        "metadatas": [{...}, {...}, ...],
        "documents": [str_or_null, ...]
    }

Deserialized / ingestion-ready representation:
    Identical in shape to the serialized representation above.
    `load_serialized()` returns exactly this dict, with no further
    conversion required -- it can be passed directly as
    `ingestion_function(**loaded)` to a future ChromaDB upsert wrapper
    shaped like:

        def upsert(ids, embeddings, metadatas, documents):
            collection.upsert(
                ids=ids, embeddings=embeddings,
                metadatas=metadatas, documents=documents,
            )

Relationship to future ChromaDB ingestion (VDB-xxx):
    This module's output is intentionally shaped to match ChromaDB's
    upsert() keyword arguments one-to-one. The ingestion layer built in
    a later task should be able to call
    `collection.upsert(**embedding_serializer.load_serialized(path))`
    with no intermediate reshaping, renaming, or type conversion.

Format choice:
    JSON was chosen because: (1) it requires no new dependency, (2) it
    is human-readable for debugging a small calibration corpus, (3) it
    round-trips cleanly for the float32 values produced by EMB-003 once
    converted to plain Python floats via `.tolist()`, and (4) nothing
    else in the repository has yet established a different persistence
    convention for embeddings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# Must stay in sync with the EMB-003/EMB-004 contract. Do not change
# without changing the embedding model itself.
EXPECTED_DIMENSION = 384


@dataclass
class EmbeddingRecord:
    """
    A single embedding record: one calibration example's ID, vector,
    and metadata, ready to be serialized for future ChromaDB ingestion.

    Attributes:
        id: Unique string identifier for this record (e.g. prompt_id
            from the calibration dataset). Must be unique within any
            single batch passed to serialize_embeddings().
        embedding: A 384-dimensional numeric vector. Any array-like
            (np.ndarray, list of floats) is accepted; it is converted
            to float32 during serialization.
        metadata: A plain dict of scalar fields to store alongside the
            vector (e.g. {"domain": "CREATE", "c1": 10.0, ...}).
            Defaults to an empty dict, never None.
        document: The original text the embedding was computed from
            (e.g. prompt_text). Optional; stored for traceability and
            for ChromaDB's `documents` field. Defaults to None.
    """

    id: str
    embedding: Any  # np.ndarray or array-like; validated at serialize time
    metadata: Dict[str, Any] = field(default_factory=dict)
    document: Optional[str] = None


def _validate_records(records: Sequence[EmbeddingRecord]) -> None:
    """
    Validate a batch of EmbeddingRecord objects before serialization.

    Raises ValueError with a specific, actionable message on the first
    problem found. Never silently drops or truncates a record.
    """
    seen_ids = set()

    for index, record in enumerate(records):
        if not record.id:
            raise ValueError(
                f"record at index {index} has an empty or falsy id: "
                f"{record.id!r}"
            )

        if record.id in seen_ids:
            raise ValueError(
                f"duplicate id found: {record.id!r}. Every record in a "
                f"batch passed to serialize_embeddings() must have a "
                f"unique id, since ChromaDB upsert keys on id."
            )
        seen_ids.add(record.id)

        vector = np.asarray(record.embedding)

        if vector.ndim != 1:
            raise ValueError(
                f"record {record.id!r}: embedding must be 1-dimensional, "
                f"got ndim={vector.ndim} (shape={vector.shape})"
            )

        if vector.shape[0] != EXPECTED_DIMENSION:
            raise ValueError(
                f"record {record.id!r}: expected embedding dimension "
                f"{EXPECTED_DIMENSION}, got {vector.shape[0]}"
            )

        if not np.issubdtype(vector.dtype, np.number):
            raise ValueError(
                f"record {record.id!r}: embedding must contain numeric "
                f"values, got dtype {vector.dtype}"
            )

        if not np.all(np.isfinite(vector.astype(np.float64))):
            raise ValueError(
                f"record {record.id!r}: embedding contains non-finite "
                f"values (NaN or Inf)"
            )

        if record.metadata is None:
            raise ValueError(
                f"record {record.id!r}: metadata must not be None -- "
                f"use an empty dict {{}} for 'no metadata'"
            )


def serialize_embeddings(records: Sequence[EmbeddingRecord]) -> Dict[str, Any]:
    """
    Convert a batch of EmbeddingRecord objects into an ingestion-ready,
    JSON-safe dict shaped to match ChromaDB's upsert() keyword arguments.

    Args:
        records: A sequence of EmbeddingRecord objects. May be empty.

    Returns:
        A dict with keys "ids", "embeddings", "metadatas", "documents".
        Each value is a plain Python list (no NumPy arrays), safe for
        `json.dump()` and directly usable as
        `ingestion_function(**serialize_embeddings(records))`.

        For empty input, returns
        {"ids": [], "embeddings": [], "metadatas": [], "documents": []}
        without performing any validation (there is nothing to validate).

    Raises:
        ValueError: If any record is invalid -- see _validate_records
            for the specific conditions checked (empty/duplicate id,
            wrong embedding shape/dimension, non-numeric or non-finite
            values, missing metadata).
    """
    records = list(records)

    if len(records) == 0:
        return {"ids": [], "embeddings": [], "metadatas": [], "documents": []}

    _validate_records(records)

    return {
        "ids": [record.id for record in records],
        "embeddings": [
            np.asarray(record.embedding, dtype=np.float32).tolist()
            for record in records
        ],
        "metadatas": [dict(record.metadata) for record in records],
        "documents": [record.document for record in records],
    }


def save_serialized(serialized: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Persist a serialized embedding batch (as returned by
    serialize_embeddings()) to disk as JSON.

    Creates parent directories if they do not already exist.

    Args:
        serialized: The dict returned by serialize_embeddings().
        path: Destination file path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as f:
        json.dump(serialized, f)


def load_serialized(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a previously-saved serialized embedding batch back into an
    ingestion-ready dict, with no further transformation required.

    Args:
        path: Path to a file previously written by save_serialized().

    Returns:
        A dict with keys "ids", "embeddings", "metadatas", "documents",
        identical in shape to what serialize_embeddings() produced.
        This can be passed directly as
        `ingestion_function(**load_serialized(path))`.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8") as f:
        return json.load(f)