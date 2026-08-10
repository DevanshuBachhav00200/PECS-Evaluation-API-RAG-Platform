"""
VDB-012 — Bulk Ingestion from Serialised Embeddings.

Provides bulk_ingest_from_serialized(), which accepts the dictionary
produced by EMB-006 (embedding_serializer) and batch-upserts all records
into a ChromaDB collection in a single operation.

This closes the handoff gap described in docs/embedding.md §12:
    data = load_serialized("data/processed/embeddings.json")
    collection.upsert(**data)

The function validates the payload structure, validates metadata for every
record, and delegates the actual write to the existing VDB-007 upsert_record.

Architecture boundary:
    EMB-006 owns serialization/deserialization.
    VDB-007 owns per-record upsert semantics.
    VDB-012 owns the bulk handoff layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.vectorstore.metadata_schema import validate_metadata

# Embedding dimension enforced across the repository
_EMBEDDING_DIM = 384

# Required top-level keys produced by EMB-006 serialize_embeddings()
_REQUIRED_PAYLOAD_KEYS = {"ids", "embeddings", "metadatas", "documents"}


def bulk_ingest_from_serialized(
    payload: Dict[str, Any],
    collection: Optional[Any] = None,
) -> int:
    """
    Batch-upsert all records from an EMB-006 serialised payload into ChromaDB.

    The ``payload`` must be a dictionary with the following keys (exactly the
    structure produced by ``embedding_serializer.serialize_embeddings()``):

    .. code-block:: python

        {
            "ids":        list[str],
            "embeddings": list[list[float]],   # each inner list has 384 elements
            "metadatas":  list[dict[str, Any]],
            "documents":  list[str],
        }

    All four lists must have the same length.  Each record is validated
    individually before the upsert is executed so that a malformed payload
    fails fast without partially writing to the collection.

    The operation is idempotent: upserting a payload that contains IDs already
    present in the collection overwrites those records with the new content,
    leaving exactly one record per ID.

    Args:
        payload:    Dictionary produced by ``serialize_embeddings()`` /
                    ``load_serialized()``.  Must contain ``ids``,
                    ``embeddings``, ``metadatas``, and ``documents``.
        collection: Optional ChromaDB collection instance.  When omitted
                    the VDB-006 client and VDB-002 collection are used.

    Returns:
        The number of records successfully upserted.

    Raises:
        TypeError:  If ``payload`` is not a dict, or if any individual field
                    has the wrong type.
        ValueError: If required keys are missing, lists have mismatched lengths,
                    any embedding has the wrong dimension, or any metadata
                    fails VDB-004 validation.
    """
    # ------------------------------------------------------------------ #
    # 1. Validate payload structure                                        #
    # ------------------------------------------------------------------ #
    if not isinstance(payload, dict):
        raise TypeError(
            f"payload must be a dictionary, got {type(payload).__name__}"
        )

    missing_keys = _REQUIRED_PAYLOAD_KEYS - set(payload.keys())
    if missing_keys:
        raise ValueError(
            f"payload is missing required keys: {sorted(missing_keys)}"
        )

    ids: List[str] = payload["ids"]
    embeddings: List[List[float]] = payload["embeddings"]
    metadatas: List[Dict[str, Any]] = payload["metadatas"]
    documents: List[str] = payload["documents"]

    # ------------------------------------------------------------------ #
    # 2. Validate list types                                               #
    # ------------------------------------------------------------------ #
    if not isinstance(ids, list):
        raise TypeError(f"payload['ids'] must be a list, got {type(ids).__name__}")
    if not isinstance(embeddings, list):
        raise TypeError(f"payload['embeddings'] must be a list, got {type(embeddings).__name__}")
    if not isinstance(metadatas, list):
        raise TypeError(f"payload['metadatas'] must be a list, got {type(metadatas).__name__}")
    if not isinstance(documents, list):
        raise TypeError(f"payload['documents'] must be a list, got {type(documents).__name__}")

    # ------------------------------------------------------------------ #
    # 3. Validate list lengths match                                       #
    # ------------------------------------------------------------------ #
    n = len(ids)
    if len(embeddings) != n or len(metadatas) != n or len(documents) != n:
        raise ValueError(
            f"All payload lists must have the same length. "
            f"Got ids={n}, embeddings={len(embeddings)}, "
            f"metadatas={len(metadatas)}, documents={len(documents)}"
        )

    if n == 0:
        return 0

    # ------------------------------------------------------------------ #
    # 4. Validate each record individually (fail fast)                     #
    # ------------------------------------------------------------------ #
    for i in range(n):
        # ID
        curr_id = ids[i]
        if not isinstance(curr_id, str) or not curr_id.strip():
            raise ValueError(
                f"payload['ids'][{i}] must be a non-empty string, got {curr_id!r}"
            )

        # Embedding
        curr_emb = embeddings[i]
        if not isinstance(curr_emb, list):
            raise TypeError(
                f"payload['embeddings'][{i}] must be a list, got {type(curr_emb).__name__}"
            )
        if len(curr_emb) != _EMBEDDING_DIM:
            raise ValueError(
                f"payload['embeddings'][{i}] must have exactly {_EMBEDDING_DIM} "
                f"dimensions, got {len(curr_emb)}"
            )
        for j, val in enumerate(curr_emb):
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(
                    f"payload['embeddings'][{i}][{j}] must be numeric, "
                    f"got {type(val).__name__}"
                )

        # Metadata — reuse VDB-004 validation
        validate_metadata(metadatas[i])

        # Document
        curr_doc = documents[i]
        if not isinstance(curr_doc, str) or not curr_doc.strip():
            raise ValueError(
                f"payload['documents'][{i}] must be a non-empty string, got {curr_doc!r}"
            )

    # ------------------------------------------------------------------ #
    # 5. Resolve collection                                                #
    # ------------------------------------------------------------------ #
    if collection is None:
        from src.vectorstore.client import get_client
        from src.vectorstore.init import get_or_create_collection
        collection = get_or_create_collection(get_client())

    # ------------------------------------------------------------------ #
    # 6. Delegate to VDB-007 batch upsert                                 #
    # ------------------------------------------------------------------ #
    from src.vectorstore.upsert import upsert_record
    upsert_record(
        collection=collection,
        record_id=ids,
        embedding=embeddings,
        document=documents,
        metadata=metadatas,
    )

    return n
