"""
VDB-007 — Document Upsert with Duplicate-ID Handling.

Provides the upsert_record function to write records idempotently
into ChromaDB collections.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.vectorstore.metadata_schema import validate_metadata


def upsert_record(
    collection: Any,
    record_id: str | List[str],
    embedding: List[float] | List[List[float]],
    document: str | List[str],
    metadata: Dict[str, Any] | List[Dict[str, Any]],
) -> None:
    """
    Upsert one or more prepared vector-store records into the ChromaDB collection.

    Idempotency:
    - If the ID already exists, ChromaDB will overwrite/update the record with the
      new content, leaving exactly one record with that ID.

    Args:
        collection: The ChromaDB collection instance.
        record_id: A single ID string, or a list of ID strings.
        embedding: A single 384-dimensional list of floats, or a list of such embeddings.
        document: A single serialized document string, or a list of strings.
        metadata: A single metadata dictionary, or a list of metadata dictionaries.

    Raises:
        TypeError: If types of inputs or elements are invalid.
        ValueError: If IDs are empty, lengths mismatch, dimensions are wrong,
                    or metadata fails validation.
    """
    if collection is None:
        raise ValueError("ChromaDB collection must be provided")

    is_batch = isinstance(record_id, list)

    if is_batch:
        if not isinstance(embedding, list) or not isinstance(document, list) or not isinstance(metadata, list):
            raise TypeError("If record_id is a list, embedding, document, and metadata must also be lists")

        n = len(record_id)
        if len(embedding) != n or len(document) != n or len(metadata) != n:
            raise ValueError(
                "All batch input lists (record_id, embedding, document, metadata) must have the same length"
            )

        ids_list = record_id
        embeddings_list = embedding
        documents_list = document
        metadatas_list = metadata
    else:
        if not isinstance(record_id, str):
            raise TypeError("record_id must be a string or a list of strings")

        ids_list = [record_id]

        if not isinstance(embedding, list):
            raise TypeError("embedding must be a list of numbers")
        embeddings_list = [embedding]  # type: ignore

        if not isinstance(document, str):
            raise TypeError("document must be a string")
        documents_list = [document]

        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary")
        metadatas_list = [metadata]

    # Validate each item in the batch
    for i in range(len(ids_list)):
        curr_id = ids_list[i]
        curr_emb = embeddings_list[i]
        curr_doc = documents_list[i]
        curr_meta = metadatas_list[i]

        # 1. Validate ID
        if not isinstance(curr_id, str) or not curr_id.strip():
            raise ValueError(f"ID at index {i} must be a non-empty string")

        # 2. Validate embedding
        if not isinstance(curr_emb, list):
            raise TypeError(f"Embedding at index {i} must be a list of numbers")
        if len(curr_emb) != 384:
            raise ValueError(
                f"Embedding at index {i} must have exactly 384 dimensions, got {len(curr_emb)}"
            )
        for val in curr_emb:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(f"Embedding value at index {i} must be numeric, got {type(val).__name__}")

        # 3. Validate document
        if not isinstance(curr_doc, str) or not curr_doc.strip():
            raise ValueError(f"Document at index {i} must be a non-empty string")

        # 4. Validate metadata using VDB-004 validation
        validate_metadata(curr_meta)

    # Perform ChromaDB upsert
    collection.upsert(
        ids=ids_list,
        embeddings=embeddings_list,
        documents=documents_list,
        metadatas=metadatas_list,
    )
