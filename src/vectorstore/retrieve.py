"""
VDB-009 — Top-K Retrieval Function.

Provides the retrieve() function for nearest-neighbour lookup against
the ChromaDB prompt_examples collection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# Embedding dimension enforced across the repository (EMB / VDB-005 convention)
_EMBEDDING_DIM = 384


def retrieve(
    query_embedding: List[float],
    n_results: int = 1,
    collection: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Query the ChromaDB prompt_examples collection for the nearest neighbours
    of the supplied query embedding.

    The function does NOT instantiate a new PersistentClient or SentenceTransformer.
    When `collection` is not provided it reuses the VDB-006 get_client() and the
    VDB-002 get_or_create_collection() to obtain the default production collection.

    Args:
        query_embedding: A 384-dimensional list of numeric values to use as the query.
        n_results: Number of nearest results to return.  Must be a positive integer.
                   Defaults to 1.
        collection: Optional ChromaDB collection instance.  When provided the function
                    queries that collection directly; when omitted the module-level
                    VDB-006 client and VDB-002 collection are used.

    Returns:
        A dictionary with the same nested structure that ChromaDB.collection.query()
        returns, exposing the following top-level keys:

            ids        – list[list[str]]            one inner list per query
            documents  – list[list[str | None]]     one inner list per query
            metadatas  – list[list[dict | None]]    one inner list per query
            distances  – list[list[float]] | None   one inner list per query

    Raises:
        TypeError:  If query_embedding is not a list, contains non-numeric values,
                    or n_results is not an int.
        ValueError: If query_embedding is empty, has wrong dimensionality,
                    or n_results is not a positive integer.
    """
    # ------------------------------------------------------------------ #
    # 1. Validate query_embedding                                          #
    # ------------------------------------------------------------------ #
    if query_embedding is None:
        raise ValueError("query_embedding must be provided")

    if not isinstance(query_embedding, list):
        raise TypeError(
            f"query_embedding must be a list of floats, got {type(query_embedding).__name__}"
        )

    if len(query_embedding) == 0:
        raise ValueError("query_embedding must not be empty")

    for idx, val in enumerate(query_embedding):
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise TypeError(
                f"query_embedding element at index {idx} must be numeric, "
                f"got {type(val).__name__}"
            )

    if len(query_embedding) != _EMBEDDING_DIM:
        raise ValueError(
            f"query_embedding must have exactly {_EMBEDDING_DIM} dimensions, "
            f"got {len(query_embedding)}"
        )

    # ------------------------------------------------------------------ #
    # 2. Validate n_results                                                #
    # ------------------------------------------------------------------ #
    if not isinstance(n_results, int) or isinstance(n_results, bool):
        raise TypeError(
            f"n_results must be a positive integer, got {type(n_results).__name__}"
        )

    if n_results <= 0:
        raise ValueError(f"n_results must be a positive integer, got {n_results}")

    # ------------------------------------------------------------------ #
    # 3. Resolve collection                                                #
    # ------------------------------------------------------------------ #
    if collection is None:
        from src.vectorstore.client import get_client
        from src.vectorstore.init import get_or_create_collection
        collection = get_or_create_collection(get_client())

    # ------------------------------------------------------------------ #
    # 4. Query ChromaDB                                                    #
    # ------------------------------------------------------------------ #
    results: Dict[str, Any] = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    return results
