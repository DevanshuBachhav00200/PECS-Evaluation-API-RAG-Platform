"""
VDB-010 — Metadata Filtering on Retrieval.

Provides filtered_retrieve(), which layers metadata-based domain/level
constraints on top of ChromaDB nearest-neighbour search.

The filter is applied server-side during the ChromaDB query so that
candidates are restricted BEFORE nearest-neighbour selection — no
post-hoc Python filtering is performed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.vectorstore.metadata_schema import VALID_DOMAINS, VALID_LEVELS

# Embedding dimension enforced across the repository
_EMBEDDING_DIM = 384


def filtered_retrieve(
    query_embedding: List[float],
    n_results: int = 1,
    domain: Optional[str] = None,
    level: Optional[str] = None,
    collection: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Query the ChromaDB prompt_examples collection with optional metadata filters.

    Metadata filtering is applied server-side inside ChromaDB before nearest-
    neighbour ranking, so only eligible candidates are ever considered.

    When neither ``domain`` nor ``level`` is supplied the function behaves
    identically to VDB-009 ``retrieve()``.

    Args:
        query_embedding: A 384-dimensional list of numeric values.
        n_results:       Number of nearest results to return (default 1).
                         Must be a positive integer.
        domain:          Optional domain filter.  Must be one of the canonical
                         values defined in VDB-004 (e.g. ``"CODE"`` or
                         ``"EDGE"``).  When supplied, results are restricted
                         to records whose ``domain`` metadata field equals
                         this value.
        level:           Optional level filter.  Must be one of the canonical
                         values defined in VDB-004 (e.g. ``"weak"``,
                         ``"developing"``, ``"strong"``).  When supplied,
                         results are restricted to records whose ``level``
                         metadata field equals this value.
        collection:      Optional ChromaDB collection instance.  When omitted
                         the VDB-006 client and VDB-002 collection are used.

    Returns:
        A dictionary mirroring the ChromaDB ``collection.query()`` structure::

            {
                "ids":       list[list[str]],
                "documents": list[list[str | None]],
                "metadatas": list[list[dict | None]],
                "distances": list[list[float]] | None,
            }

        An empty inner list (``ids[0] == []``) is returned when no records
        match the requested filter — this is never treated as an error.

    Raises:
        TypeError:  If ``query_embedding`` is not a list or contains
                    non-numeric values, or if ``n_results`` is not an int.
        ValueError: If ``query_embedding`` is None/empty/wrong dimension,
                    ``n_results`` is not positive, ``domain`` is not a
                    recognised canonical domain, or ``level`` is not a
                    recognised canonical level.
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
    # 3. Validate optional domain / level filters                          #
    # ------------------------------------------------------------------ #
    if domain is not None:
        if not isinstance(domain, str):
            raise TypeError(f"domain must be a string, got {type(domain).__name__}")
        if domain not in VALID_DOMAINS:
            raise ValueError(
                f"Invalid domain '{domain}'; must be one of {sorted(VALID_DOMAINS)}"
            )

    if level is not None:
        if not isinstance(level, str):
            raise TypeError(f"level must be a string, got {type(level).__name__}")
        if level not in VALID_LEVELS:
            raise ValueError(
                f"Invalid level '{level}'; must be one of {sorted(VALID_LEVELS)}"
            )

    # ------------------------------------------------------------------ #
    # 4. Resolve collection                                                #
    # ------------------------------------------------------------------ #
    if collection is None:
        from src.vectorstore.client import get_client
        from src.vectorstore.init import get_or_create_collection
        collection = get_or_create_collection(get_client())

    # ------------------------------------------------------------------ #
    # 5. Build ChromaDB where-clause                                       #
    # ------------------------------------------------------------------ #
    where: Optional[Dict[str, Any]] = None

    if domain is not None and level is not None:
        # Both filters: use ChromaDB's $and operator
        where = {
            "$and": [
                {"domain": {"$eq": domain}},
                {"level": {"$eq": level}},
            ]
        }
    elif domain is not None:
        where = {"domain": {"$eq": domain}}
    elif level is not None:
        where = {"level": {"$eq": level}}

    # ------------------------------------------------------------------ #
    # 6. Execute ChromaDB query                                            #
    # ------------------------------------------------------------------ #
    query_kwargs: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        query_kwargs["where"] = where

    results: Dict[str, Any] = collection.query(**query_kwargs)
    return results
