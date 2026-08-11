"""
RAG-003 — retrieve_context(student_prompt).

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Encodes the candidate prompt, queries ChromaDB through Member 3's VDB-009
retrieve(), and returns the closest pre-scored calibration example already
formatted into the compact line the prompt builder consumes.

Contract with the layer below: this module never instantiates a ChromaDB client
or a sentence-transformer itself. It reuses VDB-006/VDB-002 (via retrieve) and
EMB-001 (via embed_query), and both are injectable for testing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.rag.config import DEFAULT_TOP_K
from src.rag.empty_retrieval_handler import (
    REASON_EMPTY_COLLECTION,
    REASON_MALFORMED_RECORD,
    is_empty,
    no_calibration_result,
)
from src.rag.format_calibration import format_calibration_from_document
from src.rag.query_embed import embed_query


def _inner(results: Dict[str, Any], key: str) -> List[Any]:
    """Pull the first (only) query's inner list for ``key``, tolerating None."""
    outer = results.get(key)
    if not outer:
        return []
    first = outer[0]
    return list(first) if first else []


def build_examples(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a ChromaDB query result into formatted calibration examples.

    Args:
        results: The dict returned by VDB-009 ``retrieve()`` / VDB-010
                 ``filtered_retrieve()``.

    Returns:
        list[dict]: One entry per retrieved record, in ChromaDB's distance
                    order, each with ``record_id``, ``calibration``,
                    ``metadata`` and ``distance``. Records whose document
                    cannot be deserialized are skipped rather than fatal.
    """
    ids = _inner(results, "ids")
    documents = _inner(results, "documents")
    metadatas = _inner(results, "metadatas")
    distances = _inner(results, "distances")

    examples: List[Dict[str, Any]] = []

    for index, record_id in enumerate(ids):
        document = documents[index] if index < len(documents) else None
        metadata = metadatas[index] if index < len(metadatas) else None
        distance = distances[index] if index < len(distances) else None

        if not document:
            continue

        try:
            calibration = format_calibration_from_document(document, metadata)
        except (TypeError, ValueError):
            # A single malformed stored record must not fail the request.
            continue

        examples.append(
            {
                "record_id": record_id,
                "calibration": calibration,
                "metadata": metadata,
                "distance": distance,
            }
        )

    return examples


def retrieve_context(
    student_prompt: str,
    n_results: Optional[int] = None,
    collection: Optional[Any] = None,
    embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    """Retrieve the closest pre-scored calibration example for a prompt.

    Args:
        student_prompt: Raw candidate prompt text.
        n_results:      Top-K override. Defaults to ``config.DEFAULT_TOP_K``
                        (1 in the MVP) — changing the config value changes the
                        number of returned examples with no other code edit.
        collection:     Optional ChromaDB collection (test injection point).
        embedder:       Optional embedder instance (test injection point).

    Returns:
        dict with keys:
            available   – bool, False when nothing usable was retrieved
            calibration – the compact line for the closest example, or the
                          'no calibration available' placeholder
            record_id   – id of the closest example (None when unavailable)
            distance    – distance of the closest example (None when unavailable)
            metadata    – VDB-004 metadata of the closest example
            examples    – all retrieved examples, closest first

    Raises:
        TypeError:  If ``student_prompt`` is not a string or ``n_results`` is
                    not an int.
        ValueError: If the prompt is empty or ``n_results`` is not positive.
    """
    top_k = DEFAULT_TOP_K if n_results is None else n_results

    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError(f"n_results must be a positive integer, got {top_k!r}")
    if top_k <= 0:
        raise ValueError(f"n_results must be a positive integer, got {top_k}")

    query_embedding = embed_query(student_prompt, embedder=embedder)

    from src.vectorstore.retrieve import retrieve

    results = retrieve(
        query_embedding=query_embedding,
        n_results=top_k,
        collection=collection,
    )

    if is_empty(results):
        return no_calibration_result(
            reason=REASON_EMPTY_COLLECTION,
            detail="Vector store returned no records for this query.",
        )

    examples = build_examples(results)

    if not examples:
        return no_calibration_result(
            reason=REASON_MALFORMED_RECORD,
            detail="Retrieved records could not be parsed into calibration examples.",
        )

    closest = examples[0]

    return {
        "available": True,
        "calibration": closest["calibration"],
        "record_id": closest["record_id"],
        "distance": closest["distance"],
        "metadata": closest["metadata"],
        "examples": examples,
    }
