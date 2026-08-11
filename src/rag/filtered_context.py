"""
RAG-005 — Metadata-aware retrieval filtering.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Lets retrieval be restricted by domain/level metadata when the calling context
(the Question Engine's task record) specifies them. The filter is applied
server-side inside ChromaDB by VDB-010, so a domain-scoped evaluation request
can never retrieve an out-of-domain calibration example.

A filter that matches nothing is NOT an error: it degrades to the RAG-006
'no calibration available' result with reason ``no_filter_match``, so a
domain we have no gold examples for yet still gets graded.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.rag.config import DEFAULT_TOP_K
from src.rag.empty_retrieval_handler import (
    REASON_MALFORMED_RECORD,
    REASON_NO_FILTER_MATCH,
    is_empty,
    no_calibration_result,
)
from src.rag.query_embed import embed_query
from src.rag.retrieve_context import build_examples, retrieve_context


def retrieve_filtered_context(
    student_prompt: str,
    domain: Optional[str] = None,
    level: Optional[str] = None,
    n_results: Optional[int] = None,
    collection: Optional[Any] = None,
    embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    """Retrieve a calibration example restricted to a domain and/or level.

    With neither ``domain`` nor ``level`` supplied this is exactly RAG-003
    ``retrieve_context()``.

    Args:
        student_prompt: Raw candidate prompt text.
        domain:         Optional canonical domain filter (VDB-004 enumeration).
        level:          Optional canonical level filter (VDB-004 enumeration).
        n_results:      Top-K override; defaults to ``config.DEFAULT_TOP_K``.
        collection:     Optional ChromaDB collection (test injection point).
        embedder:       Optional embedder instance (test injection point).

    Returns:
        dict: Same shape as ``retrieve_context()``, plus the ``domain`` and
              ``level`` that were applied.

    Raises:
        TypeError:  On a non-string prompt/domain/level or non-int n_results.
        ValueError: On an empty prompt, non-positive n_results, or a domain or
                    level outside the VDB-004 enumerations.
    """
    if domain is None and level is None:
        result = retrieve_context(
            student_prompt,
            n_results=n_results,
            collection=collection,
            embedder=embedder,
        )
        result["domain"] = None
        result["level"] = None
        return result

    top_k = DEFAULT_TOP_K if n_results is None else n_results

    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError(f"n_results must be a positive integer, got {top_k!r}")
    if top_k <= 0:
        raise ValueError(f"n_results must be a positive integer, got {top_k}")

    query_embedding = embed_query(student_prompt, embedder=embedder)

    from src.vectorstore.filtered_retrieve import filtered_retrieve

    # Domain/level validation against the canonical enumerations happens inside
    # filtered_retrieve (VDB-010) — the single source of truth for those values.
    results = filtered_retrieve(
        query_embedding=query_embedding,
        n_results=top_k,
        domain=domain,
        level=level,
        collection=collection,
    )

    if is_empty(results):
        return no_calibration_result(
            reason=REASON_NO_FILTER_MATCH,
            detail=(
                "No calibration example matched the requested metadata filter "
                f"(domain={domain!r}, level={level!r})."
            ),
            domain=domain,
            level=level,
        )

    examples = build_examples(results)

    if not examples:
        return no_calibration_result(
            reason=REASON_MALFORMED_RECORD,
            detail="Filtered records could not be parsed into calibration examples.",
            domain=domain,
            level=level,
        )

    closest = examples[0]

    return {
        "available": True,
        "calibration": closest["calibration"],
        "record_id": closest["record_id"],
        "distance": closest["distance"],
        "metadata": closest["metadata"],
        "examples": examples,
        "domain": domain,
        "level": level,
    }
