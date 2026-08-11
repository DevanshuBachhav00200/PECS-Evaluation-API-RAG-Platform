"""
RAG-001 — Candidate-prompt preprocessing at query time.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Applies exactly the same normalization as EMB-002 (ingestion time) to the
incoming candidate prompt before it is embedded for retrieval. Query-time and
ingestion-time normalization MUST produce identical output for identical text,
otherwise the query vector sits in a slightly different space than the corpus.

This module deliberately delegates to ``src.embeddings.embed_preprocess`` rather
than re-implementing the rules, so the two can never drift apart.
"""

from __future__ import annotations

from src.embeddings.embed_preprocess import preprocess_for_embedding


def preprocess_query(student_prompt: str) -> str:
    """Normalize a candidate prompt for query-time embedding.

    Args:
        student_prompt: Raw candidate prompt text as submitted.

    Returns:
        str: The normalized text, byte-identical to what EMB-002 would produce
             for the same input at ingestion time.

    Raises:
        TypeError:  If ``student_prompt`` is not a string.
        ValueError: If ``student_prompt`` is empty or whitespace only.
    """
    if not isinstance(student_prompt, str):
        raise TypeError(
            f"student_prompt must be a string, got {type(student_prompt).__name__}"
        )

    if not student_prompt.strip():
        raise ValueError("student_prompt cannot be empty or whitespace only")

    return preprocess_for_embedding(student_prompt)
