"""
RAG-002 — Query embedding step.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Reuses the warm-loaded sentence-transformer from EMB-001 to embed the candidate
prompt at request time. The model is NOT reloaded per request: the import of
``src.embeddings.embedder`` is deferred to first use and the module-level
instance returned by ``get_embedder()`` is shared with batch ingestion.
"""

from __future__ import annotations

from typing import Any, List, Optional

from src.rag.config import EMBEDDING_DIM
from src.rag.query_preprocess import preprocess_query


def get_shared_embedder() -> Any:
    """Return the process-wide warm-loaded sentence-transformer (EMB-001).

    Imported lazily so that modules which only need request construction or
    scoring do not pay the model-load cost.
    """
    from src.embeddings.embedder import get_embedder

    return get_embedder()


def embed_query(
    student_prompt: str,
    embedder: Optional[Any] = None,
    preprocess: bool = True,
) -> List[float]:
    """Embed a candidate prompt for retrieval.

    Args:
        student_prompt: Raw candidate prompt text.
        embedder:       Optional embedder instance exposing ``encode()``. When
                        omitted the shared EMB-001 instance is used — the same
                        object batch ingestion uses, never a fresh load.
        preprocess:     Apply RAG-001 normalization first (default True). Pass
                        False only when the caller has already normalized.

    Returns:
        list[float]: A 384-dimensional embedding, ready for VDB-009 retrieve().

    Raises:
        TypeError:  If ``student_prompt`` is not a string.
        ValueError: If the prompt is empty, or the model returns an embedding
                    of unexpected dimensionality.
    """
    text = preprocess_query(student_prompt) if preprocess else student_prompt

    if not isinstance(text, str):
        raise TypeError(f"student_prompt must be a string, got {type(text).__name__}")
    if not text.strip():
        raise ValueError("student_prompt cannot be empty or whitespace only")

    model = embedder if embedder is not None else get_shared_embedder()

    vector = model.encode(text)

    # sentence-transformers returns a numpy array; normalise to a plain list so
    # the ChromaDB layer (VDB-009) receives exactly what it validates for.
    if hasattr(vector, "tolist"):
        vector = vector.tolist()

    if not isinstance(vector, list):
        raise ValueError(
            f"embedder returned an unsupported type: {type(vector).__name__}"
        )

    # A single-text encode() must not return a batch.
    if vector and isinstance(vector[0], list):
        if len(vector) != 1:
            raise ValueError(
                f"expected a single embedding, got a batch of {len(vector)}"
            )
        vector = vector[0]

    embedding = [float(value) for value in vector]

    if len(embedding) != EMBEDDING_DIM:
        raise ValueError(
            f"query embedding must have exactly {EMBEDDING_DIM} dimensions, "
            f"got {len(embedding)}"
        )

    return embedding
