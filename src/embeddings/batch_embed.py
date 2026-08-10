"""
EMB-003 — Batch embedding generation.

Provides a reusable function for converting raw text values into
dense embedding vectors using the warm-loaded SentenceTransformer
model from EMB-001.

Responsibilities:
- Reuse the warm-loaded model through get_embedder().
- Reuse EMB-002 preprocessing through preprocess_for_embedding().
- Process texts in configurable batches.
- Preserve input ordering.
- Return a NumPy array of 384-dimensional vectors.

This module does NOT:
- Load SentenceTransformer directly.
- Implement normalization logic.
- Persist embeddings.
- Implement ChromaDB.
- Implement retrieval or RAG.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.embeddings.embed_preprocess import preprocess_for_embedding
from src.embeddings.embedder import get_embedder


# all-MiniLM-L6-v2 produces 384-dimensional embeddings.
EMBEDDING_DIM: int = 384

# Default number of texts processed in a single encode() call.
DEFAULT_BATCH_SIZE: int = 32


def embed_texts(
    texts: Sequence[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    Generate embeddings for a sequence of text values in batches.

    Each input text is first passed through the canonical EMB-002
    preprocessing function. The preprocessed texts are then encoded
    using the single warm-loaded embedding model provided by EMB-001.

    Input ordering is preserved: row i of the returned array
    corresponds to texts[i].

    Args:
        texts:
            Sequence of raw text values to embed.

        batch_size:
            Maximum number of texts passed to model.encode() in one
            batch. Must be a positive integer. Defaults to 32.

    Returns:
        NumPy array with shape (N, 384), where N is the number of
        input texts.

        For empty input, returns an empty float32 array with shape
        (0, 384). In this case, get_embedder() and model.encode()
        are not called by this function.

    Raises:
        ValueError:
            If batch_size is not a positive integer.
    """
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise ValueError(
            f"batch_size must be a positive integer, got {batch_size!r}"
        )

    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be a positive integer, got {batch_size!r}"
        )

    text_list = list(texts)

    # Do not obtain the model for empty input.
    if not text_list:
        return np.empty(
            (0, EMBEDDING_DIM),
            dtype=np.float32,
        )

    # EMB-002: apply the canonical preprocessing before embedding.
    preprocessed_texts = [
        preprocess_for_embedding(text)
        for text in text_list
    ]

    # EMB-001: obtain the already warm-loaded shared model.
    model = get_embedder()

    batch_vectors: list[np.ndarray] = []

    for start in range(0, len(preprocessed_texts), batch_size):
        batch = preprocessed_texts[start : start + batch_size]

        encoded = model.encode(batch)

        vectors = np.asarray(
            encoded,
            dtype=np.float32,
        )

        batch_vectors.append(vectors)

    # All batches are concatenated in their original order.
    embeddings = np.vstack(batch_vectors)

    return embeddings.astype(np.float32, copy=False)