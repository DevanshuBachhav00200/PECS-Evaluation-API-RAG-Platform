"""Warm-loaded Sentence Transformer model for PECS Team 2.

Owner: Member 3 (Embedding + ChromaDB + Retrieval)
Task: EMB-001 — Set up sentence-transformer model loading.

The model is loaded once at module import time and reused by all
downstream callers through get_embedder().
"""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME: str = "all-MiniLM-L6-v2"

# Warm-load the model exactly once when this module is imported.
logger.info(
    "Loading sentence-transformer model '%s' (warm-load, once).",
    MODEL_NAME,
)

_model: SentenceTransformer = SentenceTransformer(MODEL_NAME)

logger.info(
    "Sentence-transformer model '%s' loaded and ready.",
    MODEL_NAME,
)


def get_embedder() -> SentenceTransformer:
    """Return the single warm-loaded SentenceTransformer instance.

    Returns:
        SentenceTransformer: The shared model instance.
    """
    return _model


def get_model_name() -> str:
    """Return the canonical embedding model name.

    Returns:
        str: The configured model name.
    """
    return MODEL_NAME