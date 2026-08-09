"""Embedding-input preprocessing for PECS."""

from __future__ import annotations

from src.common.normalization import normalize_text


def preprocess_for_embedding(text: str) -> str:
    """Normalize raw text using the canonical DS-005 rules."""
    return normalize_text(text)