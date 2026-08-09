"""Shared text normalization utilities for PECS."""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Apply the canonical DS-005 text normalization."""
    text = str(text).lower().strip()
    return re.sub(r"\s+", " ", text)