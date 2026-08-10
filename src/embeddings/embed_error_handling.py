"""
EMB-007 — Embedding input error handling.

Purpose:
Handle invalid embedding inputs gracefully before they reach the
embedding model.

The implementation covers the two cases explicitly required by the
EMB-007 task:

1. Empty or whitespace-only text.
2. Oversized text.

Instead of allowing these inputs to cause an unhandled exception,
the validator returns a structured result describing whether the
input is valid and, when invalid, the corresponding error code and
message.

The existing EMB-002 preprocessing remains the single source of
truth for normalization. This module does not introduce a second
normalization implementation.

The maximum character limit is configurable because the task tracker
specifies that oversized text must be handled but does not prescribe
a specific character threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.embeddings.embed_preprocess import preprocess_for_embedding


# Application-level safety guard.
# This is deliberately configurable and is NOT claimed to be the
# SentenceTransformer model's native token limit.
DEFAULT_MAX_TEXT_CHARS = 10_000


@dataclass(frozen=True)
class EmbeddingInputResult:
    """
    Structured result returned by embedding-input validation.

    Attributes:
        is_valid:
            True when the input can proceed to embedding.

        text:
            The normalized text when valid, otherwise None.

        error_code:
            Stable machine-readable error code when invalid.

        error_message:
            Human-readable explanation when invalid.
    """

    is_valid: bool
    text: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


def validate_embedding_input(
    text: str,
    max_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> EmbeddingInputResult:
    """
    Validate and preprocess text before embedding.

    The existing EMB-002 preprocessing function is reused so that
    validation and downstream embedding continue to use the same
    canonical normalization behavior.

    Args:
        text:
            Raw text that is intended to be embedded.

        max_chars:
            Maximum permitted length of the normalized text in
            characters. Defaults to DEFAULT_MAX_TEXT_CHARS.

    Returns:
        EmbeddingInputResult:
            A structured success result containing normalized text,
            or a structured error result for invalid input.

    Raises:
        ValueError:
            If max_chars is not a positive integer.

    Notes:
        Invalid text is handled through the returned structured
        result rather than by raising an exception.
    """

    if not isinstance(max_chars, int) or isinstance(max_chars, bool):
        raise ValueError("max_chars must be a positive integer")

    if max_chars <= 0:
        raise ValueError(
            f"max_chars must be a positive integer, got {max_chars!r}"
        )

    try:
        normalized_text = preprocess_for_embedding(text)
    except Exception as exc:
        return EmbeddingInputResult(
            is_valid=False,
            error_code="PREPROCESSING_ERROR",
            error_message=f"Unable to preprocess embedding input: {exc}",
        )

    if not normalized_text:
        return EmbeddingInputResult(
            is_valid=False,
            error_code="EMPTY_INPUT",
            error_message="Embedding input is empty after normalization.",
        )

    if len(normalized_text) > max_chars:
        return EmbeddingInputResult(
            is_valid=False,
            error_code="INPUT_TOO_LARGE",
            error_message=(
                f"Embedding input exceeds the maximum allowed size of "
                f"{max_chars} characters "
                f"(received {len(normalized_text)} characters)."
            ),
        )

    return EmbeddingInputResult(
        is_valid=True,
        text=normalized_text,
    )


def prepare_embedding_input(
    text: str,
    max_chars: int = DEFAULT_MAX_TEXT_CHARS,
) -> dict:
    """
    Return a JSON-friendly structured result for an embedding input.

    This helper is useful at an API boundary because it returns only
    plain Python values and therefore can be serialized directly.

    Args:
        text:
            Raw text intended for embedding.

        max_chars:
            Maximum permitted normalized text length.

    Returns:
        dict:
            A structured success or error response.
    """

    result = validate_embedding_input(
        text=text,
        max_chars=max_chars,
    )

    return {
        "is_valid": result.is_valid,
        "text": result.text,
        "error_code": result.error_code,
        "error_message": result.error_message,
    }