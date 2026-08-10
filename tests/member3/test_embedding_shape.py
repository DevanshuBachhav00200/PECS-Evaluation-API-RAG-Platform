"""
EMB-004 — Verify embedding dimensionality & dtype.

Purpose:
    Catch silent model or configuration drift in the embedding pipeline.
    If the embedding model, its config, or a dependency upgrade ever
    changes the output shape or dtype, these tests should fail loudly
    rather than let a wrong shape/dtype quietly flow downstream into
    ChromaDB (VDB-xxx) or RAG (RAG-xxx).

Contract under test (owned by EMB-003 / src/embeddings/batch_embed.py):
    embed_texts(texts, batch_size=32) -> np.ndarray
        shape == (len(texts), 384)
        dtype == np.float32

Scope:
    This file verifies the EMB-003 output contract only. It does not
    load the real SentenceTransformer model -- real-model behavior is
    already covered by EMB-001 and EMB-003's own test suites. All tests
    here use a lightweight FakeEmbedder so they run fast and
    deterministically.
"""

import numpy as np
import pytest

from src.embeddings import batch_embed

# The two invariants this file exists to protect.
EXPECTED_DIMENSION = 384
EXPECTED_DTYPE = np.float32


class FakeEmbedder:
    """
    Minimal stand-in for the real SentenceTransformer model.

    Returns deterministic vectors of a configurable dimension and dtype,
    so we can simulate both correct output and drifted (wrong shape /
    wrong dtype) output without touching the real model or production code.
    """

    def __init__(self, dimension=EXPECTED_DIMENSION, dtype=np.float32):
        self.dimension = dimension
        self.dtype = dtype

    def encode(self, batch):
        return np.ones((len(batch), self.dimension), dtype=self.dtype)


def assert_embedding_contract(embeddings: np.ndarray, expected_n: int) -> None:
    """
    Shared assertion helper expressing the EMB-004 contract explicitly.

    Written as plain `assert` statements (not a silent no-op) so that any
    violation raises an AssertionError with a clear traceback -- i.e. it
    "fails loudly", per the EMB-004 acceptance criterion.
    """
    assert isinstance(embeddings, np.ndarray), (
        f"Expected a NumPy ndarray, got {type(embeddings)}"
    )
    assert embeddings.ndim == 2, (
        f"Expected a 2D array, got ndim={embeddings.ndim}"
    )
    assert embeddings.shape[0] == expected_n, (
        f"Expected {expected_n} vectors, got {embeddings.shape[0]}"
    )
    assert embeddings.shape[1] == EXPECTED_DIMENSION, (
        f"Expected embedding dimension {EXPECTED_DIMENSION}, "
        f"got {embeddings.shape[1]}"
    )
    assert embeddings.dtype == EXPECTED_DTYPE, (
        f"Expected dtype {EXPECTED_DTYPE}, got {embeddings.dtype}"
    )


# ---------------------------------------------------------------------------
# Normal contract checks (fake embedder, correct dimension/dtype)
# ---------------------------------------------------------------------------


def test_multiple_inputs_produce_expected_shape_and_dtype(monkeypatch):
    """5 input texts should produce a (5, 384) float32 array."""
    monkeypatch.setattr(batch_embed, "get_embedder", lambda: FakeEmbedder())

    result = batch_embed.embed_texts(["a", "b", "c", "d", "e"])

    assert_embedding_contract(result, expected_n=5)


def test_empty_input_has_expected_shape_and_dtype():
    """
    embed_texts([]) must return (0, 384) float32 without ever calling
    get_embedder() -- this does not load any model, real or fake.
    """
    result = batch_embed.embed_texts([])
    assert_embedding_contract(result, expected_n=0)


# ---------------------------------------------------------------------------
# Drift-detection checks: exercise the real embed_texts() code path
# ---------------------------------------------------------------------------


def test_dimension_drift_is_not_self_corrected_and_is_caught_by_contract(
    monkeypatch,
):
    """
    Simulates model/config drift where the embedder starts returning the
    wrong dimension (e.g. a swapped model producing 512-dim vectors
    instead of 384).

    embed_texts() does NOT correct this on its own -- it passes the wrong
    shape straight through. This test proves both facts:
      1. The actual embed_texts() output has the drifted shape (2, 512).
      2. assert_embedding_contract() catches this and fails loudly.
    """
    monkeypatch.setattr(
        batch_embed, "get_embedder", lambda: FakeEmbedder(dimension=512)
    )

    result = batch_embed.embed_texts(["a", "b"])

    assert result.shape == (2, 512)
    with pytest.raises(AssertionError, match="Expected embedding dimension"):
        assert_embedding_contract(result, expected_n=2)


def test_dtype_drift_is_self_corrected_by_embed_texts(monkeypatch):
    """
    Simulates an embedder returning float64 instead of float32.

    Unlike dimension drift, embed_texts() DOES normalize dtype: it wraps
    each batch with np.asarray(..., dtype=np.float32) and casts the final
    concatenated array with .astype(np.float32). So even if the underlying
    embedder returns float64, embed_texts()'s actual output is float32.

    This test verifies that real behavior (not an assumption) and confirms
    the contract check passes for it.
    """
    monkeypatch.setattr(
        batch_embed, "get_embedder", lambda: FakeEmbedder(dtype=np.float64)
    )

    result = batch_embed.embed_texts(["a", "b"])

    assert result.dtype == EXPECTED_DTYPE, (
        "embed_texts() is expected to normalize output dtype to float32 "
        "regardless of the underlying embedder's native dtype"
    )
    assert_embedding_contract(result, expected_n=2)