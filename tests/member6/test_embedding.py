"""
TEST-002 — Embedding-layer unit tests.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Covers EMB-004 (dimensionality and dtype) and EMB-005 (determinism), which are
the two properties the rest of the stack silently assumes. A drifted model
dimension does not raise anywhere on its own — it surfaces as a ChromaDB
rejection three layers away, or worse, as retrieval that returns plausible
nonsense.

Acceptance (TEST-002): tests pass consistently across repeated runs. The
determinism cases run the encoder twice inside a single test for exactly that
reason; a property that only holds on the first call is not determinism.

These tests load the real sentence-transformer (EMB-001's warm-loaded instance),
so they need the model cache present. They skip rather than fail when
``sentence-transformers`` is unavailable, so the rest of Member 6's suite still
runs in a minimal container.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "sentence_transformers",
    reason="sentence-transformers is not installed; embedding-layer tests skipped",
)

import numpy as np  # noqa: E402

from src.embeddings.batch_embed import embed_texts  # noqa: E402
from src.embeddings.embed_error_handling import validate_embedding_input  # noqa: E402
from src.embeddings.embed_preprocess import preprocess_for_embedding  # noqa: E402
from src.embeddings.embedder import MODEL_NAME, get_embedder  # noqa: E402
from src.rag.config import EMBEDDING_DIM  # noqa: E402

SAMPLE = "Review this endpoint and list issues by severity."


# ---------------------------------------------------------------------------
# EMB-001 — warm load
# ---------------------------------------------------------------------------


def test_embedder_is_a_single_warm_loaded_instance() -> None:
    """Repeated calls return the identical object, not a reloaded model."""
    assert get_embedder() is get_embedder()


def test_embedding_model_is_the_documented_one() -> None:
    """The model string is pinned; a change here changes every stored vector."""
    assert MODEL_NAME == "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# EMB-004 — dimensionality and dtype
# ---------------------------------------------------------------------------


def test_vector_has_the_declared_dimension() -> None:
    vectors = embed_texts([SAMPLE])
    assert vectors.shape == (1, EMBEDDING_DIM)


def test_vector_dtype_is_float32() -> None:
    """float32 is what ChromaDB stores; float64 doubles the store for nothing."""
    vectors = embed_texts([SAMPLE])
    assert vectors.dtype == np.float32


def test_batch_output_count_equals_input_count() -> None:
    """EMB-003's acceptance criterion: one vector out per record in."""
    texts = [SAMPLE, "Write a haiku about winter.", "Explain recursion simply."]
    assert embed_texts(texts).shape[0] == len(texts)


# ---------------------------------------------------------------------------
# EMB-005 — determinism
# ---------------------------------------------------------------------------


def test_identical_text_produces_identical_vectors() -> None:
    """Cosine similarity between two runs on identical input is 1.0."""
    first = embed_texts([SAMPLE])[0]
    second = embed_texts([SAMPLE])[0]

    cosine = float(
        np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
    )
    assert cosine == pytest.approx(1.0, abs=1e-6)


def test_determinism_holds_across_repeated_batches() -> None:
    """The same text embeds identically whether alone or inside a batch.

    Batch position must not perturb a vector: ingestion embeds in batches and
    query time embeds one prompt at a time (RAG-002), so a batch-dependent
    vector would silently degrade every retrieval.
    """
    alone = embed_texts([SAMPLE])[0]
    in_batch = embed_texts(["unrelated filler text", SAMPLE, "more filler"])[1]

    assert np.allclose(alone, in_batch, atol=1e-6)


def test_different_text_produces_different_vectors() -> None:
    """The control case: determinism must not mean 'always the same vector'."""
    first = embed_texts([SAMPLE])[0]
    second = embed_texts(["Write a haiku about winter."])[0]
    assert not np.allclose(first, second, atol=1e-3)


# ---------------------------------------------------------------------------
# EMB-002 / EMB-007 — preprocessing parity and error handling
# ---------------------------------------------------------------------------


def test_preprocessing_is_idempotent() -> None:
    once = preprocess_for_embedding("  Review   THIS endpoint  ")
    assert preprocess_for_embedding(once) == once


def test_query_time_and_ingestion_time_preprocessing_agree() -> None:
    """RAG-001's parity requirement, asserted from the Member 6 side."""
    from src.rag.query_preprocess import preprocess_query

    raw = "  Review   THIS Endpoint\nplease "
    assert preprocess_query(raw) == preprocess_for_embedding(raw)


def test_empty_input_returns_a_structured_error_not_a_crash() -> None:
    result = validate_embedding_input("")
    assert not result.is_valid
    assert result.error_code


def test_whitespace_only_input_is_rejected() -> None:
    assert not validate_embedding_input("   \n\t ").is_valid


def test_valid_input_passes_validation() -> None:
    assert validate_embedding_input(SAMPLE).is_valid
