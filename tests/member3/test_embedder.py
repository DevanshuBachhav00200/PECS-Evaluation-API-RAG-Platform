"""Focused tests for EMB-001 warm-loaded embedder."""

from sentence_transformers import SentenceTransformer

from src.embeddings.embedder import (
    MODEL_NAME,
    get_embedder,
    get_model_name,
)


def test_embedder_module_imports_cleanly():
    """The embedder module should import without raising."""
    assert MODEL_NAME == "all-MiniLM-L6-v2"


def test_get_model_name_matches_model_name():
    """get_model_name() should return the configured model name."""
    assert get_model_name() == MODEL_NAME


def test_get_embedder_returns_real_sentence_transformer_instance():
    """get_embedder() should return a real SentenceTransformer instance."""
    model = get_embedder()

    assert model is not None
    assert isinstance(model, SentenceTransformer)


def test_repeated_access_returns_same_instance():
    """Repeated get_embedder() calls should return the same object."""
    model_a = get_embedder()
    model_b = get_embedder()
    model_c = get_embedder()

    assert model_a is model_b
    assert model_b is model_c