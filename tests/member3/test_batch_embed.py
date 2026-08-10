"""
Unit tests for EMB-003 batch embedding generation.

These tests use a lightweight fake embedder instead of the real
SentenceTransformer model so that batching, ordering, preprocessing,
and validation can be tested quickly without downloading or loading
the real model repeatedly.
"""

import numpy as np
import pytest

from src.embeddings import batch_embed


class FakeEmbedder:
    """
    Lightweight stand-in for the SentenceTransformer model.

    Each input text produces a deterministic 384-dimensional vector.
    The numeric value is extracted from the end of the text.

    Example:
        "item7" -> vector containing 7.0
    """

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.received_batches: list[list[str]] = []

    def encode(self, batch: list[str]) -> np.ndarray:
        """
        Encode a batch into deterministic 384-dimensional vectors.
        """
        self.calls.append(len(batch))
        self.received_batches.append(list(batch))

        vectors = []

        for index, _text in enumerate(batch):
            # Use the global position encoded by the test input.
            # The preprocessing test uses different text and only
            # checks received_batches, so this behavior is sufficient.
            value = float(
                sum(ord(char) for char in _text) % 100
            )

            vectors.append(
                np.full(
                    (384,),
                    value,
                    dtype=np.float32,
                )
            )

        return np.asarray(vectors, dtype=np.float32)


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    """
    Replace batch_embed.get_embedder() with a fake model.
    """
    fake = FakeEmbedder()

    monkeypatch.setattr(
        batch_embed,
        "get_embedder",
        lambda: fake,
    )

    return fake


def test_correct_number_of_vectors(
    fake_embedder: FakeEmbedder,
) -> None:
    """One output vector must be produced for every input text."""
    texts = ["item0", "item1", "item2"]

    result = batch_embed.embed_texts(texts)

    assert result.shape[0] == len(texts)


def test_correct_embedding_dimension(
    fake_embedder: FakeEmbedder,
) -> None:
    """Every embedding must contain exactly 384 dimensions."""
    texts = ["item0", "item1", "item2"]

    result = batch_embed.embed_texts(texts)

    assert result.shape[1] == 384


def test_correct_complete_shape(
    fake_embedder: FakeEmbedder,
) -> None:
    """Three inputs must produce an array of shape (3, 384)."""
    texts = ["item0", "item1", "item2"]

    result = batch_embed.embed_texts(texts)

    assert result.shape == (3, 384)


def test_output_is_numpy_float32_array(
    fake_embedder: FakeEmbedder,
) -> None:
    """Embeddings must be returned as a float32 NumPy array."""
    result = batch_embed.embed_texts(
        ["item0", "item1"],
    )

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32


def test_empty_input_returns_empty_array_without_calling_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Empty input must return (0, 384) without calling get_embedder().
    """
    call_count = {"count": 0}

    def fail_if_called() -> None:
        call_count["count"] += 1
        raise AssertionError(
            "get_embedder() must not be called for empty input"
        )

    monkeypatch.setattr(
        batch_embed,
        "get_embedder",
        fail_if_called,
    )

    result = batch_embed.embed_texts([])

    assert result.shape == (0, 384)
    assert result.dtype == np.float32
    assert call_count["count"] == 0


def test_input_ordering_is_preserved(
    fake_embedder: FakeEmbedder,
) -> None:
    """
    The order of vectors must match the order of the input texts.
    """
    texts = [
        "item0",
        "item1",
        "item2",
        "item3",
    ]

    result = batch_embed.embed_texts(
        texts,
        batch_size=2,
    )

    # The fake embedder creates a deterministic value based on each
    # individual input text. Recreate the expected values in order.
    expected_values = [
        float(sum(ord(char) for char in text) % 100)
        for text in texts
    ]

    for row, expected in zip(result, expected_values):
        assert np.all(row == expected)


def test_batch_size_is_respected(
    fake_embedder: FakeEmbedder,
) -> None:
    """
    Five inputs with batch_size=2 must be encoded as:
        2, 2, 1
    """
    texts = [
        "item0",
        "item1",
        "item2",
        "item3",
        "item4",
    ]

    batch_embed.embed_texts(
        texts,
        batch_size=2,
    )

    assert fake_embedder.calls == [2, 2, 1]


def test_preprocessing_is_applied_before_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that preprocess_for_embedding() is called before
    the text reaches the embedding model.
    """
    fake = FakeEmbedder()

    monkeypatch.setattr(
        batch_embed,
        "get_embedder",
        lambda: fake,
    )

    def fake_preprocess(text: str) -> str:
        return f"normalized-{text.strip().lower()}"

    monkeypatch.setattr(
        batch_embed,
        "preprocess_for_embedding",
        fake_preprocess,
    )

    batch_embed.embed_texts(
        ["  Hello World  "],
    )

    assert fake.received_batches == [
        ["normalized-hello world"]
    ]


def test_uses_get_embedder_to_obtain_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that EMB-003 obtains its model through get_embedder().
    """
    fake = FakeEmbedder()
    calls = {"count": 0}

    def tracking_get_embedder() -> FakeEmbedder:
        calls["count"] += 1
        return fake

    monkeypatch.setattr(
        batch_embed,
        "get_embedder",
        tracking_get_embedder,
    )

    batch_embed.embed_texts(
        ["item0", "item1"],
    )

    assert calls["count"] == 1


def test_does_not_import_sentence_transformer_directly() -> None:
    """
    EMB-003 must not expose or directly import SentenceTransformer.
    """
    assert not hasattr(
        batch_embed,
        "SentenceTransformer",
    )


def test_zero_batch_size_raises_value_error(
    fake_embedder: FakeEmbedder,
) -> None:
    """batch_size=0 must be rejected."""
    with pytest.raises(ValueError):
        batch_embed.embed_texts(
            ["item0"],
            batch_size=0,
        )


def test_negative_batch_size_raises_value_error(
    fake_embedder: FakeEmbedder,
) -> None:
    """Negative batch sizes must be rejected."""
    with pytest.raises(ValueError):
        batch_embed.embed_texts(
            ["item0"],
            batch_size=-1,
        )


def test_non_integer_batch_size_raises_value_error(
    fake_embedder: FakeEmbedder,
) -> None:
    """Non-integer batch sizes must be rejected."""
    with pytest.raises(ValueError):
        batch_embed.embed_texts(
            ["item0"],
            batch_size=2.5,  # type: ignore[arg-type]
        )


def test_repeated_calls_are_deterministic(
    fake_embedder: FakeEmbedder,
) -> None:
    """
    Repeated calls with identical inputs must return identical
    embeddings.
    """
    texts = [
        "item0",
        "item1",
        "item2",
    ]

    result_1 = batch_embed.embed_texts(texts)
    result_2 = batch_embed.embed_texts(texts)

    assert np.array_equal(
        result_1,
        result_2,
    )