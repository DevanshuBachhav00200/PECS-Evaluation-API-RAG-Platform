"""
EMB-006 — Tests for embedding_serializer.py.

Verifies the serialize -> save -> load -> ingestion round trip required
by the EMB-006 acceptance criterion: "Serialized output loads back
without transformation into the ingestion function."

Design note on TEST 6 ("count mismatch"):
    The tracker's TEST 6 describes IDs and embeddings of different
    lengths failing clearly. This module uses EmbeddingRecord, where one
    object always pairs exactly one id with exactly one embedding, so
    that specific misalignment is structurally impossible. The closest
    real analog -- a batch containing a duplicate id, which would
    corrupt a ChromaDB upsert the same way a length mismatch would --
    is tested instead, in test_duplicate_id_raises_clear_error().
"""

import json

import numpy as np
import pytest

from src.embeddings.embedding_serializer import (
    EXPECTED_DIMENSION,
    EmbeddingRecord,
    load_serialized,
    save_serialized,
    serialize_embeddings,
)


def make_record(record_id, fill_value, metadata=None, document=None,
                 dimension=EXPECTED_DIMENSION):
    """Build a small deterministic EmbeddingRecord for tests."""
    vector = np.full((dimension,), fill_value, dtype=np.float32)
    return EmbeddingRecord(
        id=record_id,
        embedding=vector,
        metadata=metadata if metadata is not None else {},
        document=document,
    )


# ---------------------------------------------------------------------------
# TEST 1 — Basic serialization
# ---------------------------------------------------------------------------


def test_basic_serialization_contains_expected_keys():
    records = [
        make_record("id-1", 0.1, metadata={"domain": "CREATE"}, document="text A"),
        make_record("id-2", 0.2, metadata={"domain": "CODE"}, document="text B"),
    ]

    serialized = serialize_embeddings(records)

    assert set(serialized.keys()) == {"ids", "embeddings", "metadatas", "documents"}
    assert serialized["ids"] == ["id-1", "id-2"]
    assert len(serialized["embeddings"]) == 2
    assert len(serialized["metadatas"]) == 2
    assert len(serialized["documents"]) == 2


# ---------------------------------------------------------------------------
# TEST 2 — Persistence round trip
# ---------------------------------------------------------------------------


def test_persistence_round_trip(tmp_path):
    records = [make_record("id-1", 0.5, metadata={"domain": "CREATE"})]
    serialized = serialize_embeddings(records)

    file_path = tmp_path / "embeddings.json"
    save_serialized(serialized, file_path)

    assert file_path.exists()
    loaded = load_serialized(file_path)

    assert loaded == serialized


# ---------------------------------------------------------------------------
# TEST 3 — Embedding preservation (all 384 dimensions)
# ---------------------------------------------------------------------------


def test_embedding_values_preserved_after_round_trip(tmp_path):
    original_vector = np.linspace(-1.0, 1.0, EXPECTED_DIMENSION, dtype=np.float32)
    records = [EmbeddingRecord(id="id-1", embedding=original_vector, metadata={})]

    serialized = serialize_embeddings(records)
    file_path = tmp_path / "embeddings.json"
    save_serialized(serialized, file_path)
    loaded = load_serialized(file_path)

    reloaded_vector = np.array(loaded["embeddings"][0], dtype=np.float32)

    assert reloaded_vector.shape == (EXPECTED_DIMENSION,)
    np.testing.assert_allclose(
        reloaded_vector, original_vector, rtol=1e-6, atol=1e-6,
        err_msg="Embedding values drifted during serialize/save/load round trip",
    )


# ---------------------------------------------------------------------------
# TEST 4 — ID preservation and alignment
# ---------------------------------------------------------------------------


def test_ids_preserved_and_aligned_with_embeddings(tmp_path):
    records = [
        make_record("alpha", 1.0),
        make_record("beta", 2.0),
        make_record("gamma", 3.0),
    ]
    serialized = serialize_embeddings(records)
    file_path = tmp_path / "embeddings.json"
    save_serialized(serialized, file_path)
    loaded = load_serialized(file_path)

    assert loaded["ids"] == ["alpha", "beta", "gamma"]
    # Confirm alignment: id at index i corresponds to embedding at index i.
    for i, expected_fill in enumerate([1.0, 2.0, 3.0]):
        vector = np.array(loaded["embeddings"][i])
        assert np.all(vector == expected_fill), (
            f"embedding at index {i} does not match its id {loaded['ids'][i]!r}"
        )


# ---------------------------------------------------------------------------
# TEST 5 — Metadata preservation
# ---------------------------------------------------------------------------


def test_metadata_preserved_without_loss(tmp_path):
    metadata = {"domain": "CREATE", "c1": 9.5, "c2": 7.0, "review_status": "approved"}
    records = [make_record("id-1", 0.3, metadata=metadata)]

    serialized = serialize_embeddings(records)
    file_path = tmp_path / "embeddings.json"
    save_serialized(serialized, file_path)
    loaded = load_serialized(file_path)

    assert loaded["metadatas"][0] == metadata


# ---------------------------------------------------------------------------
# TEST 6 — Duplicate ID (structural analog of "count mismatch")
# ---------------------------------------------------------------------------


def test_duplicate_id_raises_clear_error():
    """
    EmbeddingRecord pairs one id with one embedding by construction, so a
    raw ids/embeddings length mismatch cannot occur here. The closest
    real risk -- two records sharing the same id, which would silently
    overwrite each other on ChromaDB upsert -- is what this test guards.
    """
    records = [
        make_record("duplicate-id", 0.1),
        make_record("duplicate-id", 0.2),
    ]

    with pytest.raises(ValueError, match="duplicate id"):
        serialize_embeddings(records)


# ---------------------------------------------------------------------------
# TEST 7 — Wrong dimensionality
# ---------------------------------------------------------------------------


def test_wrong_dimensionality_raises_clear_error():
    records = [make_record("id-1", 0.1, dimension=512)]

    with pytest.raises(ValueError, match="expected embedding dimension"):
        serialize_embeddings(records)


# ---------------------------------------------------------------------------
# TEST 8 — Empty input
# ---------------------------------------------------------------------------


def test_empty_input_produces_empty_structure(tmp_path):
    serialized = serialize_embeddings([])
    assert serialized == {"ids": [], "embeddings": [], "metadatas": [], "documents": []}

    file_path = tmp_path / "empty.json"
    save_serialized(serialized, file_path)
    loaded = load_serialized(file_path)

    assert loaded == {"ids": [], "embeddings": [], "metadatas": [], "documents": []}


# ---------------------------------------------------------------------------
# TEST 9 — Ingestion compatibility (the core EMB-006 acceptance criterion)
# ---------------------------------------------------------------------------


def fake_chromadb_upsert(ids, embeddings, metadatas, documents):
    """
    Stand-in for the future ChromaDB upsert wrapper (VDB-xxx). Mirrors
    the conceptual signature:

        collection.upsert(ids=..., embeddings=..., metadatas=..., documents=...)

    Returns what it received so the test can assert nothing was altered.
    """
    return {
        "ids": ids,
        "embeddings": embeddings,
        "metadatas": metadatas,
        "documents": documents,
    }


def test_loaded_output_is_directly_usable_by_ingestion_function_without_transformation(
    tmp_path,
):
    """
    This is the direct demonstration of the EMB-006 acceptance criterion:
    "Serialized output loads back without transformation into the
    ingestion function."

    No reshaping, renaming, or type conversion happens between
    load_serialized() and the mock ingestion call below.
    """
    records = [
        make_record("id-1", 0.1, metadata={"domain": "CREATE"}, document="prompt A"),
        make_record("id-2", 0.2, metadata={"domain": "CODE"}, document="prompt B"),
    ]
    serialized = serialize_embeddings(records)

    file_path = tmp_path / "embeddings.json"
    save_serialized(serialized, file_path)
    loaded = load_serialized(file_path)

    # The critical line: loaded is passed directly, unpacked, with zero
    # intermediate conversion steps.
    result = fake_chromadb_upsert(**loaded)

    assert result["ids"] == ["id-1", "id-2"]
    assert len(result["embeddings"]) == 2
    assert all(len(vec) == EXPECTED_DIMENSION for vec in result["embeddings"])
    assert result["metadatas"] == [{"domain": "CREATE"}, {"domain": "CODE"}]
    assert result["documents"] == ["prompt A", "prompt B"]