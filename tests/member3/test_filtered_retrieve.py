"""
Unit tests for VDB-010 — Metadata Filtering on Retrieval.

Domain values in the PECS schema: "CODE", "EDGE"
Level values in the PECS schema:  "weak", "developing", "strong"
"""

from __future__ import annotations

from typing import Any, Dict
import pytest
import chromadb
import numpy as np

from src.embeddings import embedder


# ---------------------------------------------------------------------------
# Deterministic embeddings
# ---------------------------------------------------------------------------

def _make_embedding(seed: int) -> list[float]:
    """Deterministic 384-dimensional unit-normalised embedding."""
    rng = np.random.default_rng(seed)
    vec = rng.random(384).astype(np.float64)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


# Four embeddings from very different seeds — distances are unambiguous
EMB_A = _make_embedding(1)         # CODE / weak
EMB_B = _make_embedding(999)       # EDGE / weak
EMB_C = _make_embedding(77777)     # CODE / strong
EMB_D = _make_embedding(123456)    # EDGE / strong

META_A: Dict[str, Any] = {
    "domain": "CODE", "level": "weak",
    "c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5,
    "source": "train", "version": "pecs-v1.0",
}
META_B: Dict[str, Any] = {
    "domain": "EDGE", "level": "weak",
    "c1": 2, "c2": 3, "c3": 4, "c4": 5, "c5": 6, "c6": 7,
    "source": "validation", "version": "pecs-v1.0",
}
META_C: Dict[str, Any] = {
    "domain": "CODE", "level": "strong",
    "c1": 5, "c2": 5, "c3": 5, "c4": 5, "c5": 5, "c6": 5,
    "source": "train", "version": "pecs-v2.0",
}
META_D: Dict[str, Any] = {
    "domain": "EDGE", "level": "strong",
    "c1": 3, "c2": 4, "c3": 5, "c4": 6, "c5": 7, "c6": 8,
    "source": "validation", "version": "pecs-v2.0",
}

DOC_A = '{"scores": {"c1": 8}, "justification": "Record A - CODE weak"}'
DOC_B = '{"scores": {"c1": 2}, "justification": "Record B - EDGE weak"}'
DOC_C = '{"scores": {"c1": 5}, "justification": "Record C - CODE strong"}'
DOC_D = '{"scores": {"c1": 3}, "justification": "Record D - EDGE strong"}'


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def col() -> chromadb.Collection:
    """Isolated in-memory collection seeded with 4 records across 2 domains × 2 levels.

    Drops any existing ``prompt_examples`` collection first: ephemeral ChromaDB
    clients share one in-process database, so this module and
    ``test_retrieval.py`` were overwriting each other's ``record-A/B/C`` vectors
    and failing whichever ran second (Member 6, TEST-003).
    """
    from src.vectorstore import init, upsert
    client = chromadb.EphemeralClient()

    if init.COLLECTION_NAME in {c.name for c in client.list_collections()}:
        client.delete_collection(name=init.COLLECTION_NAME)

    collection = init.get_or_create_collection(client)

    upsert.upsert_record(collection, "record-A", EMB_A, DOC_A, META_A)  # CODE / weak
    upsert.upsert_record(collection, "record-B", EMB_B, DOC_B, META_B)  # EDGE / weak
    upsert.upsert_record(collection, "record-C", EMB_C, DOC_C, META_C)  # CODE / strong
    upsert.upsert_record(collection, "record-D", EMB_D, DOC_D, META_D)  # EDGE / strong

    yield collection

    client.delete_collection(name=init.COLLECTION_NAME)


# ---------------------------------------------------------------------------
# 1. Unfiltered retrieval behaves like VDB-009
# ---------------------------------------------------------------------------

def test_unfiltered_returns_own_record(col: chromadb.Collection) -> None:
    """With no filter, querying EMB_A returns record-A as the top result."""
    from src.vectorstore.filtered_retrieve import filtered_retrieve

    result = filtered_retrieve(EMB_A, n_results=1, collection=col)
    assert result["ids"][0][0] == "record-A"


def test_unfiltered_n_results_4(col: chromadb.Collection) -> None:
    """Unfiltered n_results=4 returns all four records."""
    from src.vectorstore.filtered_retrieve import filtered_retrieve

    result = filtered_retrieve(EMB_A, n_results=4, collection=col)
    assert len(result["ids"][0]) == 4


# ---------------------------------------------------------------------------
# 2. Domain filtering
# ---------------------------------------------------------------------------

def test_domain_code_never_returns_edge(col: chromadb.Collection) -> None:
    """domain='CODE' must never return an EDGE record."""
    from src.vectorstore.filtered_retrieve import filtered_retrieve

    result = filtered_retrieve(EMB_A, n_results=2, domain="CODE", collection=col)
    ids = result["ids"][0]
    metas = result["metadatas"][0]

    assert len(ids) == 2
    for meta in metas:
        assert meta["domain"] == "CODE", f"Expected CODE, got {meta['domain']}"


def test_domain_edge_never_returns_code(col: chromadb.Collection) -> None:
    """domain='EDGE' must never return a CODE record."""
    from src.vectorstore.filtered_retrieve import filtered_retrieve

    result = filtered_retrieve(EMB_B, n_results=2, domain="EDGE", collection=col)
    ids = result["ids"][0]
    metas = result["metadatas"][0]

    assert len(ids) == 2
    for meta in metas:
        assert meta["domain"] == "EDGE", f"Expected EDGE, got {meta['domain']}"


def test_domain_filter_globally_nearest_wrong_domain(col: chromadb.Collection) -> None:
    """
    Critical test: EMB_A is closest to record-A (CODE/weak).
    When we query with domain='EDGE', record-A must NOT appear;
    the nearest eligible EDGE record must be returned instead.
    """
    from src.vectorstore.filtered_retrieve import filtered_retrieve

    result = filtered_retrieve(EMB_A, n_results=1, domain="EDGE", collection=col)
    top_id = result["ids"][0][0]
    top_meta = result["metadatas"][0][0]

    assert top_id != "record-A", "record-A (CODE) must not appear under domain='EDGE'"
    assert top_meta["domain"] == "EDGE"


# ---------------------------------------------------------------------------
# 3. Level filtering
# ---------------------------------------------------------------------------

def test_level_weak_never_returns_strong(col: chromadb.Collection) -> None:
    """level='weak' must never return a strong or developing record."""
    from src.vectorstore.filtered_retrieve import filtered_retrieve

    result = filtered_retrieve(EMB_A, n_results=2, level="weak", collection=col)
    metas = result["metadatas"][0]

    assert len(metas) == 2
    for meta in metas:
        assert meta["level"] == "weak", f"Expected weak, got {meta['level']}"


def test_level_strong_never_returns_weak(col: chromadb.Collection) -> None:
    """level='strong' must never return a weak record."""
    from src.vectorstore.filtered_retrieve import filtered_retrieve

    result = filtered_retrieve(EMB_C, n_results=2, level="strong", collection=col)
    metas = result["metadatas"][0]

    assert len(metas) == 2
    for meta in metas:
        assert meta["level"] == "strong", f"Expected strong, got {meta['level']}"


# ---------------------------------------------------------------------------
# 4. Combined domain + level filtering
# ---------------------------------------------------------------------------

def test_combined_domain_and_level_both_enforced(col: chromadb.Collection) -> None:
    """domain='CODE' + level='weak' must return only record-A."""
    from src.vectorstore.filtered_retrieve import filtered_retrieve

    result = filtered_retrieve(EMB_A, n_results=1, domain="CODE", level="weak", collection=col)
    ids = result["ids"][0]
    metas = result["metadatas"][0]

    assert len(ids) == 1
    assert ids[0] == "record-A"
    assert metas[0]["domain"] == "CODE"
    assert metas[0]["level"] == "weak"


def test_combined_edge_strong_returns_only_record_d(col: chromadb.Collection) -> None:
    """domain='EDGE' + level='strong' must return only record-D."""
    from src.vectorstore.filtered_retrieve import filtered_retrieve

    result = filtered_retrieve(EMB_D, n_results=1, domain="EDGE", level="strong", collection=col)
    ids = result["ids"][0]
    metas = result["metadatas"][0]

    assert ids[0] == "record-D"
    assert metas[0]["domain"] == "EDGE"
    assert metas[0]["level"] == "strong"


# ---------------------------------------------------------------------------
# 5. n_results respects the filter for every result
# ---------------------------------------------------------------------------

def test_n_results_2_domain_filter_all_match(col: chromadb.Collection) -> None:
    """n_results=2 with domain='CODE' must return exactly 2 CODE records."""
    from src.vectorstore.filtered_retrieve import filtered_retrieve

    result = filtered_retrieve(EMB_A, n_results=2, domain="CODE", collection=col)
    metas = result["metadatas"][0]

    assert len(metas) == 2
    for meta in metas:
        assert meta["domain"] == "CODE"


# ---------------------------------------------------------------------------
# 6. Empty result when no records match
# ---------------------------------------------------------------------------

def test_no_matching_domain_returns_empty_safely(col: chromadb.Collection) -> None:
    """
    A filter that matches no records should return an empty result without raising.
    Since 'developing' level has no records, that combination returns empty.
    """
    from src.vectorstore.filtered_retrieve import filtered_retrieve

    # 'developing' level has no records seeded
    result = filtered_retrieve(EMB_A, n_results=1, level="developing", collection=col)
    assert result["ids"][0] == []


def test_combined_no_match_returns_empty_safely(col: chromadb.Collection) -> None:
    """domain='CODE' + level='developing' has no records — must return empty safely."""
    from src.vectorstore.filtered_retrieve import filtered_retrieve

    result = filtered_retrieve(EMB_A, n_results=1, domain="CODE", level="developing", collection=col)
    assert result["ids"][0] == []


# ---------------------------------------------------------------------------
# 7. Metadata and distances preserved in results
# ---------------------------------------------------------------------------

def test_metadata_fully_preserved(col: chromadb.Collection) -> None:
    """Metadata returned for a filtered result must exactly match stored metadata."""
    from src.vectorstore.filtered_retrieve import filtered_retrieve

    result = filtered_retrieve(EMB_A, n_results=1, domain="CODE", level="weak", collection=col)
    assert result["metadatas"][0][0] == META_A


def test_distances_present_and_non_negative(col: chromadb.Collection) -> None:
    """Distances must be present and non-negative."""
    from src.vectorstore.filtered_retrieve import filtered_retrieve

    result = filtered_retrieve(EMB_A, n_results=1, domain="CODE", collection=col)
    assert "distances" in result
    assert result["distances"] is not None
    dist = result["distances"][0][0]
    assert isinstance(dist, float)
    assert dist >= 0.0


# ---------------------------------------------------------------------------
# 8. Input validation
# ---------------------------------------------------------------------------

def test_invalid_domain_raises_value_error(col: chromadb.Collection) -> None:
    from src.vectorstore.filtered_retrieve import filtered_retrieve
    with pytest.raises(ValueError, match="domain"):
        filtered_retrieve(EMB_A, domain="INVALID_DOMAIN", collection=col)


def test_invalid_level_raises_value_error(col: chromadb.Collection) -> None:
    from src.vectorstore.filtered_retrieve import filtered_retrieve
    with pytest.raises(ValueError, match="level"):
        filtered_retrieve(EMB_A, level="L9000", collection=col)


def test_n_results_zero_raises(col: chromadb.Collection) -> None:
    from src.vectorstore.filtered_retrieve import filtered_retrieve
    with pytest.raises(ValueError):
        filtered_retrieve(EMB_A, n_results=0, collection=col)


def test_n_results_negative_raises(col: chromadb.Collection) -> None:
    from src.vectorstore.filtered_retrieve import filtered_retrieve
    with pytest.raises(ValueError):
        filtered_retrieve(EMB_A, n_results=-3, collection=col)


def test_n_results_float_raises(col: chromadb.Collection) -> None:
    from src.vectorstore.filtered_retrieve import filtered_retrieve
    with pytest.raises(TypeError):
        filtered_retrieve(EMB_A, n_results=2.5, collection=col)  # type: ignore


def test_empty_embedding_raises(col: chromadb.Collection) -> None:
    from src.vectorstore.filtered_retrieve import filtered_retrieve
    with pytest.raises(ValueError):
        filtered_retrieve([], collection=col)


def test_none_embedding_raises(col: chromadb.Collection) -> None:
    from src.vectorstore.filtered_retrieve import filtered_retrieve
    with pytest.raises(ValueError):
        filtered_retrieve(None, collection=col)  # type: ignore


def test_non_numeric_embedding_raises(col: chromadb.Collection) -> None:
    from src.vectorstore.filtered_retrieve import filtered_retrieve
    with pytest.raises(TypeError):
        filtered_retrieve(["a"] * 384, collection=col)  # type: ignore


def test_wrong_dimension_raises(col: chromadb.Collection) -> None:
    from src.vectorstore.filtered_retrieve import filtered_retrieve
    with pytest.raises(ValueError, match="384"):
        filtered_retrieve([0.1] * 100, collection=col)


# ---------------------------------------------------------------------------
# 9. Architecture boundary tests
# ---------------------------------------------------------------------------

def test_no_persistent_client_created(
    col: chromadb.Collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """filtered_retrieve() must not instantiate PersistentClient when collection is supplied."""
    from src.vectorstore.filtered_retrieve import filtered_retrieve

    created = False

    def mock_pc(*args: Any, **kwargs: Any) -> Any:
        nonlocal created
        created = True
        return chromadb.EphemeralClient()

    monkeypatch.setattr(chromadb, "PersistentClient", mock_pc)
    filtered_retrieve(EMB_A, domain="CODE", collection=col)
    assert not created, "filtered_retrieve() must not instantiate PersistentClient"


def test_no_sentence_transformer_loaded(
    col: chromadb.Collection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """filtered_retrieve() must not load SentenceTransformer."""
    from src.vectorstore.filtered_retrieve import filtered_retrieve

    loaded = False

    def mock_st(*args: Any, **kwargs: Any) -> Any:
        nonlocal loaded
        loaded = True
        return embedder.get_embedder()

    monkeypatch.setattr(embedder, "SentenceTransformer", mock_st)
    filtered_retrieve(EMB_A, domain="CODE", collection=col)
    assert not loaded, "filtered_retrieve() must not load SentenceTransformer"


def test_production_db_never_modified(col: chromadb.Collection) -> None:
    """Collection used is the isolated EphemeralClient fixture, never production."""
    from src.vectorstore.filtered_retrieve import filtered_retrieve

    result = filtered_retrieve(EMB_A, n_results=1, domain="CODE", collection=col)
    assert result["ids"][0][0] == "record-A"
