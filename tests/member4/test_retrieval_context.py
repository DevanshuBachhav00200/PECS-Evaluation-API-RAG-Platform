"""
Tests for RAG-001 … RAG-008: query preprocessing, query embedding, retrieval,
Top-K configurability, metadata filtering, empty-retrieval handling, future
ranking, and the compact calibration formatter.
"""

from __future__ import annotations

import pytest

from src.common.normalization import normalize_text
from src.embeddings.embed_preprocess import preprocess_for_embedding
from src.rag import config
from src.rag.context_ranking import rank_candidates, rank_score
from src.rag.empty_retrieval_handler import (
    NO_CALIBRATION_TEXT,
    REASON_EMPTY_COLLECTION,
    REASON_NO_FILTER_MATCH,
    is_empty,
    no_calibration_result,
)
from src.rag.filtered_context import retrieve_filtered_context
from src.rag.format_calibration import (
    format_calibration,
    format_calibration_from_document,
)
from src.rag.query_embed import embed_query
from src.rag.query_preprocess import preprocess_query
from src.rag.retrieve_context import retrieve_context

from m4_fakes import (  # noqa: E402  (pytest inserts the test dir on sys.path)
    FakeCollection,
    FakeEmbedder,
    calibration_document,
    calibration_metadata,
    empty_query_result,
    query_result,
)


# --------------------------------------------------------------------------- #
# RAG-001 — query-time normalization parity with ingestion                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw",
    [
        "  Write   a REST review PROMPT ",
        "Tabs\tand\nnewlines   collapse",
        "MiXeD CaSe Input",
    ],
)
def test_query_preprocess_matches_ingestion(raw: str) -> None:
    """RAG-001 acceptance: query-time and ingestion-time output are identical."""
    assert preprocess_query(raw) == preprocess_for_embedding(raw)
    assert preprocess_query(raw) == normalize_text(raw)


def test_query_preprocess_is_idempotent() -> None:
    once = preprocess_query("  Review   THIS  endpoint  ")
    assert preprocess_query(once) == once


def test_query_preprocess_rejects_empty() -> None:
    with pytest.raises(ValueError):
        preprocess_query("   ")
    with pytest.raises(TypeError):
        preprocess_query(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# RAG-002 — query embedding reuses the shared model instance                   #
# --------------------------------------------------------------------------- #


def test_embed_query_returns_384_floats(fake_embedder) -> None:
    vector = embed_query("Review this endpoint", embedder=fake_embedder)
    assert len(vector) == config.EMBEDDING_DIM
    assert all(isinstance(value, float) for value in vector)


def test_embed_query_normalizes_before_encoding(fake_embedder) -> None:
    embed_query("  Review   THIS Endpoint ", embedder=fake_embedder)
    assert fake_embedder.encoded == ["review this endpoint"]


def test_embed_query_does_not_reload_the_model(fake_embedder) -> None:
    """One embedder instance serves every call — no per-request model load."""
    embed_query("first prompt", embedder=fake_embedder)
    embed_query("second prompt", embedder=fake_embedder)
    assert len(fake_embedder.encoded) == 2


def test_embed_query_uses_the_shared_emb001_instance(monkeypatch) -> None:
    """RAG-002 acceptance: the query path uses the *same* warm-loaded model
    instance as batch ingestion — it never constructs its own.

    ``src.embeddings.embedder`` loads a SentenceTransformer at import time, so
    this test stubs the module rather than importing it: the assertion is that
    ``embed_query`` resolves its model through EMB-001's ``get_embedder()``.
    """
    import sys
    import types

    shared = FakeEmbedder()
    calls = {"count": 0}

    stub = types.ModuleType("src.embeddings.embedder")

    def get_embedder():  # noqa: ANN202
        calls["count"] += 1
        return shared

    stub.get_embedder = get_embedder
    monkeypatch.setitem(sys.modules, "src.embeddings.embedder", stub)

    from src.rag.query_embed import get_shared_embedder

    assert get_shared_embedder() is shared

    # No embedder argument: the pipeline must fall through to EMB-001's instance.
    embed_query("first prompt")
    embed_query("second prompt")

    assert shared.encoded == ["first prompt", "second prompt"]
    assert calls["count"] == 3  # resolved per call, but always the same object


def test_embed_query_rejects_wrong_dimensionality() -> None:
    class ShortEmbedder:
        def encode(self, text: str):  # noqa: ANN001, ANN201
            return [0.1, 0.2, 0.3]

    with pytest.raises(ValueError, match="384 dimensions"):
        embed_query("prompt", embedder=ShortEmbedder())


# --------------------------------------------------------------------------- #
# RAG-008 — compact calibration formatter                                      #
# --------------------------------------------------------------------------- #


def test_format_calibration_exact_format() -> None:
    """RAG-008 acceptance: output matches the documented compact format."""
    line = format_calibration(
        {"c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5},
        "Clear framing, weak verification.",
    )
    assert line == (
        "Scores: C1:8 | C2:7 | C3:9 | C4:6 | C5:10 | C6:5 "
        "→ Justification: Clear framing, weak verification."
    )


def test_format_calibration_accepts_suffixed_keys() -> None:
    suffixed = {f"c{i}_score": i for i in range(1, 7)}
    plain = {f"c{i}": i for i in range(1, 7)}
    assert format_calibration(suffixed, "x") == format_calibration(plain, "x")


def test_format_calibration_renders_integral_floats_without_decimal() -> None:
    line = format_calibration({f"c{i}": float(i) for i in range(1, 7)}, "x")
    assert "C1:1 |" in line
    assert "C1:1.0" not in line


def test_format_calibration_rejects_missing_criterion() -> None:
    with pytest.raises(ValueError, match="c5"):
        format_calibration({"c1": 1, "c2": 2, "c3": 3, "c4": 4, "c6": 6}, "x")


def test_format_calibration_from_document_falls_back_to_metadata() -> None:
    """A partial stored document recovers its scores from VDB-004 metadata."""
    partial = '{"scores": {"c1": 8}, "justification": "partial record"}'
    line = format_calibration_from_document(partial, calibration_metadata())
    assert line.startswith("Scores: C1:8 | C2:7")
    assert "partial record" in line


# --------------------------------------------------------------------------- #
# RAG-003 / RAG-004 — retrieve_context and Top-K configurability               #
# --------------------------------------------------------------------------- #


def test_retrieve_context_returns_one_formatted_example(
    populated_collection, fake_embedder
) -> None:
    """RAG-003 acceptance: one formatted calibration example for a known prompt."""
    result = retrieve_context(
        "Review this endpoint",
        collection=populated_collection,
        embedder=fake_embedder,
    )

    assert result["available"] is True
    assert result["record_id"] == "rec-001"
    assert result["calibration"].startswith("Scores: C1:8 |")
    assert "→ Justification:" in result["calibration"]
    assert len(result["examples"]) == 1


def test_retrieve_context_uses_configured_top_k(
    populated_collection, fake_embedder
) -> None:
    retrieve_context(
        "Review this endpoint",
        collection=populated_collection,
        embedder=fake_embedder,
    )
    assert populated_collection.calls[0]["n_results"] == config.DEFAULT_TOP_K


def test_retrieve_context_top_k_override_changes_request(
    fake_embedder, monkeypatch
) -> None:
    """RAG-004 acceptance: changing the parameter changes the number returned,
    with no code edit elsewhere."""
    three = query_result(
        ids=["a", "b", "c"],
        documents=[calibration_document()] * 3,
        metadatas=[calibration_metadata()] * 3,
        distances=[0.1, 0.2, 0.3],
    )
    collection = FakeCollection(three)

    explicit = retrieve_context(
        "prompt", n_results=3, collection=collection, embedder=fake_embedder
    )
    assert collection.calls[0]["n_results"] == 3
    assert len(explicit["examples"]) == 3

    monkeypatch.setattr(config, "DEFAULT_TOP_K", 3)
    monkeypatch.setattr("src.rag.retrieve_context.DEFAULT_TOP_K", 3)
    from_config = retrieve_context(
        "prompt", collection=collection, embedder=fake_embedder
    )
    assert collection.calls[1]["n_results"] == 3
    assert len(from_config["examples"]) == 3


def test_retrieve_context_rejects_non_positive_top_k(
    populated_collection, fake_embedder
) -> None:
    with pytest.raises(ValueError):
        retrieve_context(
            "prompt", n_results=0, collection=populated_collection, embedder=fake_embedder
        )


def test_retrieve_context_skips_malformed_records(fake_embedder) -> None:
    collection = FakeCollection(
        query_result(
            ids=["bad", "good"],
            documents=["not json at all", calibration_document()],
            metadatas=[calibration_metadata(), calibration_metadata()],
            distances=[0.05, 0.30],
        )
    )
    result = retrieve_context("prompt", n_results=2, collection=collection, embedder=fake_embedder)
    assert result["available"] is True
    assert result["record_id"] == "good"
    assert len(result["examples"]) == 1


# --------------------------------------------------------------------------- #
# RAG-006 — empty-retrieval handling                                           #
# --------------------------------------------------------------------------- #


def test_empty_collection_returns_structured_result(
    empty_collection, fake_embedder
) -> None:
    """RAG-006 acceptance: an empty collection returns a structured result,
    not a crash."""
    result = retrieve_context(
        "Review this endpoint", collection=empty_collection, embedder=fake_embedder
    )

    assert result["available"] is False
    assert result["calibration"] == NO_CALIBRATION_TEXT
    assert result["reason"] == REASON_EMPTY_COLLECTION
    assert result["record_id"] is None


@pytest.mark.parametrize(
    "payload",
    [None, {}, {"ids": None}, {"ids": []}, {"ids": [[]]}],
)
def test_is_empty_tolerates_every_miss_shape(payload) -> None:
    assert is_empty(payload) is True


def test_is_empty_false_for_a_hit() -> None:
    assert is_empty(query_result()) is False


def test_no_calibration_result_rejects_unknown_reason() -> None:
    with pytest.raises(ValueError):
        no_calibration_result(reason="because")


# --------------------------------------------------------------------------- #
# RAG-005 — metadata-aware retrieval filtering                                 #
# --------------------------------------------------------------------------- #


def test_filtered_context_applies_server_side_where_clause(fake_embedder) -> None:
    collection = FakeCollection(query_result())
    retrieve_filtered_context(
        "prompt",
        domain="CODE",
        level="strong",
        collection=collection,
        embedder=fake_embedder,
    )
    where = collection.calls[0]["where"]
    assert where == {
        "$and": [{"domain": {"$eq": "CODE"}}, {"level": {"$eq": "strong"}}]
    }


def test_filtered_context_never_returns_out_of_domain(fake_embedder) -> None:
    """RAG-005 acceptance: a domain-scoped request cannot retrieve out-of-domain.

    The filter is enforced by ChromaDB, so a zero-match filter must degrade to
    'no calibration available' rather than falling back to an unfiltered query.
    """
    collection = FakeCollection(empty_query_result())
    result = retrieve_filtered_context(
        "prompt", domain="EDGE", collection=collection, embedder=fake_embedder
    )

    assert result["available"] is False
    assert result["reason"] == REASON_NO_FILTER_MATCH
    assert result["domain"] == "EDGE"
    assert len(collection.calls) == 1  # no unfiltered retry


def test_filtered_context_without_scope_delegates_to_retrieve_context(
    populated_collection, fake_embedder
) -> None:
    result = retrieve_filtered_context(
        "prompt", collection=populated_collection, embedder=fake_embedder
    )
    assert result["available"] is True
    assert result["domain"] is None
    assert "where" not in populated_collection.calls[0]


def test_filtered_context_rejects_unknown_domain(
    populated_collection, fake_embedder
) -> None:
    with pytest.raises(ValueError, match="Invalid domain"):
        retrieve_filtered_context(
            "prompt",
            domain="NOT_A_DOMAIN",
            collection=populated_collection,
            embedder=fake_embedder,
        )


# --------------------------------------------------------------------------- #
# RAG-007 — future ranking, unit-testable independently of the n=1 MVP          #
# --------------------------------------------------------------------------- #


def test_rank_candidates_orders_by_distance() -> None:
    candidates = [
        {"record_id": "far", "distance": 0.9, "calibration": "x"},
        {"record_id": "near", "distance": 0.1, "calibration": "y"},
    ]
    ranked = rank_candidates(candidates)
    assert [item["record_id"] for item in ranked] == ["near", "far"]


def test_rank_candidates_applies_domain_bonus() -> None:
    candidates = [
        {
            "record_id": "slightly-nearer-wrong-domain",
            "distance": 0.20,
            "metadata": {"domain": "EDGE", "level": "weak"},
        },
        {
            "record_id": "matching-domain",
            "distance": 0.25,
            "metadata": {"domain": "CODE", "level": "weak"},
        },
    ]
    ranked = rank_candidates(candidates, domain="CODE")
    assert ranked[0]["record_id"] == "matching-domain"


def test_rank_score_sinks_candidates_without_distance() -> None:
    assert rank_score({"distance": None}) == float("-inf")
    assert rank_score({"distance": 0.5}) > rank_score({"distance": 0.9})
