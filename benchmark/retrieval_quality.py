"""
BENCH-003 — Retrieval-quality test cases.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Verifies that Top-K retrieval (VDB-009 / RAG-003) returns *relevant* calibration
examples, not merely non-empty ones. Retrieval sits upstream of every score, so
a plausible-looking grading result can be built on a calibration example that
has nothing to do with the candidate prompt, and nothing downstream would show it.

Three families of check, in increasing strength:

``self_retrieval``   Query with an ingested record's own text; it must come back
                     as the top match. This is the sanity floor — a store that
                     fails it is misindexed, and every other retrieval number is
                     meaningless.
``near_pair``        Query with a curated near-paraphrase of an ingested record;
                     the intended record should rank in the Top-K.
``domain_scope``     Query under a domain filter (VDB-010 / RAG-005); no result
                     may come back from another domain.

Relevance for the ``near_pair`` family is a judgement call, so those cases are
reported for manual review with the retrieved text alongside the query, which is
what BENCH-003's acceptance criterion asks for ("retrieval relevance is manually
reviewed and logged for each test case"). The other two families are mechanically
decidable and are asserted outright.

Requires a populated vector store. Without one the module reports
``store_unavailable`` and explains how to build it, rather than reporting zero
relevance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from benchmark.test_set_loader import REPO_ROOT

#: EMB-006 serialised payload used to populate the store (Member 3's export).
DEFAULT_EMBEDDINGS_PATH = REPO_ROOT / "data" / "processed" / "embeddings.json"

#: Number of neighbours requested for relevance checks. Deliberately above the
#: MVP default of 1: a record that is rank 2 out of 3 is a different finding from
#: one that is absent entirely, and Top-1 alone cannot tell them apart.
RELEVANCE_TOP_K = 3


class StoreUnavailable(RuntimeError):
    """Raised when no populated vector store is reachable."""


def load_serialized_payload(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load Member 3's EMB-006 serialised embedding payload."""
    resolved = Path(path) if path is not None else DEFAULT_EMBEDDINGS_PATH
    if not resolved.exists():
        raise StoreUnavailable(
            f"serialised embeddings not found at {resolved}. Regenerate them "
            "with Member 3's embedding pipeline (see docs/embedding.md)."
        )
    return json.loads(resolved.read_text(encoding="utf-8"))


def ensure_populated_collection(collection: Optional[Any] = None) -> Any:
    """Return the calibration collection, asserting it is actually populated.

    Deliberately does *not* ingest on the fly. Populating the store is
    ``scripts/build_vector_store.py``'s job, and it has to make two decisions
    this module must not make silently: which dataset splits to include (the
    EMB-006 export contains the held-out gold set, whose ingestion would
    invalidate every benchmark figure) and how to adapt the export onto the
    VDB-003/VDB-004 schemas. An implicit ingest here would pick both silently.

    Raises:
        StoreUnavailable: If ChromaDB is unavailable, unreadable, or empty.
    """
    try:
        from src.vectorstore.client import get_client
        from src.vectorstore.init import get_or_create_collection
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise StoreUnavailable(
            "chromadb is not installed — install requirements.txt to run "
            "retrieval-quality checks"
        ) from exc

    resolved = collection if collection is not None else get_or_create_collection(get_client())

    try:
        count = resolved.count()
    except Exception as exc:  # noqa: BLE001 - surfaced as a precondition failure
        raise StoreUnavailable(f"vector store is not readable: {exc}") from exc

    if count == 0:
        raise StoreUnavailable(
            "the calibration collection is empty. Populate it with:\n"
            "    python -m scripts.build_vector_store\n"
            "(train split only by default — see that script's split-leakage note)."
        )

    return resolved


def ingested_ids(collection: Any) -> set:
    """Ids actually present in the collection."""
    stored = collection.get(include=[])
    return set(stored.get("ids") or [])


def _record_text(payload: Dict[str, Any], index: int) -> str:
    """Recover the prompt text an ingested record was embedded from."""
    documents = payload.get("documents") or []
    metadatas = payload.get("metadatas") or []

    if index < len(metadatas):
        metadata = metadatas[index] or {}
        for key in ("normalized_prompt", "prompt_text", "text", "prompt"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value

    if index < len(documents):
        raw = documents[index]
        try:
            parsed = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return str(raw)
        for key in ("normalized_prompt", "prompt_text", "text", "prompt"):
            value = parsed.get(key) if isinstance(parsed, dict) else None
            if isinstance(value, str) and value.strip():
                return value
        justification = parsed.get("justification") if isinstance(parsed, dict) else None
        if isinstance(justification, str):
            return justification

    return ""


def self_retrieval_cases(
    payload: Dict[str, Any],
    sample: int = 10,
    available_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Build self-retrieval cases from ingested records.

    Args:
        payload:       The EMB-006 export, used for the original prompt text.
        sample:        How many cases to build.
        available_ids: Restrict to ids actually in the store. Required in
                       practice, because the export holds all 331 records while
                       the store deliberately holds only the train split — a
                       case built on an excluded record would report a
                       retrieval miss that is really a correct exclusion.

    Sampled evenly across the corpus rather than taking the first N, so the
    check does not silently only ever exercise one domain's block of records.
    """
    all_ids = payload.get("ids") or []
    if not all_ids:
        return []

    eligible = [
        index
        for index, record_id in enumerate(all_ids)
        if available_ids is None or record_id in available_ids
    ]
    if not eligible:
        return []

    step = max(1, len(eligible) // max(1, sample))
    selected = [eligible[position] for position in range(0, len(eligible), step)][:sample]

    cases: List[Dict[str, Any]] = []
    for index in selected:
        text = _record_text(payload, index)
        if not text.strip():
            continue
        metadata = (payload.get("metadatas") or [{}] * len(all_ids))[index] or {}
        cases.append(
            {
                "case_id": f"RQ-SELF-{index:04d}",
                "family": "self_retrieval",
                "query": text,
                "expected_record_id": all_ids[index],
                "domain": metadata.get("domain"),
            }
        )
    return cases


#: Curated near-paraphrase probes. Kept small and hand-written: the point is a
#: human-reviewable relevance judgement, not a large auto-generated set whose
#: "expected" answers nobody checked.
NEAR_PAIR_PROBES: Sequence[Dict[str, Any]] = (
    {
        "case_id": "RQ-NEAR-001",
        "family": "near_pair",
        "query": (
            "Act as a data scientist and explain approaches for dealing with "
            "outliers in financial data, comparing the options and noting the "
            "risks of getting it wrong."
        ),
        "expectation": (
            "Should retrieve an outlier-handling / data-cleaning calibration "
            "example from the CODE domain."
        ),
        "domain": "CODE",
    },
    {
        "case_id": "RQ-NEAR-002",
        "family": "near_pair",
        "query": (
            "You are a senior analyst. Present analysis findings to a "
            "non-technical audience in plain language with bullet points, and "
            "state the assumptions."
        ),
        "expectation": (
            "Should retrieve a communicate-results-to-stakeholders example."
        ),
        "domain": "CODE",
    },
    {
        "case_id": "RQ-NEAR-003",
        "family": "near_pair",
        "query": "write a story twist about a clock that stopped working",
        "expectation": (
            "Should retrieve the CREATE-domain plot-twist calibration example "
            "seeded by ANN-006, if CREATE records are present in the store."
        ),
        "domain": "CREATE",
    },
)


def run_retrieval_quality(
    collection: Optional[Any] = None,
    payload: Optional[Dict[str, Any]] = None,
    top_k: int = RELEVANCE_TOP_K,
    self_sample: int = 10,
    embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run all three retrieval-quality families against a populated store.

    Returns:
        dict: Per-case results, a mechanical pass rate for the decidable
              families, and the near-pair results laid out for manual review.

    Raises:
        StoreUnavailable: When no populated store or embedding model is reachable.
    """
    resolved_payload = payload if payload is not None else load_serialized_payload()
    resolved_collection = ensure_populated_collection(collection)
    stored_ids = ingested_ids(resolved_collection)

    from src.rag.retrieve_context import retrieve_context
    from src.rag.filtered_context import retrieve_filtered_context

    results: List[Dict[str, Any]] = []

    # --- family 1: self-retrieval (mechanically decidable) -----------------
    for case in self_retrieval_cases(
        resolved_payload, sample=self_sample, available_ids=stored_ids
    ):
        context = retrieve_context(
            case["query"],
            n_results=top_k,
            collection=resolved_collection,
            embedder=embedder,
        )
        returned_ids = [example["record_id"] for example in context.get("examples", [])]
        results.append(
            {
                **case,
                "available": context.get("available"),
                "top_id": returned_ids[0] if returned_ids else None,
                "returned_ids": returned_ids,
                "rank_of_expected": (
                    returned_ids.index(case["expected_record_id"]) + 1
                    if case["expected_record_id"] in returned_ids
                    else None
                ),
                "top1_correct": bool(returned_ids)
                and returned_ids[0] == case["expected_record_id"],
                "decidable": True,
                "distance": (
                    context["examples"][0].get("distance")
                    if context.get("examples")
                    else None
                ),
            }
        )

    # --- family 2: near-pair relevance (manual review) ---------------------
    for probe in NEAR_PAIR_PROBES:
        context = retrieve_context(
            probe["query"],
            n_results=top_k,
            collection=resolved_collection,
            embedder=embedder,
        )
        examples = context.get("examples", [])
        results.append(
            {
                **probe,
                "available": context.get("available"),
                "returned_ids": [example["record_id"] for example in examples],
                "retrieved": [
                    {
                        "record_id": example["record_id"],
                        "distance": example.get("distance"),
                        "domain": (example.get("metadata") or {}).get("domain"),
                        "level": (example.get("metadata") or {}).get("level"),
                        # The formatted calibration line the evaluator would see —
                        # this is what a reviewer judges relevance from.
                        "calibration": example.get("calibration", "")[:400],
                    }
                    for example in examples
                ],
                "decidable": False,
                "review_status": "pending_manual_review",
            }
        )

    # --- family 3: domain scoping (mechanically decidable) -----------------
    domains = sorted(
        {
            (metadata or {}).get("domain")
            for metadata in (resolved_payload.get("metadatas") or [])
            if (metadata or {}).get("domain")
        }
    )
    for domain in domains:
        context = retrieve_filtered_context(
            "Explain your approach and state the assumptions you are making.",
            domain=domain,
            collection=resolved_collection,
            embedder=embedder,
        )
        examples = context.get("examples", [])
        returned_domains = sorted(
            {(example.get("metadata") or {}).get("domain") for example in examples}
        )
        results.append(
            {
                "case_id": f"RQ-SCOPE-{domain}",
                "family": "domain_scope",
                "query_domain": domain,
                "available": context.get("available"),
                "returned_domains": returned_domains,
                "returned_ids": [example["record_id"] for example in examples],
                "decidable": True,
                # Vacuously true when nothing matched — recorded explicitly so a
                # store with no records in a domain cannot read as a pass.
                "scope_respected": all(
                    value == domain for value in returned_domains if value is not None
                ),
                "matched_any": bool(examples),
            }
        )

    decidable = [case for case in results if case["decidable"]]
    self_cases = [case for case in decidable if case["family"] == "self_retrieval"]
    scope_cases = [case for case in decidable if case["family"] == "domain_scope"]

    return {
        "top_k": top_k,
        "corpus_size": len(resolved_payload.get("ids") or []),
        "collection_count": resolved_collection.count(),
        "cases": results,
        "summary": {
            "self_retrieval": {
                "n": len(self_cases),
                "top1_correct": sum(1 for case in self_cases if case["top1_correct"]),
                "found_in_top_k": sum(
                    1 for case in self_cases if case["rank_of_expected"] is not None
                ),
                "pass_rate": (
                    round(
                        sum(1 for case in self_cases if case["top1_correct"])
                        / len(self_cases),
                        4,
                    )
                    if self_cases
                    else None
                ),
            },
            "domain_scope": {
                "n": len(scope_cases),
                "scope_respected": sum(
                    1 for case in scope_cases if case["scope_respected"]
                ),
                "domains_with_no_match": [
                    case["query_domain"] for case in scope_cases if not case["matched_any"]
                ],
            },
            "near_pair": {
                "n": sum(1 for case in results if case["family"] == "near_pair"),
                "review_status": "pending_manual_review",
            },
        },
    }
