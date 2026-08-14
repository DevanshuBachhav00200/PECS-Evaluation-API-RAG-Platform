"""
Build the ChromaDB calibration store from Member 3's serialised embeddings.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)
Supports: BENCH-003 (needs a populated store), DEP-003 (the container's mounted
volume has to be populated by something), DEP-004/DEP-008 (a container whose
``/evaluate`` returns real calibration).

Why this script exists rather than the one-liner in ``docs/embedding.md`` §12
(``collection.upsert(**load_serialized(path))``): the artifact EMB-006 actually
produces does not satisfy the schemas VDB-003/VDB-004 enforce, so the documented
handoff fails on the real file. Two concrete mismatches, both reproducible:

1. **Metadata shape.** ``data/processed/embeddings.json`` carries 32 metadata
   keys including ``prompt_id``, ``skill_level`` and ``c1_score``…``c6_score``.
   VDB-004 requires *exactly* ten — ``domain``, ``level``, ``c1``…``c6``,
   ``source``, ``version`` — and rejects both extras and the ``_score`` suffix.
2. **Document shape.** The export stores the raw prompt text as the document.
   VDB-003 expects a JSON string of ``{"scores": {...}, "justification": "..."}``,
   and RAG-008's calibration formatter reads exactly that. Ingesting raw text
   produces a store whose records retrieve but cannot be formatted, so every
   evaluation silently runs uncalibrated.

This script maps one to the other losslessly and ingests through the official
VDB-012 → VDB-007 path, so the store it builds is the same store the production
code expects. It is a **bridge, not a fix**: the defect is filed against EMB-006
in ``docs/known_issues_and_limitations.md``, and this script should be deleted
once the serializer emits VDB-004 metadata directly.

### Split leakage

By default the ``validation`` and ``test`` splits are **excluded**. The export
contains all 331 records, including the 39 held-out gold records BENCH-002
freezes as the comparison baseline. Ingesting those would hand the evaluator the
graded record's own reference scores as its "calibration example" during
benchmarking — every MAE figure produced afterwards would be measuring recall of
a leaked answer. ``--include-splits`` overrides this, deliberately noisily.

Usage::

    python -m scripts.build_vector_store                 # train split only
    python -m scripts.build_vector_store --dry-run       # report, write nothing
    python -m scripts.build_vector_store --include-splits train validation
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - script entry convenience
    sys.path.insert(0, str(REPO_ROOT))

from src.vectorstore.document_format import serialize_document  # noqa: E402
from src.vectorstore.metadata_schema import (  # noqa: E402
    VALID_DOMAINS,
    VALID_LEVELS,
    build_metadata,
)

DEFAULT_EMBEDDINGS_PATH = REPO_ROOT / "data" / "processed" / "embeddings.json"

#: Splits ingested unless overridden. See "Split leakage" above.
DEFAULT_SPLITS = ("train",)

CRITERIA = ("c1", "c2", "c3", "c4", "c5", "c6")

#: Corpus version tag stamped into VDB-004 metadata.
DEFAULT_VERSION = "pecs-v1.0"


def _scores(metadata: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Pull c1..c6 out of either key style, or None if incomplete."""
    resolved: Dict[str, float] = {}
    for criterion in CRITERIA:
        value = metadata.get(criterion, metadata.get(f"{criterion}_score"))
        if value is None:
            return None
        try:
            resolved[criterion] = float(value)
        except (TypeError, ValueError):
            return None
    return resolved


def adapt_record(
    record_id: str,
    metadata: Dict[str, Any],
    document: Any,
    version: str = DEFAULT_VERSION,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Map one EMB-006 record onto the VDB-003/VDB-004 shape.

    Returns:
        tuple: ``(adapted, None)`` on success, or ``(None, reason)`` when the
               record cannot be represented — never a silently patched record.
    """
    domain = str(metadata.get("domain") or "").strip()
    if domain not in VALID_DOMAINS:
        return None, f"domain {domain!r} is outside the VDB-004 enum {sorted(VALID_DOMAINS)}"

    level = str(metadata.get("level") or metadata.get("skill_level") or "").strip().lower()
    if level not in VALID_LEVELS:
        return None, f"level {level!r} is outside the VDB-004 enum {sorted(VALID_LEVELS)}"

    scores = _scores(metadata)
    if scores is None:
        return None, "record is missing one or more c1..c6 scores"

    justification = str(metadata.get("justification") or "").strip()
    if not justification:
        return None, "record has no justification — VDB-003 requires one"

    # ``source`` in VDB-004 means the split the record belongs to (the vector
    # store's own tests use train/validation), not the data's origin. The
    # export's ``source_type`` (intern/synthetic/public_dataset) is provenance
    # and is deliberately not conflated with it.
    source = str(metadata.get("dataset_split") or metadata.get("source") or "train").strip()

    try:
        adapted_metadata = build_metadata(
            domain=domain,
            level=level,
            scores=scores,
            source=source,
            version=version,
        )
        adapted_document = serialize_document(scores, justification)
    except (TypeError, ValueError) as exc:
        return None, f"VDB-003/VDB-004 validation rejected the record: {exc}"

    return (
        {
            "id": record_id,
            "metadata": adapted_metadata,
            "document": adapted_document,
            # Kept out of the store, used only for the split filter and report.
            "_split": str(metadata.get("dataset_split") or "").strip(),
            "_prompt_text": str(document or ""),
        },
        None,
    )


def adapt_payload(
    payload: Dict[str, Any],
    splits: Sequence[str] = DEFAULT_SPLITS,
    version: str = DEFAULT_VERSION,
) -> Dict[str, Any]:
    """Adapt a whole EMB-006 payload, reporting every record it could not take."""
    ids = payload.get("ids") or []
    embeddings = payload.get("embeddings") or []
    metadatas = payload.get("metadatas") or []
    documents = payload.get("documents") or []

    if not (len(ids) == len(embeddings) == len(metadatas) == len(documents)):
        raise ValueError(
            "serialised payload lists have mismatched lengths: "
            f"ids={len(ids)} embeddings={len(embeddings)} "
            f"metadatas={len(metadatas)} documents={len(documents)}"
        )

    allowed = {split.strip().lower() for split in splits}

    accepted_ids: List[str] = []
    accepted_embeddings: List[List[float]] = []
    accepted_metadatas: List[Dict[str, Any]] = []
    accepted_documents: List[str] = []

    rejected: List[Dict[str, str]] = []
    excluded_by_split = Counter()
    accepted_by_split = Counter()
    accepted_by_domain = Counter()

    for index, record_id in enumerate(ids):
        metadata = metadatas[index] or {}
        split = str(metadata.get("dataset_split") or "").strip().lower()

        if allowed and split and split not in allowed:
            excluded_by_split[split] += 1
            continue

        adapted, reason = adapt_record(
            str(record_id), metadata, documents[index], version=version
        )
        if adapted is None:
            rejected.append({"record_id": str(record_id), "reason": reason or "unknown"})
            continue

        accepted_ids.append(adapted["id"])
        accepted_embeddings.append(list(embeddings[index]))
        accepted_metadatas.append(adapted["metadata"])
        accepted_documents.append(adapted["document"])
        accepted_by_split[split or "unspecified"] += 1
        accepted_by_domain[adapted["metadata"]["domain"]] += 1

    return {
        "payload": {
            "ids": accepted_ids,
            "embeddings": accepted_embeddings,
            "metadatas": accepted_metadatas,
            "documents": accepted_documents,
        },
        "report": {
            "source_records": len(ids),
            "accepted": len(accepted_ids),
            "excluded_by_split": dict(excluded_by_split),
            "rejected": rejected,
            "accepted_by_split": dict(accepted_by_split),
            "accepted_by_domain": dict(accepted_by_domain),
            "splits_ingested": sorted(allowed),
            "version": version,
        },
    }


def build_store(
    embeddings_path: Optional[Path] = None,
    splits: Sequence[str] = DEFAULT_SPLITS,
    version: str = DEFAULT_VERSION,
    collection: Optional[Any] = None,
    reset: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Adapt and ingest the calibration corpus into ChromaDB.

    Args:
        embeddings_path: EMB-006 export. Defaults to ``data/processed/embeddings.json``.
        splits:          Dataset splits to ingest.
        version:         Corpus version tag written into metadata.
        collection:      Target collection. Defaults to the VDB-006 production one.
        reset:           Drop the collection before ingesting.
        dry_run:         Adapt and report without writing anything.

    Returns:
        dict: The adaptation report plus ``ingested`` and ``collection_count``.
    """
    resolved = Path(embeddings_path) if embeddings_path else DEFAULT_EMBEDDINGS_PATH
    if not resolved.exists():
        raise FileNotFoundError(
            f"serialised embeddings not found at {resolved}. Produce them with "
            "Member 3's embedding pipeline (docs/embedding.md)."
        )

    payload = json.loads(resolved.read_text(encoding="utf-8"))
    adapted = adapt_payload(payload, splits=splits, version=version)
    report = dict(adapted["report"])
    report["source_file"] = str(resolved)

    if dry_run:
        report["ingested"] = 0
        report["collection_count"] = None
        report["dry_run"] = True
        return report

    from src.vectorstore.bulk_ingest import bulk_ingest_from_serialized
    from src.vectorstore.client import get_client
    from src.vectorstore.init import COLLECTION_NAME, get_or_create_collection

    client = get_client()
    if collection is None:
        if reset and COLLECTION_NAME in {c.name for c in client.list_collections()}:
            client.delete_collection(name=COLLECTION_NAME)
        collection = get_or_create_collection(client)

    ingested = bulk_ingest_from_serialized(adapted["payload"], collection=collection)

    report["ingested"] = ingested
    report["collection_count"] = collection.count()
    report["dry_run"] = False
    return report


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(
        description="Populate the ChromaDB calibration store from EMB-006 embeddings"
    )
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument(
        "--include-splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help=(
            "dataset splits to ingest (default: train). Adding 'test' leaks the "
            "BENCH-002 gold baseline into calibration retrieval."
        ),
    )
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--reset", action="store_true", help="drop the collection first")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    report = build_store(
        embeddings_path=args.embeddings,
        splits=args.include_splits,
        version=args.version,
        reset=args.reset,
        dry_run=args.dry_run,
    )

    if "test" in {s.lower() for s in args.include_splits}:
        print(
            "WARNING: the 'test' split is being ingested. Benchmark results "
            "produced against this store are invalid — the evaluator can "
            "retrieve each gold record's own reference scores.",
            file=sys.stderr,
        )

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
