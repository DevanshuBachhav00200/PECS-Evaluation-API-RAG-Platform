"""
VDB-005 — Implement embedding ingestion function.

Provides the ingest_record routine to embed raw text prompts and upsert
them into a ChromaDB collection using deterministic IDs and standard schemas.
"""

from __future__ import annotations

from typing import Any, Dict

from src.vectorstore.document_format import serialize_document
from src.vectorstore.metadata_schema import build_metadata


def ingest_record(record: Dict[str, Any], collection: Any = None) -> None:
    """
    Embed and ingest a single calibration record into the provided ChromaDB collection.

    Validation rules:
    - `collection` must be provided.
    - `record` must be a dictionary.
    - `prompt_id` must be present and non-empty.
    - `student_prompt` must be present.
    - `justification` must be present.
    - Input scores must contain exactly c1..c6 or c1_score..c6_score, with values
      between 1 and 10.
    - `domain`, `level`, `source`, `version` must be valid metadata.
    - If `embedding` is pre-supplied in the record, it must have length 384 and numeric values.
    - If `embedding` is not pre-supplied, `student_prompt` is embedded using the warm-loaded model.

    Idempotency:
    - Uses the record's `prompt_id` as the deterministic ChromaDB vector ID.
    - Upserts the record so that duplicate IDs are overwritten (updated) rather than duplicated.

    Args:
        record: The dictionary calibration record.
        collection: The ChromaDB collection instance to ingest into.

    Raises:
        TypeError: If types of arguments or record fields are invalid.
        ValueError: If required fields are missing, empty, or fail schema validation.
    """
    if collection is None:
        raise ValueError("ChromaDB collection must be provided")

    if not isinstance(record, dict):
        raise TypeError("record must be a dictionary")

    # 1. Validate prompt_id
    if "prompt_id" not in record:
        raise ValueError("record is missing required field: 'prompt_id'")
    prompt_id = record["prompt_id"]
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ValueError("prompt_id must be a non-empty string")

    # 2. Validate student_prompt
    if "student_prompt" not in record:
        raise ValueError("record is missing required field: 'student_prompt'")
    student_prompt = record["student_prompt"]

    # 3. Extract and validate scores and justification
    justification = record.get("justification")
    if justification is None:
        raise ValueError("record is missing required field: 'justification'")

    if "scores" in record and isinstance(record["scores"], dict):
        scores = record["scores"]
    else:
        # Fallback: extract scores from top-level of the record dictionary
        scores = {}
        expected_c1_c6 = {f"c{i}" for i in range(1, 7)}
        expected_c1_c6_score = {f"c{i}_score" for i in range(1, 7)}
        for k in record:
            if k in expected_c1_c6 or k in expected_c1_c6_score:
                scores[k] = record[k]

    # VDB-003 document serialization handles internal validation of scores and justification
    document_str = serialize_document(scores, justification)

    # 4. Extract and validate metadata
    domain = record.get("domain")
    level = record.get("level")
    source = record.get("source")
    version = record.get("version")

    # VDB-004 metadata building handles domain, level, source, and version validation
    metadata_dict = build_metadata(
        domain=domain,  # type: ignore (build_metadata checks types)
        level=level,    # type: ignore
        scores=scores,
        source=source,  # type: ignore
        version=version # type: ignore
    )

    # 5. Extract or generate embedding
    embedding = record.get("embedding")
    if embedding is not None:
        # Validate pre-computed embedding
        if not isinstance(embedding, list):
            raise TypeError("embedding must be a list of numbers")
        if len(embedding) != 384:
            raise ValueError(f"embedding dimension must be 384, got {len(embedding)}")
        for val in embedding:
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError("embedding values must be numeric")
    else:
        if not isinstance(student_prompt, str):
            raise TypeError("student_prompt must be a string")

        # EMB-007: Validate prompt text using existing error handling
        from src.embeddings.embed_error_handling import validate_embedding_input
        validation_res = validate_embedding_input(student_prompt)
        if not validation_res.is_valid:
            raise ValueError(
                f"Embedding input validation failed ({validation_res.error_code}): {validation_res.error_message}"
            )

        # EMB-003: Generate embedding using existing model pipeline (which uses EMB-001 warm-loaded model)
        from src.embeddings.batch_embed import embed_texts
        embeddings_arr = embed_texts([student_prompt])
        embedding = embeddings_arr[0].tolist()

    # 6. Idempotent upsert into the ChromaDB collection.
    #
    # Delegated to VDB-007's upsert_record rather than calling collection.upsert
    # directly: VDB-007 owns duplicate-ID semantics, and a second implementation
    # of the write here would be free to drift from it. Imported at call time
    # because src.vectorstore.upsert imports back into this package.
    from src.vectorstore import upsert as _upsert

    _upsert.upsert_record(
        collection=collection,
        record_id=prompt_id,
        embedding=embedding,
        document=document_str,
        metadata=metadata_dict,
    )
