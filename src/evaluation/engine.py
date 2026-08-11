"""
Member 4 milestone entry point — the RAG evaluation engine.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Composes the whole Member 4 stack into the one function Member 5's
``POST /evaluate`` calls (API-005):

    retrieve calibration (RAG-003/005)  ->  assemble batch (RAG-009/010)
    ->  build request (RAG-011/012/013) ->  token pre-flight (RAG-019)
    ->  single LLM call + retries (RAG-014/015/016/017/018)
    ->  score guard (EVAL-004) -> composite (EVAL-006) -> skill level (EVAL-007)
    ->  ethics handling (EVAL-008) -> justifications (EVAL-005)
    ->  batch summary (EVAL-009) -> version tags (EVAL-003/013) -> usage log (RAG-020)

Two entry points, deliberately:

* ``evaluate_batch``      raises ``PecsEvaluationError`` on failure. Use inside
  Python when you want exceptions.
* ``evaluate_batch_safe`` returns the structured error object instead. This is
  what the FastAPI layer should call — it maps our ``error.code`` onto an HTTP
  status without ever seeing a traceback (API-010).

No step is inlined here; the engine only sequences the modules that own each
task, which keeps the separation of concerns documented in EVAL-014 real rather
than aspirational.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from src.evaluation.calculate_scores import calculate_scores
from src.evaluation.confidence import extract_confidence
from src.evaluation.error_types import (
    SUBMISSION_COUNT_MISMATCH,
    EvaluatorError,
    PecsEvaluationError,
)
from src.evaluation.ethics_gate import apply_ethics_gate
from src.evaluation.justification import extract_justifications
from src.evaluation.model_version import resolve_model_version, tag_with_model_version
from src.evaluation.overall_summary import build_overall_summary
from src.evaluation.rubric_loader import load_rubric
from src.evaluation.rubric_version import tag_with_rubric_version
from src.rag.build_batch_content import build_batch_content_with_items
from src.rag.build_request import build_request, caching_diagnostics
from src.rag.config import snapshot
from src.rag.retry_handler import call_and_validate
from src.rag.token_safety import preflight_check
from src.rag.usage_logger import aggregate_usages, estimate_cost_usd, log_usage


def evaluate_batch(
    qa_pairs: Sequence[Any],
    collection: Optional[Any] = None,
    embedder: Optional[Any] = None,
    client: Optional[Any] = None,
    retriever: Optional[Callable[..., Dict[str, Any]]] = None,
    max_attempts: Optional[int] = None,
    retrieve_calibration: bool = True,
    sleep: Optional[Callable[[float], None]] = None,
) -> Dict[str, Any]:
    """Grade a batch of candidate prompts and return structured C1-C6 results.

    Args:
        qa_pairs:  Task/prompt pairs. Each may be a mapping (``task`` +
                   ``student_prompt``, optionally ``domain`` / ``level``) or a
                   ``(task, student_prompt)`` tuple.
        collection: Optional ChromaDB collection (injection point).
        embedder:   Optional embedder instance (injection point).
        client:     Optional Anthropic-style client (injection point).
        retriever:  Optional retrieval callable (injection point).
        max_attempts: Retry budget override for malformed responses.
        retrieve_calibration: Set False to grade without calibration context.
        sleep:      Injectable sleep for the retry backoff (tests).

    Returns:
        dict: The evaluation result — see module docstring of
              ``docs/evaluation_architecture.md`` for the field-by-field contract.

    Raises:
        PecsEvaluationError: Any failure in the RAG or evaluation layer, carrying
                             a classified code (see EVAL-011).
    """
    rubric = load_rubric()

    batch_content, items = build_batch_content_with_items(
        qa_pairs,
        collection=collection,
        embedder=embedder,
        retriever=retriever,
        retrieve_calibration=retrieve_calibration,
    )

    request = build_request(batch_content)

    # RAG-019: refuse an oversized batch before spending a call on it.
    preflight = preflight_check(rubric.system_prompt, batch_content)

    call_result = call_and_validate(
        request,
        expected_count=len(items),
        client=client,
        max_attempts=max_attempts,
        sleep=sleep,
    )

    output = call_result["output"]
    response = call_result["response"]

    # Resolved once and stamped on every per-submission record as well as on the
    # batch result: Member 6's benchmark harness stores one row per submission
    # (BENCH-004/005), and each stored row must be self-describing rather than
    # relying on the caller to copy the batch-level tags down (EVAL-003/013).
    model_version = resolve_model_version(response)

    items_by_index = {item.submission_index: item for item in items}
    evaluations: List[Dict[str, Any]] = []

    for entry in output["evaluations"]:
        index = entry["submission_index"]
        item = items_by_index.get(index)

        if item is None:
            raise EvaluatorError(
                f"evaluator returned submission_index {index}, which was not "
                "part of the request",
                code=SUBMISSION_COUNT_MISMATCH,
                details={
                    "submission_index": index,
                    "submitted_indices": sorted(items_by_index),
                },
            )

        scored = calculate_scores(entry["scores"], submission_index=index)
        gated = apply_ethics_gate(scored["scores"], scored["composite_score"])
        justifications = extract_justifications(
            entry["justifications"], submission_index=index
        )
        confidence = extract_confidence(entry)

        evaluations.append(
            {
                "submission_index": index,
                "task": item.task,
                "scores": scored["scores"],
                "composite_score": gated["composite_score"],
                "skill_level": gated["skill_level"],
                "ethics_gate": gated["ethics_gate"],
                "justifications": justifications["justifications"],
                "justification_truncated": justifications["truncated"],
                "prompt_truncated": item.truncated,
                "calibration_used": item.calibration_available,
                "calibration_record_id": item.record_id,
                "domain": item.domain,
                "level": item.level,
                "model_version": model_version,
                "rubric_version": rubric.version,
                "rubric_tag": rubric.tag,
                **confidence,
            }
        )

    missing = sorted(set(items_by_index) - {e["submission_index"] for e in evaluations})
    if missing:
        raise EvaluatorError(
            f"evaluator returned no result for submission(s) {missing}",
            code=SUBMISSION_COUNT_MISMATCH,
            details={"missing_submission_indices": missing},
        )

    usage_totals = aggregate_usages(call_result["usages"])

    result: Dict[str, Any] = {
        "evaluations": evaluations,
        "overall_summary": build_overall_summary(
            output.get("overall_summary"), evaluations
        ),
        "usage": {
            **usage_totals,
            "attempts": call_result["attempts"],
            "estimated_input_tokens": preflight["estimated_input_tokens"],
            "cost_usd": estimate_cost_usd(usage_totals)["total_usd"],
        },
        "meta": {
            "submissions": len(items),
            "calibrated_submissions": sum(
                1 for item in items if item.calibration_available
            ),
            "weights": dict(rubric.weights),
            "config": snapshot(),
            "caching": caching_diagnostics(request),
        },
    }

    tag_with_rubric_version(result)
    tag_with_model_version(result, model_version=model_version)

    # RAG-020: exactly one usage line per completed evaluation.
    log_usage(
        usage_totals,
        submissions=len(items),
        attempts=call_result["attempts"],
        model=result["model_version"],
        rubric_tag=result["rubric_tag"],
        extra={"estimated_input_tokens": preflight["estimated_input_tokens"]},
    )

    return result


def evaluate_batch_safe(
    qa_pairs: Sequence[Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """``evaluate_batch`` with structured errors instead of exceptions.

    Returns:
        dict: Either the evaluation result, or ``{"error": {...}}`` from EVAL-011.
              Never raises for a pipeline failure, so the API layer has a single
              shape to map onto HTTP status codes.
    """
    try:
        return evaluate_batch(qa_pairs, **kwargs)
    except PecsEvaluationError as exc:
        return exc.to_dict()
