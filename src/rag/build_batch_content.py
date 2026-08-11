"""
RAG-009 — build_batch_content(qa_pairs).

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Iterates over all task/prompt pairs, retrieves calibration context for each
(RAG-003 / RAG-005), truncates each prompt (RAG-010), and assembles one text
block for the batched LLM user message (RAG-013).

The output block contains exactly one correctly formatted entry per submitted
qa_pair, in the order they were submitted, with 1-based indices that the
evaluator must echo back as ``submission_index``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.evaluation.error_types import RETRIEVAL_FAILED, RagError
from src.evaluation.schema import EvaluationInput, normalize_qa_pair
from src.rag.empty_retrieval_handler import NO_CALIBRATION_TEXT
from src.rag.truncate import truncate_with_flag

#: Rendered section header for each submission.
_ENTRY_TEMPLATE = (
    "### Submission {index}\n"
    "Task: {task}\n"
    "Candidate prompt: {prompt}\n"
    "Calibration: {calibration}"
)


def _default_retriever(
    student_prompt: str,
    domain: Optional[str],
    level: Optional[str],
    collection: Optional[Any],
    embedder: Optional[Any],
) -> Dict[str, Any]:
    from src.rag.filtered_context import retrieve_filtered_context

    return retrieve_filtered_context(
        student_prompt,
        domain=domain,
        level=level,
        collection=collection,
        embedder=embedder,
    )


def build_batch_items(
    qa_pairs: Sequence[Any],
    collection: Optional[Any] = None,
    embedder: Optional[Any] = None,
    retriever: Optional[Callable[..., Dict[str, Any]]] = None,
    retrieve_calibration: bool = True,
) -> List[EvaluationInput]:
    """Build the per-submission evaluator inputs for a batch.

    Args:
        qa_pairs:  Sequence of task/prompt pairs. Each element may be a mapping
                   (``task`` + ``student_prompt``, optionally ``domain`` and
                   ``level``) or a ``(task, student_prompt)`` tuple.
        collection: Optional ChromaDB collection (test injection point).
        embedder:   Optional embedder instance (test injection point).
        retriever:  Optional retrieval callable, for tests and for future
                    ranking experiments. Defaults to RAG-005 filtered retrieval,
                    which falls through to RAG-003 when no scope is given.
        retrieve_calibration: Set False to assemble the batch without touching
                    the vector store at all (used by the benchmark harness to
                    measure uncalibrated grading).

    Returns:
        list[EvaluationInput]: One entry per qa_pair, in submission order.

    Raises:
        RagError:       On an invalid batch shape, or when retrieval itself
                        fails (as opposed to legitimately matching nothing).
        EvaluatorError: On a qa_pair that is missing a task or a prompt.
    """
    if isinstance(qa_pairs, (str, bytes)) or not isinstance(qa_pairs, Sequence):
        raise RagError(
            "qa_pairs must be a sequence of task/prompt pairs",
            code=RETRIEVAL_FAILED,
            details={"type": type(qa_pairs).__name__},
        )
    if len(qa_pairs) == 0:
        raise RagError(
            "qa_pairs must contain at least one submission",
            code=RETRIEVAL_FAILED,
            details={"count": 0},
        )

    fetch = retriever or _default_retriever
    items: List[EvaluationInput] = []

    for offset, pair in enumerate(qa_pairs):
        index = offset + 1
        normalized = normalize_qa_pair(pair, index)

        prompt, was_truncated = truncate_with_flag(normalized["student_prompt"])

        if retrieve_calibration:
            try:
                context = fetch(
                    prompt,
                    normalized["domain"],
                    normalized["level"],
                    collection,
                    embedder,
                )
            except Exception as exc:  # noqa: BLE001 - re-raised as structured
                raise RagError(
                    "Calibration retrieval failed for submission "
                    f"{index}: {type(exc).__name__}: {exc}",
                    code=RETRIEVAL_FAILED,
                    details={"submission_index": index},
                ) from exc
        else:
            context = {
                "available": False,
                "calibration": NO_CALIBRATION_TEXT,
                "record_id": None,
                "metadata": None,
            }

        items.append(
            EvaluationInput(
                submission_index=index,
                task=normalized["task"],
                student_prompt=prompt,
                calibration=context.get("calibration") or NO_CALIBRATION_TEXT,
                calibration_available=bool(context.get("available")),
                truncated=was_truncated,
                record_id=context.get("record_id"),
                domain=normalized["domain"],
                level=normalized["level"],
                metadata=context.get("metadata"),
            )
        )

    return items


def render_batch_content(items: Sequence[EvaluationInput]) -> str:
    """Render evaluator inputs into the single batched user-message block.

    Kept separate from retrieval so prompt assembly is testable without a
    vector store, and so the same rendering serves the token pre-flight
    estimate (RAG-019).
    """
    if not items:
        raise RagError(
            "cannot render an empty batch",
            code=RETRIEVAL_FAILED,
            details={"count": 0},
        )

    entries = [
        _ENTRY_TEMPLATE.format(
            index=item.submission_index,
            task=item.task.strip(),
            prompt=item.student_prompt.strip(),
            calibration=item.calibration.strip(),
        )
        for item in items
    ]

    header = (
        f"Evaluate the following {len(entries)} submission(s). "
        "Return one evaluations entry per submission, in this order."
    )

    return header + "\n\n" + "\n\n".join(entries)


def build_batch_content(
    qa_pairs: Sequence[Any],
    collection: Optional[Any] = None,
    embedder: Optional[Any] = None,
    retriever: Optional[Callable[..., Dict[str, Any]]] = None,
    retrieve_calibration: bool = True,
) -> str:
    """Assemble the batched user-message text block for ``qa_pairs``.

    This is the named RAG-009 deliverable. Use ``build_batch_content_with_items``
    when the caller also needs the per-submission metadata (which submissions
    got calibration, which prompts were truncated).
    """
    content, _ = build_batch_content_with_items(
        qa_pairs,
        collection=collection,
        embedder=embedder,
        retriever=retriever,
        retrieve_calibration=retrieve_calibration,
    )
    return content


def build_batch_content_with_items(
    qa_pairs: Sequence[Any],
    collection: Optional[Any] = None,
    embedder: Optional[Any] = None,
    retriever: Optional[Callable[..., Dict[str, Any]]] = None,
    retrieve_calibration: bool = True,
) -> Tuple[str, List[EvaluationInput]]:
    """Return ``(batch_content, items)`` for a batch of qa_pairs."""
    items = build_batch_items(
        qa_pairs,
        collection=collection,
        embedder=embedder,
        retriever=retriever,
        retrieve_calibration=retrieve_calibration,
    )
    return render_batch_content(items), items
