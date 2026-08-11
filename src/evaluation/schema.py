"""
EVAL-001 — Evaluation input schema.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

The internal contract passed from the RAG layer into the evaluator: one entry
per submission carrying the task, the (already truncated) student prompt, and
the retrieved calibration example. This is deliberately *internal* — Member 5's
Pydantic request models (API-002) sit in front of it and convert inbound HTTP
payloads into these objects.

The schema is what ``build_batch_content()`` actually produces; ``from_qa_pair``
is the single conversion point, so the two cannot drift.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

from src.evaluation.error_types import (
    INVALID_EVALUATION_INPUT,
    EvaluatorError,
)

#: Accepted key spellings for the task field of an inbound qa_pair.
_TASK_KEYS = ("task", "task_description", "question")

#: Accepted key spellings for the prompt field of an inbound qa_pair.
_PROMPT_KEYS = ("student_prompt", "prompt", "answer", "candidate_prompt")


@dataclass(frozen=True)
class EvaluationInput:
    """One submission as the evaluator sees it.

    Attributes:
        submission_index: 1-based position in the batch. Used to match the
                          evaluator's ``submission_index`` back to the request.
        task:             The task the candidate was asked to write a prompt for.
        student_prompt:   The candidate's prompt, post RAG-010 truncation.
        calibration:      Compact calibration line (RAG-008), or the RAG-006
                          'no calibration available' placeholder.
        calibration_available: Whether a real calibration example was found.
        truncated:        Whether the prompt was truncated during assembly.
        record_id:        Id of the calibration record used, when available.
        domain:           Optional domain scope applied at retrieval.
        level:            Optional level scope applied at retrieval.
        metadata:         VDB-004 metadata of the calibration record.
    """

    submission_index: int
    task: str
    student_prompt: str
    calibration: str
    calibration_available: bool = False
    truncated: bool = False
    record_id: Optional[str] = None
    domain: Optional[str] = None
    level: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.submission_index, int) or isinstance(
            self.submission_index, bool
        ):
            raise EvaluatorError(
                "submission_index must be an integer",
                code=INVALID_EVALUATION_INPUT,
                details={"submission_index": repr(self.submission_index)},
            )
        if self.submission_index < 1:
            raise EvaluatorError(
                "submission_index must be 1-based",
                code=INVALID_EVALUATION_INPUT,
                details={"submission_index": self.submission_index},
            )
        for name in ("task", "student_prompt", "calibration"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise EvaluatorError(
                    f"{name} must be a string",
                    code=INVALID_EVALUATION_INPUT,
                    details={name: type(value).__name__},
                )
            if not value.strip():
                raise EvaluatorError(
                    f"{name} cannot be empty or whitespace only",
                    code=INVALID_EVALUATION_INPUT,
                    details={"submission_index": self.submission_index},
                )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _first_present(pair: Mapping[str, Any], keys: tuple) -> Optional[Any]:
    for key in keys:
        if key in pair and pair[key] is not None:
            return pair[key]
    return None


def normalize_qa_pair(pair: Any, submission_index: int) -> Dict[str, Any]:
    """Extract ``task`` / ``student_prompt`` / scope hints from one qa_pair.

    Accepts a mapping (several key spellings, so the Question Engine's payload
    shape does not have to match ours exactly) or a 2-tuple of
    ``(task, student_prompt)``.

    Raises:
        EvaluatorError: If the pair is not a recognised shape or is missing a
                        required field. Code ``EVAL_INVALID_INPUT``.
    """
    if isinstance(pair, Mapping):
        task = _first_present(pair, _TASK_KEYS)
        prompt = _first_present(pair, _PROMPT_KEYS)
        domain = pair.get("domain")
        level = pair.get("level")
    elif isinstance(pair, (tuple, list)) and len(pair) == 2:
        task, prompt = pair
        domain = None
        level = None
    else:
        raise EvaluatorError(
            "each qa_pair must be a mapping or a (task, student_prompt) pair",
            code=INVALID_EVALUATION_INPUT,
            details={"submission_index": submission_index, "type": type(pair).__name__},
        )

    if not isinstance(task, str) or not task.strip():
        raise EvaluatorError(
            "qa_pair is missing a non-empty task",
            code=INVALID_EVALUATION_INPUT,
            details={"submission_index": submission_index},
        )
    if not isinstance(prompt, str) or not prompt.strip():
        raise EvaluatorError(
            "qa_pair is missing a non-empty student_prompt",
            code=INVALID_EVALUATION_INPUT,
            details={"submission_index": submission_index},
        )

    return {
        "task": task,
        "student_prompt": prompt,
        "domain": domain if isinstance(domain, str) else None,
        "level": level if isinstance(level, str) else None,
    }
