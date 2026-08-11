"""
EVAL-009 — Batch-level strengths / gaps / improvement generation.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Builds the ``overall_summary`` object (strength, weakness, improvement) for the
batch from the evaluator output, with bounded word limits. The evaluator writes
the prose — this module validates it is present, caps it, and (only when the
evaluator omitted a field, which RAG-016 normally rejects first) falls back to a
deterministic summary derived from the computed scores so the API contract always
holds.

The fallback is labelled ``source: "derived"`` so a reader can never mistake a
generated line for the evaluator's own judgement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.evaluation.justification import enforce_summary_word_cap
from src.evaluation.rubric_loader import load_rubric
from src.rag.system_prompt import CRITERION_NAMES

SUMMARY_FIELDS = ("strength", "weakness", "improvement")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def criterion_averages(
    evaluations: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    """Average each criterion across the batch.

    Args:
        evaluations: Per-submission result blocks, each with a ``scores`` mapping.

    Returns:
        dict[str, float]: criterion -> mean score, rounded to two decimals.
    """
    criteria = load_rubric().criteria
    averages: Dict[str, float] = {}

    for criterion in criteria:
        values: List[float] = []
        for entry in evaluations:
            scores = entry.get("scores") or {}
            value = scores.get(criterion)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        averages[criterion] = round(_mean(values), 2)

    return averages


def _derive_summary(evaluations: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    """Build a deterministic fallback summary from the computed scores."""
    averages = criterion_averages(evaluations)

    if not averages:
        return {
            "strength": "No submissions were scored.",
            "weakness": "No submissions were scored.",
            "improvement": "Submit at least one candidate prompt for evaluation.",
        }

    best = max(averages, key=lambda criterion: averages[criterion])
    worst = min(averages, key=lambda criterion: averages[criterion])

    return {
        "strength": (
            f"Strongest criterion across the batch is "
            f"{best.upper()} {CRITERION_NAMES[best]} "
            f"(mean {averages[best]:.2f})."
        ),
        "weakness": (
            f"Weakest criterion across the batch is "
            f"{worst.upper()} {CRITERION_NAMES[worst]} "
            f"(mean {averages[worst]:.2f})."
        ),
        "improvement": (
            f"Focus the next iteration on {worst.upper()} "
            f"{CRITERION_NAMES[worst]}."
        ),
    }


def build_overall_summary(
    evaluator_summary: Optional[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Assemble the batch-level summary.

    Args:
        evaluator_summary: The ``overall_summary`` object from the validated
                           evaluator output, or None.
        evaluations:       Per-submission result blocks (used for the fallback
                           and for the criterion averages).

    Returns:
        dict: ``strength`` / ``weakness`` / ``improvement`` (each word-capped),
              plus ``source`` (``"evaluator"``, ``"derived"`` or ``"mixed"``),
              ``truncated`` (fields that were cut) and ``criterion_averages``.
    """
    derived = _derive_summary(evaluations)

    result: Dict[str, Any] = {}
    truncated: List[str] = []
    used_evaluator = False
    used_derived = False

    for field in SUMMARY_FIELDS:
        raw = None
        if isinstance(evaluator_summary, Mapping):
            candidate = evaluator_summary.get(field)
            if isinstance(candidate, str) and candidate.strip():
                raw = candidate
                used_evaluator = True

        if raw is None:
            raw = derived[field]
            used_derived = True

        capped, was_truncated = enforce_summary_word_cap(raw)
        result[field] = capped
        if was_truncated:
            truncated.append(field)

    if used_evaluator and used_derived:
        source = "mixed"
    elif used_evaluator:
        source = "evaluator"
    else:
        source = "derived"

    result["source"] = source
    result["truncated"] = truncated
    result["criterion_averages"] = criterion_averages(evaluations)

    return result
