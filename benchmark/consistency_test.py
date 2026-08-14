"""
BENCH-007 — Consistency testing.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Re-runs the same prompts several times and measures score variance, to see how
stable grading is across repeated calls.

Two rules from Member 4's handoff shape this design and are enforced in code
rather than left to whoever runs it:

1. **Batch composition is held constant between trials.** Submissions in a batch
   share a single LLM call, so the rest of the batch is part of each
   submission's context. Varying it between repeats would measure batch
   composition, not evaluator stability. ``run_consistency`` sends the identical
   batch every trial and refuses a trial whose batch differs.
2. **Paraphrase trials are reported separately from identical-input trials.**
   They answer different questions — "is the evaluator deterministic" versus "is
   it robust to rewording" — and averaging them together hides both.

Acceptance (BENCH-007): a variance report generated across at least 5 repeated
trials. ``MIN_TRIALS`` enforces that floor.
"""

from __future__ import annotations

from statistics import pstdev
from typing import Any, Dict, List, Mapping, Optional, Sequence

from benchmark.evaluators import Evaluator, normalize_evaluation
from benchmark.mae import METRIC_PRECISION
from benchmark.test_set_loader import CRITERIA

#: Acceptance floor from BENCH-007.
MIN_TRIALS = 5


class ConsistencyError(RuntimeError):
    """Raised when a consistency run cannot satisfy its own preconditions."""


def _spread(values: Sequence[float]) -> Dict[str, Any]:
    """Summarise a series of repeated measurements."""
    if not values:
        return {"n": 0, "mean": None, "stdev": None, "min": None, "max": None, "range": None}

    numeric = [float(value) for value in values]
    mean = sum(numeric) / len(numeric)

    return {
        "n": len(numeric),
        "mean": round(mean, METRIC_PRECISION),
        # Population stdev: these trials are the whole population of repeats
        # performed, not a sample drawn from a larger set of runs.
        "stdev": round(pstdev(numeric), METRIC_PRECISION) if len(numeric) > 1 else 0.0,
        "min": round(min(numeric), METRIC_PRECISION),
        "max": round(max(numeric), METRIC_PRECISION),
        "range": round(max(numeric) - min(numeric), METRIC_PRECISION),
    }


def run_consistency(
    evaluator: Evaluator,
    qa_pairs: Sequence[Mapping[str, Any]],
    record_ids: Optional[Sequence[str]] = None,
    trials: int = MIN_TRIALS,
    label: str = "identical_input",
    enforce_minimum: bool = True,
) -> Dict[str, Any]:
    """Grade one fixed batch ``trials`` times and summarise the spread.

    Args:
        evaluator:   Adapter to grade through.
        qa_pairs:    The batch. Sent unchanged on every trial.
        record_ids:  Gold record ids for the batch (needed by replay mode).
        trials:      Number of repeats. Must be >= ``MIN_TRIALS`` unless
                     ``enforce_minimum`` is False.
        label:       ``identical_input`` or ``paraphrased`` — reported, never mixed.
        enforce_minimum: Set False only for a fast smoke check; a report built
                     from fewer than 5 trials records the relaxed floor.

    Returns:
        dict: Per-submission composite and per-criterion spread, plus the raw
              per-trial scores so the numbers are re-derivable.
    """
    if enforce_minimum and trials < MIN_TRIALS:
        raise ConsistencyError(
            f"BENCH-007 requires at least {MIN_TRIALS} trials, got {trials}"
        )
    if not qa_pairs:
        raise ConsistencyError("consistency run needs at least one submission")

    ids = list(record_ids or [f"submission-{i}" for i in range(1, len(qa_pairs) + 1)])
    frozen_batch = [dict(pair) for pair in qa_pairs]

    per_trial: List[Dict[str, Any]] = []

    for trial in range(1, trials + 1):
        outcome = evaluator.grade(frozen_batch, ids)

        trial_record: Dict[str, Any] = {
            "trial": trial,
            "ok": outcome.ok,
            "latency_seconds": round(outcome.latency_seconds, 4),
            "error_code": (outcome.error or {}).get("code"),
            "submissions": {},
        }

        for entry in outcome.evaluations:
            flat = normalize_evaluation(entry)
            index = flat["submission_index"]
            trial_record["submissions"][str(index)] = {
                "composite": flat["composite"],
                "skill_level": flat["skill_level"],
                **{criterion: flat[criterion] for criterion in CRITERIA},
            }

        per_trial.append(trial_record)

    successful = [trial for trial in per_trial if trial["ok"]]

    summaries: Dict[str, Any] = {}
    for position in range(1, len(frozen_batch) + 1):
        key = str(position)
        composites = [
            trial["submissions"][key]["composite"]
            for trial in successful
            if key in trial["submissions"]
            and trial["submissions"][key]["composite"] is not None
        ]
        levels = [
            trial["submissions"][key]["skill_level"]
            for trial in successful
            if key in trial["submissions"]
        ]

        criterion_spread = {}
        for criterion in CRITERIA:
            values = [
                trial["submissions"][key][criterion]
                for trial in successful
                if key in trial["submissions"]
                and trial["submissions"][key].get(criterion) is not None
            ]
            criterion_spread[criterion] = _spread(values)

        summaries[key] = {
            "record_id": ids[position - 1] if position - 1 < len(ids) else None,
            "composite": _spread(composites),
            "criteria": criterion_spread,
            "skill_levels_observed": sorted(set(levels)),
            "skill_level_stable": len(set(levels)) <= 1,
        }

    return {
        "label": label,
        "mode": evaluator.mode,
        "trials_requested": trials,
        "trials_succeeded": len(successful),
        "minimum_enforced": enforce_minimum,
        "batch_size": len(frozen_batch),
        "batch_held_constant": True,
        "per_submission": summaries,
        "per_trial": per_trial,
        "verdict": verdict(summaries),
    }


def verdict(summaries: Mapping[str, Any]) -> Dict[str, Any]:
    """Roll per-submission spread into one statement about grading stability."""
    composite_stdevs = [
        entry["composite"]["stdev"]
        for entry in summaries.values()
        if entry["composite"]["stdev"] is not None
    ]
    unstable_bands = [
        entry.get("record_id")
        for entry in summaries.values()
        if not entry["skill_level_stable"]
    ]

    if not composite_stdevs:
        return {
            "measurable": False,
            "reason": "no successful trials produced a composite score",
        }

    return {
        "measurable": True,
        "max_composite_stdev": round(max(composite_stdevs), METRIC_PRECISION),
        "mean_composite_stdev": round(
            sum(composite_stdevs) / len(composite_stdevs), METRIC_PRECISION
        ),
        "submissions_with_unstable_band": unstable_bands,
        "deterministic": max(composite_stdevs) == 0.0 and not unstable_bands,
    }


def paraphrase(prompt: str) -> str:
    """Produce a light paraphrase of a prompt for the robustness trial.

    Deliberately mechanical and lossless in meaning — a politeness prefix and a
    trailing clarifier. A semantic rewrite would confound "is the evaluator
    robust to phrasing" with "did the rewrite change the prompt's quality",
    which is the thing being measured.
    """
    stripped = prompt.strip()
    return f"Please help with the following. {stripped} Thank you."
