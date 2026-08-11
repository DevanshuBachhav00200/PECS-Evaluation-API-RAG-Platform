"""
EVAL-008 — C5 Ethics handling logic.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Two behaviours are possible for C5 (Ethics & Safety) and the tracker requires the
one in force to be *confirmed*, not assumed:

1. **Weighted-only** (this build's default): C5 contributes its 15% weight to the
   composite like any other criterion, and nothing else.
2. **Hard-gate veto**: a C5 score at or below the gate threshold caps the
   composite regardless of how strong the other five criteria are.

**Current state: weighted-only.** ``config.ETHICS_HARD_GATE_ENABLED`` defaults to
False because the written confirmation flagged in the task tracker
([DATA NEEDED] — Team Beta / Alliance Compliance) has not been received. Both
paths are implemented and tested, so switching the decision is a config flip and
a rubric-version bump, not a code change.

Every result records which behaviour applied, so a benchmark run can never mix
gated and ungated scores without it being visible in the data.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from src.evaluation.skill_level_map import map_skill_level
from src.rag.config import (
    ETHICS_GATE_CAPPED_SCORE,
    ETHICS_GATE_THRESHOLD,
    ETHICS_HARD_GATE_ENABLED,
)

#: The criterion the gate is keyed on.
ETHICS_CRITERION = "c5"

MODE_WEIGHTED_ONLY = "weighted_only"
MODE_HARD_GATE = "hard_gate"

#: Recorded on every result so the pending confirmation is visible in the data
#: rather than only in this docstring.
CONFIRMATION_STATUS = "pending_alliance_compliance_confirmation"


def active_mode(enabled: Optional[bool] = None) -> str:
    """Return the ethics-handling mode currently in force."""
    gate_on = ETHICS_HARD_GATE_ENABLED if enabled is None else bool(enabled)
    return MODE_HARD_GATE if gate_on else MODE_WEIGHTED_ONLY


def apply_ethics_gate(
    scores: Mapping[str, Any],
    composite_score: float,
    enabled: Optional[bool] = None,
    threshold: Optional[float] = None,
    capped_score: Optional[float] = None,
) -> Dict[str, Any]:
    """Apply the configured C5 ethics handling to a composite score.

    Args:
        scores:          Validated per-criterion scores (must contain ``c5``).
        composite_score: The weighted composite from EVAL-006.
        enabled:         Override for ``config.ETHICS_HARD_GATE_ENABLED``.
        threshold:       Override for ``config.ETHICS_GATE_THRESHOLD``.
        capped_score:    Override for ``config.ETHICS_GATE_CAPPED_SCORE``.

    Returns:
        dict with keys:
            composite_score  – possibly capped composite
            skill_level      – band recomputed from the returned composite
            ethics_gate      – audit block: mode, whether it triggered, the C5
                               score, threshold, the pre-gate composite, and the
                               confirmation status of the policy decision

    Raises:
        KeyError:  If ``scores`` has no ``c5`` entry.
        TypeError: If ``composite_score`` is not numeric.
    """
    if isinstance(composite_score, bool) or not isinstance(
        composite_score, (int, float)
    ):
        raise TypeError(
            f"composite_score must be numeric, got {type(composite_score).__name__}"
        )

    if ETHICS_CRITERION not in scores:
        raise KeyError(
            f"scores must contain {ETHICS_CRITERION!r} to apply ethics handling"
        )

    mode = active_mode(enabled)
    ethics_score = float(scores[ETHICS_CRITERION])
    gate_threshold = (
        ETHICS_GATE_THRESHOLD if threshold is None else float(threshold)
    )
    cap = ETHICS_GATE_CAPPED_SCORE if capped_score is None else float(capped_score)

    pre_gate = round(float(composite_score), 2)
    triggered = mode == MODE_HARD_GATE and ethics_score <= gate_threshold
    final = min(pre_gate, cap) if triggered else pre_gate

    return {
        "composite_score": final,
        "skill_level": map_skill_level(final),
        "ethics_gate": {
            "mode": mode,
            "triggered": triggered,
            "c5_score": ethics_score,
            "threshold": gate_threshold,
            "capped_score": cap if triggered else None,
            "pre_gate_composite": pre_gate,
            "policy_confirmation": CONFIRMATION_STATUS,
        },
    }
