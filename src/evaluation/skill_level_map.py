"""
EVAL-007 — Skill-level mapping from composite score.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Maps a 0-10 composite score to a skill band:

    weak        0.00 - 3.99
    developing  4.00 - 6.99
    strong      7.00 - 10.00

The band boundaries are NOT re-implemented here. The mapping delegates to
Member 2's ANN-005 ``skill_level()`` so the two can never disagree — that was
the acceptance criterion for this task. The canonical band labels are lowercase
(``weak`` / ``developing`` / ``strong``), matching both ANN-005 and the
``level`` enumeration the vector store validates against (VDB-004).
"""

from __future__ import annotations

from typing import Dict, Tuple

from src.dataset.annotation.scoring import skill_level as _ann005_skill_level

#: Canonical band labels, weakest first.
BANDS: Tuple[str, ...] = ("weak", "developing", "strong")

#: Inclusive-lower / inclusive-upper display ranges for each band, for the
#: system prompt and the API documentation. The live boundary logic lives in
#: ANN-005; these values exist to describe it, not to enforce it.
BAND_RANGES: Dict[str, Tuple[float, float]] = {
    "weak": (0.0, 3.99),
    "developing": (4.0, 6.99),
    "strong": (7.0, 10.0),
}


def map_skill_level(composite_score: float) -> str:
    """Map a composite score to its skill band.

    Args:
        composite_score: Weighted composite score in the range 0-10.

    Returns:
        str: ``"weak"``, ``"developing"`` or ``"strong"``.

    Raises:
        TypeError:  If ``composite_score`` is not numeric.
        ValueError: If ``composite_score`` is outside 0-10.
    """
    if isinstance(composite_score, bool) or not isinstance(
        composite_score, (int, float)
    ):
        raise TypeError(
            f"composite_score must be a number, got {type(composite_score).__name__}"
        )

    return _ann005_skill_level(float(composite_score))


def describe_bands() -> str:
    """Render the bands as a single line for the system prompt / docs."""
    return " | ".join(
        f"{band} {low:.2f}-{high:.2f}" for band, (low, high) in BAND_RANGES.items()
    )
