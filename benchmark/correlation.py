"""
BENCH-006 — [Requires Validation] Correlation calculation.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Pearson and Spearman correlation between AI composite scores and the frozen
gold baseline (BENCH-002), to be compared against the documented >0.85 target.

The task is marked *[Requires Validation]* in the tracker for a reason, and this
module encodes that rather than leaving it to the reader: ``correlate()`` always
returns an ``interpretable`` flag and a ``sample_size_status``. Below the
configured minimum the coefficient is still computed — the function works — but
it is labelled ``insufficient_sample`` so no report can quote it as though it
settled the >0.85 question.

[DATA NEEDED] The minimum gold-set size for a conclusive correlation is Team
Beta's call and has not been signed off. ``DEFAULT_MIN_SAMPLE`` below is a
placeholder threshold used only to gate interpretation, and is labelled as such
everywhere it surfaces.

Acceptance (BENCH-006): the correlation function runs correctly; the result is
interpreted only once the sample size is adequate.
"""

from __future__ import annotations

from math import sqrt
from typing import Any, Dict, Mapping, Optional, Sequence

from benchmark.mae import METRIC_PRECISION, MetricError, error_series

#: Provisional interpretation threshold. NOT a signed-off statistical minimum —
#: see the [DATA NEEDED] note above. Raising or lowering it changes only whether
#: a coefficient is *labelled* interpretable, never whether it is computed.
DEFAULT_MIN_SAMPLE = 30

#: Documented programme target for AI-vs-human agreement.
TARGET_CORRELATION = 0.85


def _validated(
    predicted: Sequence[float], reference: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(predicted) != len(reference):
        raise MetricError(
            f"paired series must be the same length, got {len(predicted)} and "
            f"{len(reference)}"
        )
    if len(predicted) < 2:
        raise MetricError(
            "correlation needs at least 2 paired observations, got "
            f"{len(predicted)}"
        )
    return tuple(float(v) for v in predicted), tuple(float(v) for v in reference)


def pearson(predicted: Sequence[float], reference: Sequence[float]) -> Optional[float]:
    """Pearson product-moment correlation coefficient.

    Returns:
        float | None: ``r`` in [-1, 1], or ``None`` when either series has zero
        variance (every value identical), for which ``r`` is undefined rather
        than zero.
    """
    left, right = _validated(predicted, reference)
    n = len(left)

    mean_left = sum(left) / n
    mean_right = sum(right) / n

    covariance = sum(
        (a - mean_left) * (b - mean_right) for a, b in zip(left, right)
    )
    variance_left = sum((a - mean_left) ** 2 for a in left)
    variance_right = sum((b - mean_right) ** 2 for b in right)

    denominator = sqrt(variance_left * variance_right)
    if denominator == 0:
        return None

    return round(covariance / denominator, METRIC_PRECISION)


def _ranks(values: Sequence[float]) -> list[float]:
    """Fractional ranks, averaging ties (the standard Spearman tie handling)."""
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)

    position = 0
    while position < len(indexed):
        end = position
        while (
            end + 1 < len(indexed)
            and values[indexed[end + 1]] == values[indexed[position]]
        ):
            end += 1
        average_rank = (position + end) / 2.0 + 1.0
        for offset in range(position, end + 1):
            ranks[indexed[offset]] = average_rank
        position = end + 1

    return ranks


def spearman(predicted: Sequence[float], reference: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation coefficient (ties averaged).

    Reported alongside Pearson because grading agreement is partly an ordering
    question — "does the evaluator rank candidates the way the reference does" —
    and rank correlation answers that without assuming a linear relationship
    between the two score scales.
    """
    left, right = _validated(predicted, reference)
    return pearson(_ranks(left), _ranks(right))


def correlate(
    rows: Sequence[Mapping[str, Any]],
    predicted_key: str = "ai_composite",
    reference_key: str = "gold_composite",
    min_sample: int = DEFAULT_MIN_SAMPLE,
) -> Dict[str, Any]:
    """Correlate AI scores against the gold baseline, with an honesty gate.

    Args:
        rows:          Comparison rows from BENCH-004.
        predicted_key: Row key holding the AI value.
        reference_key: Row key holding the baseline value.
        min_sample:    Provisional interpretation threshold ([DATA NEEDED]).

    Returns:
        dict with:
            n                    – paired observations actually used
            pearson / spearman   – coefficients, or None when undefined
            target               – the documented 0.85 programme target
            meets_target         – None unless the sample is interpretable
            interpretable        – False below ``min_sample``
            sample_size_status   – ``adequate`` | ``insufficient_sample``
            note                 – why the result is or is not conclusive
    """
    predicted, reference = error_series(rows, predicted_key, reference_key)

    if len(predicted) < 2:
        return {
            "n": len(predicted),
            "pearson": None,
            "spearman": None,
            "target": TARGET_CORRELATION,
            "meets_target": None,
            "interpretable": False,
            "sample_size_status": "insufficient_sample",
            "min_sample_threshold": min_sample,
            "note": (
                "Fewer than 2 comparable pairs — correlation is undefined, not "
                "zero."
            ),
        }

    r_pearson = pearson(predicted, reference)
    r_spearman = spearman(predicted, reference)

    interpretable = len(predicted) >= min_sample

    if not interpretable:
        note = (
            f"Computed over {len(predicted)} pairs, below the provisional "
            f"{min_sample}-pair interpretation threshold. Do not compare against "
            f"the {TARGET_CORRELATION} target. [DATA NEEDED] minimum gold-set "
            "size sign-off from Team Beta."
        )
    elif r_pearson is None:
        note = (
            "One of the series has zero variance, so Pearson r is undefined. "
            "This usually means the evaluator returned an identical composite "
            "for every record — a defect, not a correlation of 0."
        )
    else:
        note = (
            f"Computed over {len(predicted)} pairs against a baseline of "
            "3-judge consensus means, not SME human reference scores "
            "(BENCH-002). Read as model-vs-judge-consensus agreement pending "
            "SME sign-off."
        )

    return {
        "n": len(predicted),
        "pearson": r_pearson,
        "spearman": r_spearman,
        "target": TARGET_CORRELATION,
        "meets_target": (
            None
            if (not interpretable or r_pearson is None)
            else r_pearson >= TARGET_CORRELATION
        ),
        "interpretable": interpretable,
        "sample_size_status": "adequate" if interpretable else "insufficient_sample",
        "min_sample_threshold": min_sample,
        "note": note,
    }
