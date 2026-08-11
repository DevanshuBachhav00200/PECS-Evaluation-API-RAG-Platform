"""
RAG-007 — [Future / Post-MVP] Multi-candidate context ranking.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Designed but NOT active by default. The MVP retrieves a single calibration
example (``config.DEFAULT_TOP_K == 1``), so there is nothing to rank. This
module exists so that future edge-case tuning with ``n_results > 1`` has a
ranking strategy that is unit-testable on its own, independently of the n=1
default.

``ENABLED`` is the feature flag. ``apply_ranking()`` is a no-op while it is
False, so wiring this into the pipeline early cannot change MVP behaviour.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

#: Feature flag. Leave False for the MVP — see RAG-004 / RAG-007.
ENABLED: bool = False

#: Score bonus applied when a candidate's domain matches the requested domain.
DOMAIN_MATCH_BONUS: float = 0.10

#: Score bonus applied when a candidate's level matches the requested level.
LEVEL_MATCH_BONUS: float = 0.05


def _metadata_value(example: Mapping[str, Any], key: str) -> Optional[str]:
    metadata = example.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get(key)
    return value if isinstance(value, str) else None


def rank_score(
    example: Mapping[str, Any],
    domain: Optional[str] = None,
    level: Optional[str] = None,
) -> float:
    """Compute a ranking score for one candidate calibration example.

    Higher is better. Base signal is inverted distance (ChromaDB returns
    smaller-is-closer), with additive bonuses for metadata agreement.

    A candidate with no distance is treated as maximally distant so that
    records missing distance information never outrank real matches.
    """
    distance = example.get("distance")

    if isinstance(distance, (int, float)) and not isinstance(distance, bool):
        base = -float(distance)
    else:
        base = float("-inf")

    score = base

    if domain is not None and _metadata_value(example, "domain") == domain:
        score += DOMAIN_MATCH_BONUS
    if level is not None and _metadata_value(example, "level") == level:
        score += LEVEL_MATCH_BONUS

    return score


def rank_candidates(
    examples: Sequence[Mapping[str, Any]],
    domain: Optional[str] = None,
    level: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return the candidates ordered best-first.

    Pure function, deliberately independent of retrieval and of the MVP's n=1
    default so it can be unit-tested in isolation. The sort is stable, so
    equally-scored candidates keep ChromaDB's original ordering.

    Args:
        examples: Candidate examples as produced by ``build_examples()``.
        domain:   Optional requested domain, for the match bonus.
        level:    Optional requested level, for the match bonus.

    Returns:
        list[dict]: Copies of the inputs, each with a ``rank_score`` field,
                    sorted descending by that score.
    """
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        raise TypeError("examples must be a sequence of calibration examples")

    scored = [
        {**dict(example), "rank_score": rank_score(example, domain, level)}
        for example in examples
    ]

    return sorted(scored, key=lambda item: item["rank_score"], reverse=True)


def apply_ranking(
    retrieval_result: Dict[str, Any],
    domain: Optional[str] = None,
    level: Optional[str] = None,
) -> Dict[str, Any]:
    """Re-rank a retrieval result in place-safe fashion, if the flag is on.

    While ``ENABLED`` is False this returns ``retrieval_result`` unchanged, so
    the call site is safe to add before the feature is switched on.
    """
    if not ENABLED:
        return retrieval_result

    examples = retrieval_result.get("examples") or []
    if len(examples) < 2:
        return retrieval_result

    ranked = rank_candidates(examples, domain=domain, level=level)
    best = ranked[0]

    updated = dict(retrieval_result)
    updated["examples"] = ranked
    updated["calibration"] = best["calibration"]
    updated["record_id"] = best.get("record_id")
    updated["distance"] = best.get("distance")
    updated["metadata"] = best.get("metadata")
    return updated
