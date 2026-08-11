"""
RAG-020 — Usage and cost logging per call.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Extracts input / output / cache token counts from the API response and logs one
line per successful evaluation with an estimated USD cost, for audit. Every
successful call produces exactly one usage log line.

Prices are per million tokens for the configured evaluator model and are
overridable from the environment so a price change does not need a code change.
Cache multipliers follow the documented Anthropic pricing model: a cache write
costs 1.25x base input, a cache read 0.1x.

The logger follows the FOUND-005 module-level logger pattern: a named logger,
no handler configuration here, so the API service and the container decide where
records land.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, Mapping, Optional

from src.rag.config import MODEL_ID

logger = logging.getLogger("pecs.rag.usage")


def _price_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


#: USD per 1M input tokens for the evaluator model (Claude Haiku 4.5: $1.00).
INPUT_PRICE_PER_MTOK = _price_env("PECS_PRICE_INPUT_PER_MTOK", 1.00)

#: USD per 1M output tokens for the evaluator model (Claude Haiku 4.5: $5.00).
OUTPUT_PRICE_PER_MTOK = _price_env("PECS_PRICE_OUTPUT_PER_MTOK", 5.00)

#: Cache-write premium multiplier on the input price (5-minute ephemeral TTL).
CACHE_WRITE_MULTIPLIER = 1.25

#: Cache-read discount multiplier on the input price.
CACHE_READ_MULTIPLIER = 0.10

_MILLION = 1_000_000.0


def estimate_cost_usd(usage: Mapping[str, Any]) -> Dict[str, float]:
    """Estimate the USD cost of one call from its usage counters.

    Args:
        usage: Counters as returned by ``llm_call.usage_to_dict()``.

    Returns:
        dict: Per-component costs plus ``total_usd``. Values are rounded to 6
              decimal places — a single Haiku call costs a fraction of a cent, so
              rounding to 2 would report every call as $0.00.
    """

    def counter(name: str) -> int:
        value = usage.get(name, 0)
        return int(value) if isinstance(value, (int, float)) else 0

    input_tokens = counter("input_tokens")
    output_tokens = counter("output_tokens")
    cache_write = counter("cache_creation_input_tokens")
    cache_read = counter("cache_read_input_tokens")

    input_usd = input_tokens / _MILLION * INPUT_PRICE_PER_MTOK
    output_usd = output_tokens / _MILLION * OUTPUT_PRICE_PER_MTOK
    cache_write_usd = (
        cache_write / _MILLION * INPUT_PRICE_PER_MTOK * CACHE_WRITE_MULTIPLIER
    )
    cache_read_usd = (
        cache_read / _MILLION * INPUT_PRICE_PER_MTOK * CACHE_READ_MULTIPLIER
    )

    total = input_usd + output_usd + cache_write_usd + cache_read_usd

    return {
        "input_usd": round(input_usd, 6),
        "output_usd": round(output_usd, 6),
        "cache_write_usd": round(cache_write_usd, 6),
        "cache_read_usd": round(cache_read_usd, 6),
        "total_usd": round(total, 6),
    }


def aggregate_usages(usages: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    """Sum per-attempt usage counters into one set of totals.

    A retried call (RAG-017) bills for every attempt, so the audit line must
    reflect the sum, not just the attempt that finally parsed.
    """
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    for usage in usages:
        for key in totals:
            value = usage.get(key, 0)
            if isinstance(value, (int, float)):
                totals[key] += int(value)

    return totals


def log_usage(
    usage: Mapping[str, Any],
    submissions: int,
    attempts: int = 1,
    model: Optional[str] = None,
    rubric_tag: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Emit one audit log line for a completed evaluation call.

    Args:
        usage:       Aggregated token counters for the call.
        submissions: Number of submissions graded.
        attempts:    Number of API calls actually made (>1 after a retry).
        model:       Model string; defaults to ``config.MODEL_ID``.
        rubric_tag:  Rubric tag that produced the result (EVAL-003).
        extra:       Additional fields to record (request id, estimate, ...).

    Returns:
        dict: The record that was logged, so callers can attach it to a
              response body or a benchmark row without recomputing it.
    """
    cost = estimate_cost_usd(usage)

    record: Dict[str, Any] = {
        "event": "evaluation_call",
        "model": model or MODEL_ID,
        "submissions": submissions,
        "attempts": attempts,
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "cache_creation_input_tokens": int(
            usage.get("cache_creation_input_tokens", 0) or 0
        ),
        "cache_read_input_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
        "cost_usd": cost["total_usd"],
        "cost_breakdown_usd": cost,
    }

    if rubric_tag is not None:
        record["rubric_tag"] = rubric_tag
    if extra:
        record.update(extra)

    logger.info("%s", json.dumps(record, sort_keys=True))

    return record
