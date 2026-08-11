"""
RAG-004 — Expose Top-K (and the rest of the RAG knobs) as configurable parameters.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Single place where every tunable of the RAG + evaluation layer lives, so a
change to Top-K, the truncation limit, the token cap or the model string does
not require edits anywhere else in the pipeline.

Every value can be overridden through the environment (prefix ``PECS_``) so the
FastAPI service (Member 5) and the container (Member 6) can tune behaviour
without a code change.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# src/rag/config.py -> parents[2] == repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _str_env(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


# ---------------------------------------------------------------------------
# Retrieval (RAG-003 / RAG-004 / RAG-005)
# ---------------------------------------------------------------------------

#: Number of calibration examples pulled per candidate prompt.
#: MVP default is 1 (single closest pre-scored example). Documented as a
#: tunable for future edge-case testing (n>1) — see RAG-007.
DEFAULT_TOP_K: int = _int_env("PECS_RAG_TOP_K", 1)

#: Embedding dimensionality enforced across the repository (EMB-004 / VDB-005).
EMBEDDING_DIM: int = 384


# ---------------------------------------------------------------------------
# Prompt assembly (RAG-009 / RAG-010)
# ---------------------------------------------------------------------------

#: Per-submission student-prompt character limit applied before batch assembly.
MAX_PROMPT_CHARS: int = _int_env("PECS_MAX_PROMPT_CHARS", 1000)

#: Marker appended to a truncated prompt so the evaluator can see it was cut.
TRUNCATION_SUFFIX: str = "... [truncated]"


# ---------------------------------------------------------------------------
# Token pre-flight safety (RAG-019)
# ---------------------------------------------------------------------------

#: Heuristic characters-per-token ratio used for the pre-flight estimate.
CHARS_PER_TOKEN: float = _float_env("PECS_CHARS_PER_TOKEN", 3.5)

#: Estimated-input-token ceiling. Exceeding it aborts before any LLM call.
TOKEN_SAFETY_CAP: int = _int_env("PECS_TOKEN_SAFETY_CAP", 4000)


# ---------------------------------------------------------------------------
# LLM call (RAG-012 / RAG-014 / RAG-017 / EVAL-012 / EVAL-013)
# ---------------------------------------------------------------------------

#: Evaluator model. Dated snapshot rather than the ``claude-haiku-4-5`` alias so
#: EVAL-013 can record exactly which model produced a result.
MODEL_ID: str = _str_env("PECS_EVALUATOR_MODEL", "claude-haiku-4-5-20251001")

#: Output-token ceiling for one batched evaluation call.
MAX_TOKENS: int = _int_env("PECS_EVALUATOR_MAX_TOKENS", 4096)

#: Sampling temperature. 0.0 for the most repeatable scoring we can get from a
#: sampled model — see docs/evaluator_determinism.md (EVAL-012).
TEMPERATURE: float = _float_env("PECS_EVALUATOR_TEMPERATURE", 0.0)

#: Total attempts (1 initial + retries) when a response fails parse/validation.
MAX_LLM_ATTEMPTS: int = _int_env("PECS_MAX_LLM_ATTEMPTS", 3)

#: Mark the system block for ephemeral prompt caching (RAG-012).
ENABLE_PROMPT_CACHING: bool = _bool_env("PECS_ENABLE_PROMPT_CACHING", True)


# ---------------------------------------------------------------------------
# Evaluation output shaping (EVAL-005 / EVAL-008 / EVAL-009)
# ---------------------------------------------------------------------------

#: How an out-of-range evaluator score is handled (EVAL-004):
#: ``reject`` raises a structured error, ``clamp`` pins the value into 1-10.
#: Default is ``reject`` — a score of 15 means the evaluator misread the rubric,
#: and silently clamping it to 10 would bury that signal in the benchmark.
SCORE_RANGE_MODE: str = _str_env("PECS_SCORE_RANGE_MODE", "reject")

#: Word cap applied to each per-criterion justification.
JUSTIFICATION_WORD_CAP: int = _int_env("PECS_JUSTIFICATION_WORD_CAP", 40)

#: Word cap applied to each field of the batch-level overall_summary.
SUMMARY_WORD_CAP: int = _int_env("PECS_SUMMARY_WORD_CAP", 60)

#: Whether C5 (Ethics & Safety) acts as a hard-gate veto on the composite
#: score. Default OFF — the MVP composite is the plain weighted sum. Flipping
#: this on is blocked pending written confirmation from Team Beta / Alliance
#: Compliance (EVAL-008, [DATA NEEDED]). See src/evaluation/ethics_gate.py.
ETHICS_HARD_GATE_ENABLED: bool = _bool_env("PECS_ETHICS_HARD_GATE", False)

#: C5 score at or below which the hard gate fires (only when enabled above).
ETHICS_GATE_THRESHOLD: float = _float_env("PECS_ETHICS_GATE_THRESHOLD", 3.0)

#: Composite score a gated submission is capped at (only when enabled above).
ETHICS_GATE_CAPPED_SCORE: float = _float_env("PECS_ETHICS_GATE_CAPPED_SCORE", 3.99)


def snapshot() -> dict[str, object]:
    """Return the effective configuration.

    Used by the usage logger and the handoff docs so a run can be reproduced
    from its log line alone.
    """
    return {
        "top_k": DEFAULT_TOP_K,
        "max_prompt_chars": MAX_PROMPT_CHARS,
        "chars_per_token": CHARS_PER_TOKEN,
        "token_safety_cap": TOKEN_SAFETY_CAP,
        "model_id": MODEL_ID,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "max_llm_attempts": MAX_LLM_ATTEMPTS,
        "prompt_caching": ENABLE_PROMPT_CACHING,
        "score_range_mode": SCORE_RANGE_MODE,
        "justification_word_cap": JUSTIFICATION_WORD_CAP,
        "summary_word_cap": SUMMARY_WORD_CAP,
        "ethics_hard_gate": ETHICS_HARD_GATE_ENABLED,
    }
