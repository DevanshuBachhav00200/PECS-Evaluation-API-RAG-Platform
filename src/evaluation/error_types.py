"""
EVAL-011 — Evaluator-specific error handling and the structured error object.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Distinguishes evaluator-layer failures (bad score range, missing criterion,
malformed evaluator output) from RAG-layer failures (retrieval problems, LLM
call and transport problems, token safety aborts). Two mocked failures of
different kinds must produce two distinguishable error codes — that is the
acceptance criterion, and it is what lets Member 5 map errors onto HTTP status
codes (API-010) without string-matching messages.

Every failure leaves this layer as a structured object, never a raw exception
and never a stack trace:

    {"error": {"code": ..., "layer": ..., "message": ..., "details": {...},
               "retryable": bool}}
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

LAYER_RAG = "rag"
LAYER_EVALUATION = "evaluation"


# ---------------------------------------------------------------------------
# RAG-layer codes (retrieval, prompt assembly, LLM transport)
# ---------------------------------------------------------------------------

RETRIEVAL_FAILED = "RAG_RETRIEVAL_FAILED"
TOKEN_SAFETY_ABORT = "RAG_TOKEN_SAFETY_ABORT"
LLM_AUTH_ERROR = "RAG_LLM_AUTH_ERROR"
LLM_RATE_LIMITED = "RAG_LLM_RATE_LIMITED"
LLM_CONNECTION_ERROR = "RAG_LLM_CONNECTION_ERROR"
LLM_TIMEOUT = "RAG_LLM_TIMEOUT"
LLM_BAD_REQUEST = "RAG_LLM_BAD_REQUEST"
LLM_SERVER_ERROR = "RAG_LLM_SERVER_ERROR"
LLM_REFUSAL = "RAG_LLM_REFUSAL"
LLM_EMPTY_RESPONSE = "RAG_LLM_EMPTY_RESPONSE"
LLM_UNKNOWN_ERROR = "RAG_LLM_UNKNOWN_ERROR"
RESPONSE_NOT_JSON = "RAG_RESPONSE_NOT_JSON"
RESPONSE_SCHEMA_INVALID = "RAG_RESPONSE_SCHEMA_INVALID"
RETRY_BUDGET_EXHAUSTED = "RAG_RETRY_BUDGET_EXHAUSTED"

RAG_CODES = frozenset(
    {
        RETRIEVAL_FAILED,
        TOKEN_SAFETY_ABORT,
        LLM_AUTH_ERROR,
        LLM_RATE_LIMITED,
        LLM_CONNECTION_ERROR,
        LLM_TIMEOUT,
        LLM_BAD_REQUEST,
        LLM_SERVER_ERROR,
        LLM_REFUSAL,
        LLM_EMPTY_RESPONSE,
        LLM_UNKNOWN_ERROR,
        RESPONSE_NOT_JSON,
        RESPONSE_SCHEMA_INVALID,
        RETRY_BUDGET_EXHAUSTED,
    }
)

#: Failures that mean "the response body was unusable" — a fresh sample of the
#: same request may well succeed, so RAG-017 retries these and only these.
#: A refusal is deliberately excluded: it is a deterministic policy outcome, so
#: re-sending identical input can only burn budget.
RETRYABLE_RESPONSE_CODES = frozenset(
    {
        RESPONSE_NOT_JSON,
        RESPONSE_SCHEMA_INVALID,
        LLM_EMPTY_RESPONSE,
    }
)


# ---------------------------------------------------------------------------
# Evaluation-layer codes (scoring, rubric, output shaping)
# ---------------------------------------------------------------------------

INVALID_EVALUATION_INPUT = "EVAL_INVALID_INPUT"
SCORE_OUT_OF_RANGE = "EVAL_SCORE_OUT_OF_RANGE"
MISSING_CRITERION = "EVAL_MISSING_CRITERION"
SUBMISSION_COUNT_MISMATCH = "EVAL_SUBMISSION_COUNT_MISMATCH"
RUBRIC_UNAVAILABLE = "EVAL_RUBRIC_UNAVAILABLE"
ETHICS_GATE_UNCONFIRMED = "EVAL_ETHICS_GATE_UNCONFIRMED"

EVALUATION_CODES = frozenset(
    {
        INVALID_EVALUATION_INPUT,
        SCORE_OUT_OF_RANGE,
        MISSING_CRITERION,
        SUBMISSION_COUNT_MISMATCH,
        RUBRIC_UNAVAILABLE,
        ETHICS_GATE_UNCONFIRMED,
    }
)

#: Codes worth retrying at the caller's discretion (Member 5 / Member 6).
RETRYABLE_CODES = frozenset(
    {
        LLM_RATE_LIMITED,
        LLM_CONNECTION_ERROR,
        LLM_TIMEOUT,
        LLM_SERVER_ERROR,
    }
)


def layer_for_code(code: str) -> str:
    """Return the layer a code belongs to.

    Raises:
        ValueError: If the code is not a registered PECS error code.
    """
    if code in RAG_CODES:
        return LAYER_RAG
    if code in EVALUATION_CODES:
        return LAYER_EVALUATION
    raise ValueError(f"Unknown PECS error code: {code!r}")


def structured_error(
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the canonical structured error payload.

    Args:
        code:    One of the code constants in this module.
        message: Operator-facing description. Must not embed a stack trace.
        details: Optional machine-readable context (attempt counts, offending
                 criterion, estimated tokens, ...).

    Returns:
        dict: ``{"error": {...}}``, safe to serialise straight into an API
              response body.
    """
    return {
        "error": {
            "code": code,
            "layer": layer_for_code(code),
            "message": message,
            "details": details or {},
            "retryable": code in RETRYABLE_CODES,
        }
    }


def is_error(payload: Any) -> bool:
    """Return True when ``payload`` is a structured error object."""
    return isinstance(payload, dict) and "error" in payload


class PecsEvaluationError(Exception):
    """Base class for errors raised inside the RAG + evaluation stack.

    Carries a structured code so a caller that prefers exceptions to result
    objects still gets the same classification.
    """

    default_code = LLM_UNKNOWN_ERROR

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.details = details or {}

    @property
    def layer(self) -> str:
        return layer_for_code(self.code)

    def to_dict(self) -> Dict[str, Any]:
        """Render this exception as the structured error payload."""
        return structured_error(self.code, self.message, self.details)


class RagError(PecsEvaluationError):
    """A failure in retrieval, prompt assembly, or the LLM call itself."""

    default_code = RETRIEVAL_FAILED


class EvaluatorError(PecsEvaluationError):
    """A failure in scoring, rubric handling, or evaluator output shaping."""

    default_code = INVALID_EVALUATION_INPUT


class TokenSafetyAbort(RagError):
    """Pre-flight token estimate exceeded the configured cap (RAG-019)."""

    default_code = TOKEN_SAFETY_ABORT


class ResponseValidationError(RagError):
    """The evaluator response failed JSON parsing or schema validation."""

    default_code = RESPONSE_SCHEMA_INVALID
