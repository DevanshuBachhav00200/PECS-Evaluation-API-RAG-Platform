"""
RAG-018 — LLM API failure handling.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Catches network, timeout, authentication and status errors coming out of the
Anthropic SDK and converts them into structured error objects. A raw exception
or a stack trace must never escape this layer — the API (Member 5) maps our
codes onto HTTP status codes, and a traceback in a response body would leak
internals to the Question Engine.

Classification is done by exception class name and ``status_code`` rather than by
``isinstance`` against SDK classes, for two reasons: the SDK is imported lazily
(so this module is testable without it installed), and a fake exception in a unit
test classifies identically to the real one.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.evaluation.error_types import (
    LLM_AUTH_ERROR,
    LLM_BAD_REQUEST,
    LLM_CONNECTION_ERROR,
    LLM_RATE_LIMITED,
    LLM_SERVER_ERROR,
    LLM_TIMEOUT,
    LLM_UNKNOWN_ERROR,
    RagError,
    structured_error,
)

#: SDK exception class name -> PECS error code.
_NAME_TO_CODE = {
    "AuthenticationError": LLM_AUTH_ERROR,
    "PermissionDeniedError": LLM_AUTH_ERROR,
    "RateLimitError": LLM_RATE_LIMITED,
    "APITimeoutError": LLM_TIMEOUT,
    "APIConnectionError": LLM_CONNECTION_ERROR,
    "BadRequestError": LLM_BAD_REQUEST,
    "NotFoundError": LLM_BAD_REQUEST,
    "UnprocessableEntityError": LLM_BAD_REQUEST,
    "InternalServerError": LLM_SERVER_ERROR,
}

#: HTTP status -> PECS error code, used when the class name is unfamiliar.
_STATUS_TO_CODE = {
    400: LLM_BAD_REQUEST,
    401: LLM_AUTH_ERROR,
    403: LLM_AUTH_ERROR,
    404: LLM_BAD_REQUEST,
    408: LLM_TIMEOUT,
    422: LLM_BAD_REQUEST,
    429: LLM_RATE_LIMITED,
}


def classify_api_error(exc: BaseException) -> str:
    """Map an exception raised by the Anthropic SDK to a PECS error code."""
    name = type(exc).__name__

    if name in _NAME_TO_CODE:
        return _NAME_TO_CODE[name]

    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status in _STATUS_TO_CODE:
            return _STATUS_TO_CODE[status]
        if status >= 500:
            return LLM_SERVER_ERROR
        return LLM_BAD_REQUEST

    if "Timeout" in name:
        return LLM_TIMEOUT
    if "Connection" in name:
        return LLM_CONNECTION_ERROR

    return LLM_UNKNOWN_ERROR


def _retry_after(exc: BaseException) -> Optional[str]:
    """Extract a ``retry-after`` hint when the SDK exposes response headers."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        return headers.get("retry-after")
    except Exception:  # noqa: BLE001 - header access is best-effort only
        return None


def to_rag_error(exc: BaseException, context: Optional[Dict[str, Any]] = None) -> RagError:
    """Wrap an SDK exception in a ``RagError`` carrying a classified code.

    Args:
        exc:     The exception raised by the SDK call.
        context: Optional extra details (model, attempt number, batch size).

    Returns:
        RagError: Never raised by this function — returned, so the caller
                  decides whether to raise it or convert it to a result object.
    """
    code = classify_api_error(exc)

    details: Dict[str, Any] = {
        "exception_type": type(exc).__name__,
    }
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        details["status_code"] = status
    request_id = getattr(exc, "request_id", None) or getattr(
        exc, "_request_id", None
    )
    if isinstance(request_id, str):
        details["request_id"] = request_id
    retry_after = _retry_after(exc)
    if retry_after is not None:
        details["retry_after"] = retry_after
    if context:
        details.update(context)

    return RagError(
        f"Evaluator LLM call failed ({type(exc).__name__}): {exc}",
        code=code,
        details=details,
    )


def handle_api_error(
    exc: BaseException,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert an SDK exception directly into a structured error payload."""
    error = to_rag_error(exc, context)
    return structured_error(error.code, error.message, error.details)
