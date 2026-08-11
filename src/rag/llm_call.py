"""
RAG-012 / RAG-014 — Single-call LLM request with ephemeral prompt caching.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Sends one Claude Haiku 4.5 call carrying the whole batch, with the rubric system
block marked for ephemeral caching. A batch of N submissions is graded in exactly
one API call — that is the point of the batched design, and the acceptance
criterion for RAG-014.

Everything that can fail at the transport layer is funnelled through RAG-018, so
callers see a ``RagError`` with a classified code rather than an SDK exception.

The client is created lazily and cached at module level: the Anthropic client is
cheap but not free to construct, and the API service (Member 5) will call this on
every request.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.evaluation.error_types import (
    LLM_AUTH_ERROR,
    LLM_EMPTY_RESPONSE,
    LLM_REFUSAL,
    LLM_UNKNOWN_ERROR,
    RagError,
)
from src.rag.api_error_handler import to_rag_error
from src.rag.build_request import build_request, caching_diagnostics

_client: Any = None


def get_client(api_key: Optional[str] = None) -> Any:
    """Return the shared Anthropic client, creating it on first use.

    Args:
        api_key: Explicit key. When omitted the SDK resolves credentials from
                 the environment (``ANTHROPIC_API_KEY``).

    Raises:
        RagError: If the SDK is not installed, or no credential is available.
                  Code ``RAG_LLM_AUTH_ERROR`` / ``RAG_LLM_UNKNOWN_ERROR``.
    """
    global _client

    if _client is not None and api_key is None:
        return _client

    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - environment problem
        raise RagError(
            "The 'anthropic' package is not installed; cannot reach the evaluator.",
            code=LLM_UNKNOWN_ERROR,
            details={"missing_dependency": "anthropic"},
        ) from exc

    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key or not key.strip():
        raise RagError(
            "ANTHROPIC_API_KEY is not set; cannot authenticate against the evaluator.",
            code=LLM_AUTH_ERROR,
            details={"credential": "ANTHROPIC_API_KEY"},
        )

    client = anthropic.Anthropic(api_key=key)

    if api_key is None:
        _client = client

    return client


def reset_client() -> None:
    """Drop the cached client. Used by tests and after a credential rotation."""
    global _client
    _client = None


def _field(response: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` off an SDK response object or a plain dict."""
    if isinstance(response, dict):
        return response.get(name, default)
    return getattr(response, name, default)


def call_llm(
    request: Dict[str, Any],
    client: Optional[Any] = None,
) -> Any:
    """Send one batched evaluation request.

    Args:
        request: Payload from ``build_request()``.
        client:  Optional client with a ``messages.create(**kwargs)`` method.
                 Tests inject a fake here; production uses ``get_client()``.

    Returns:
        The raw SDK response (or whatever the injected client returned).

    Raises:
        RagError: On any transport, auth, rate-limit or status failure, with the
                  code assigned by RAG-018.
    """
    active = client if client is not None else get_client()

    try:
        return active.messages.create(**request)
    except RagError:
        raise
    except Exception as exc:  # noqa: BLE001 - classified by RAG-018
        raise to_rag_error(
            exc,
            context={
                "model": request.get("model"),
                "max_tokens": request.get("max_tokens"),
            },
        ) from exc


def extract_text(response: Any) -> str:
    """Extract the assistant's text from a Messages API response.

    Concatenates every ``text`` content block, skipping any non-text block.

    Raises:
        RagError: If the model refused (``stop_reason == "refusal"``) or the
                  response carries no text at all.
    """
    stop_reason = _field(response, "stop_reason")

    if stop_reason == "refusal":
        raise RagError(
            "Evaluator declined to grade this batch.",
            code=LLM_REFUSAL,
            details={"stop_reason": stop_reason},
        )

    content = _field(response, "content") or []
    chunks: List[str] = []

    for block in content:
        block_type = _field(block, "type")
        if block_type == "text":
            text = _field(block, "text")
            if isinstance(text, str):
                chunks.append(text)

    if not chunks:
        # Distinct from a refusal: the model returned no usable body at all, so
        # a fresh sample may succeed. RAG-017 retries this code (see
        # error_types.RETRYABLE_RESPONSE_CODES); it does not retry a refusal.
        raise RagError(
            "Evaluator response contained no text content.",
            code=LLM_EMPTY_RESPONSE,
            details={"stop_reason": stop_reason},
        )

    return "".join(chunks)


def usage_to_dict(response: Any) -> Dict[str, int]:
    """Extract the token-usage counters from a response.

    Returns:
        dict: ``input_tokens``, ``output_tokens``,
              ``cache_creation_input_tokens`` and ``cache_read_input_tokens``.
              Missing counters read as 0 rather than None, so cost arithmetic
              downstream never has to guard.
    """
    usage = _field(response, "usage")

    def counter(name: str) -> int:
        if usage is None:
            return 0
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, 0)
        return int(value) if isinstance(value, (int, float)) else 0

    return {
        "input_tokens": counter("input_tokens"),
        "output_tokens": counter("output_tokens"),
        "cache_creation_input_tokens": counter("cache_creation_input_tokens"),
        "cache_read_input_tokens": counter("cache_read_input_tokens"),
    }


def call_with_batch_content(
    batch_content: str,
    client: Optional[Any] = None,
    **request_overrides: Any,
) -> Any:
    """Build the request for ``batch_content`` and send it in one call.

    Convenience wrapper for the benchmark harness (BENCH-004) and manual
    smoke-testing; the engine composes the steps itself so it can run the token
    pre-flight (RAG-019) in between.
    """
    request = build_request(batch_content, **request_overrides)
    return call_llm(request, client=client)


def describe_caching(request: Dict[str, Any]) -> Dict[str, Any]:
    """Report the caching posture of a request (see RAG-012)."""
    return caching_diagnostics(request)
