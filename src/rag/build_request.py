"""
RAG-013 — Batched user-message construction (Anthropic Messages API payload).

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Combines the assembled batch content (RAG-009) and the rubric system prompt
(RAG-011, loaded through EVAL-002) into the single request payload sent to the
evaluator model.

Two structural decisions matter here:

1. **Cache-friendly ordering.** Prompt caching is a prefix match: the frozen
   rubric goes in ``system`` (rendered first), the per-request batch content
   goes in ``messages`` (rendered last). Nothing volatile is ever interpolated
   into the system block, so the cached prefix stays byte-identical between
   calls (RAG-012).
2. **One call per batch.** The whole batch travels as one user message, so a
   batch of N submissions costs exactly one API call (RAG-014).

``validate_payload()`` is the dry-run check: it asserts the payload matches the
Messages API shape without sending anything.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.evaluation.rubric_loader import get_system_prompt
from src.rag.config import (
    ENABLE_PROMPT_CACHING,
    MAX_TOKENS,
    MODEL_ID,
    TEMPERATURE,
)
from src.rag.token_safety import estimate_tokens

#: Minimum cacheable prefix for the configured evaluator model, in tokens.
#: Anthropic silently declines to cache a shorter prefix — no error, just
#: ``cache_creation_input_tokens: 0``. Claude Haiku 4.5's minimum is 4,096,
#: which the condensed PECS rubric alone does not reach; see
#: docs/evaluation_architecture.md for what that means for RAG-012.
MIN_CACHEABLE_PREFIX_TOKENS: Dict[str, int] = {
    "claude-haiku-4-5": 4096,
    "claude-haiku-4-5-20251001": 4096,
}

_DEFAULT_MIN_CACHEABLE_PREFIX_TOKENS = 4096


def build_system_blocks(
    system_prompt: Optional[str] = None,
    enable_caching: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Build the ``system`` parameter as a list of content blocks.

    The rubric is sent as a single text block. When caching is enabled the block
    carries ``cache_control: {"type": "ephemeral"}`` (RAG-012), which marks the
    tools+system prefix for reuse by later calls within the cache window.
    """
    text = get_system_prompt() if system_prompt is None else system_prompt
    caching = ENABLE_PROMPT_CACHING if enable_caching is None else bool(enable_caching)

    block: Dict[str, Any] = {"type": "text", "text": text}
    if caching:
        block["cache_control"] = {"type": "ephemeral"}

    return [block]


def build_request(
    batch_content: str,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    enable_caching: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build the full Messages API request payload for one batched evaluation.

    Args:
        batch_content:  The assembled batch block from RAG-009.
        system_prompt:  Override for the rubric prompt (tests only).
        model:          Override for ``config.MODEL_ID``.
        max_tokens:     Override for ``config.MAX_TOKENS``.
        temperature:    Override for ``config.TEMPERATURE``.
        enable_caching: Override for ``config.ENABLE_PROMPT_CACHING``.

    Returns:
        dict: Keyword arguments for ``client.messages.create(**request)``.

    Raises:
        TypeError:  If ``batch_content`` is not a string.
        ValueError: If ``batch_content`` is empty, or the payload fails the
                    dry-run structural validation.
    """
    if not isinstance(batch_content, str):
        raise TypeError(
            f"batch_content must be a string, got {type(batch_content).__name__}"
        )
    if not batch_content.strip():
        raise ValueError("batch_content cannot be empty or whitespace only")

    payload: Dict[str, Any] = {
        "model": MODEL_ID if model is None else model,
        "max_tokens": MAX_TOKENS if max_tokens is None else int(max_tokens),
        "temperature": TEMPERATURE if temperature is None else float(temperature),
        "system": build_system_blocks(system_prompt, enable_caching),
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": batch_content}],
            }
        ],
    }

    validate_payload(payload)
    return payload


def validate_payload(payload: Dict[str, Any]) -> None:
    """Dry-run structural validation of a Messages API payload.

    Checks the shape the API requires — model string, positive ``max_tokens``,
    a non-empty system block list, and exactly one user message whose content is
    a list of text blocks — without performing any network call.

    Raises:
        ValueError: On the first structural problem found.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")

    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("payload.model must be a non-empty string")

    max_tokens = payload.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
        raise ValueError("payload.max_tokens must be an integer")
    if max_tokens <= 0:
        raise ValueError(f"payload.max_tokens must be positive, got {max_tokens}")

    temperature = payload.get("temperature")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("payload.temperature must be numeric")
        if not 0.0 <= float(temperature) <= 1.0:
            raise ValueError(
                f"payload.temperature must lie in [0, 1], got {temperature}"
            )

    system = payload.get("system")
    if not isinstance(system, list) or not system:
        raise ValueError("payload.system must be a non-empty list of content blocks")
    for index, block in enumerate(system):
        if not isinstance(block, dict):
            raise ValueError(f"payload.system[{index}] must be a dict")
        if block.get("type") != "text":
            raise ValueError(f"payload.system[{index}].type must be 'text'")
        if not isinstance(block.get("text"), str) or not block["text"].strip():
            raise ValueError(f"payload.system[{index}].text must be a non-empty string")
        cache_control = block.get("cache_control")
        if cache_control is not None and cache_control.get("type") != "ephemeral":
            raise ValueError(
                f"payload.system[{index}].cache_control.type must be 'ephemeral'"
            )

    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("payload.messages must contain exactly one user message")

    message = messages[0]
    if not isinstance(message, dict) or message.get("role") != "user":
        raise ValueError("payload.messages[0].role must be 'user'")

    content = message.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("payload.messages[0].content must be a non-empty list")
    for index, block in enumerate(content):
        if not isinstance(block, dict) or block.get("type") != "text":
            raise ValueError(f"payload.messages[0].content[{index}].type must be 'text'")
        if not isinstance(block.get("text"), str) or not block["text"].strip():
            raise ValueError(
                f"payload.messages[0].content[{index}].text must be non-empty"
            )


def caching_diagnostics(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Report whether the cached prefix is long enough to actually cache.

    Marking a block with ``cache_control`` is necessary but not sufficient: a
    prefix shorter than the model's minimum is silently not cached. This helper
    makes that visible instead of leaving RAG-012's acceptance criterion looking
    like a bug.

    Returns:
        dict: ``requested`` (was cache_control set), ``prefix_tokens``
              (estimated), ``minimum_tokens`` (model's floor), and
              ``expected_effective`` (whether a cache hit is plausible).
    """
    system = payload.get("system") or []
    requested = any(
        isinstance(block, dict) and block.get("cache_control") for block in system
    )
    prefix_tokens = sum(
        estimate_tokens(block.get("text", ""))
        for block in system
        if isinstance(block, dict)
    )
    minimum = MIN_CACHEABLE_PREFIX_TOKENS.get(
        payload.get("model", ""), _DEFAULT_MIN_CACHEABLE_PREFIX_TOKENS
    )

    return {
        "requested": requested,
        "prefix_tokens": prefix_tokens,
        "minimum_tokens": minimum,
        "expected_effective": requested and prefix_tokens >= minimum,
    }
