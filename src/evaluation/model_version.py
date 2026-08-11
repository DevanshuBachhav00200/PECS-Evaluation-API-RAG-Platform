"""
EVAL-013 — Model-version tracking.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Records which LLM produced each evaluation result. Every stored evaluation record
carries a non-null ``model_version``, so a benchmark comparison can never
silently mix results from two different models.

The value is read from the API response when one is available (the response
reports the model that actually served the request, which is the authoritative
answer) and falls back to the configured ``config.MODEL_ID`` otherwise.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.rag.config import MODEL_ID


def resolve_model_version(response: Any = None) -> str:
    """Return the model string that produced a result.

    Args:
        response: Optional API response (object or dict).

    Returns:
        str: The response's ``model`` when present and non-empty, otherwise the
             configured evaluator model. Never empty.
    """
    if response is not None:
        model = (
            response.get("model")
            if isinstance(response, dict)
            else getattr(response, "model", None)
        )
        if isinstance(model, str) and model.strip():
            return model.strip()

    return MODEL_ID


def tag_with_model_version(
    result: Dict[str, Any],
    response: Any = None,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach ``model_version`` to an evaluation result.

    Args:
        result:        The result dict to tag.
        response:      Optional API response to read the model from.
        model_version: Explicit override, used by replay and test fixtures.

    Returns:
        dict: The same dict, mutated in place and returned for inline use.

    Raises:
        TypeError: If ``result`` is not a dict.
    """
    if not isinstance(result, dict):
        raise TypeError(f"result must be a dict, got {type(result).__name__}")

    result["model_version"] = (
        model_version
        if isinstance(model_version, str) and model_version.strip()
        else resolve_model_version(response)
    )
    return result
