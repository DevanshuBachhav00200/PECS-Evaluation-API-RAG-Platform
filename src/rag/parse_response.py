"""
RAG-015 — LLM response handling with fence stripping.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

The system prompt instructs the evaluator to return raw JSON, but models
occasionally wrap output in a markdown code fence anyway. Stripping fences
defensively before parsing turns a whole class of avoidable retries into a
no-op: a response wrapped in ```json ... ``` still parses successfully.

Also tolerated: leading/trailing prose around a single JSON object (the object is
extracted by brace matching), and BOM/whitespace padding.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from src.evaluation.error_types import RESPONSE_NOT_JSON, ResponseValidationError

#: ```json\n{...}\n```  or  ```\n{...}\n```
_FENCE_PATTERN = re.compile(
    r"^\s*```(?:[a-zA-Z0-9_+-]*)\s*\n(?P<body>.*?)\n?\s*```\s*$",
    re.DOTALL,
)


def strip_code_fences(text: str) -> str:
    """Remove a wrapping markdown code fence, if present.

    Args:
        text: Raw model output.

    Returns:
        str: The fence body when the whole response is one fenced block,
             otherwise the input with surrounding whitespace stripped.

    Raises:
        TypeError: If ``text`` is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}")

    cleaned = text.lstrip("﻿").strip()

    match = _FENCE_PATTERN.match(cleaned)
    if match:
        return match.group("body").strip()

    return cleaned


def _extract_json_object(text: str) -> str:
    """Return the outermost ``{...}`` span, ignoring braces inside strings."""
    start = text.find("{")
    if start == -1:
        raise ResponseValidationError(
            "Evaluator response contained no JSON object.",
            code=RESPONSE_NOT_JSON,
            details={"response_preview": text[:200]},
        )

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ResponseValidationError(
        "Evaluator response contained an unterminated JSON object.",
        code=RESPONSE_NOT_JSON,
        details={"response_preview": text[:200]},
    )


def parse_llm_response(text: str) -> Dict[str, Any]:
    """Parse an evaluator response into a dict.

    Applies fence stripping, then a direct ``json.loads``, then a brace-matched
    extraction as a last resort before giving up.

    Args:
        text: Raw model output.

    Returns:
        dict: The parsed JSON object.

    Raises:
        ResponseValidationError: If no JSON object can be recovered, or the
                                 recovered value is not an object. Code
                                 ``RAG_RESPONSE_NOT_JSON``.
    """
    cleaned = strip_code_fences(text)

    if not cleaned:
        raise ResponseValidationError(
            "Evaluator response was empty.",
            code=RESPONSE_NOT_JSON,
            details={"response_preview": ""},
        )

    try:
        parsed: Any = json.loads(cleaned)
    except json.JSONDecodeError:
        candidate = _extract_json_object(cleaned)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ResponseValidationError(
                f"Evaluator response is not valid JSON: {exc.msg} "
                f"(line {exc.lineno}, column {exc.colno})",
                code=RESPONSE_NOT_JSON,
                details={"response_preview": cleaned[:200]},
            ) from exc

    if not isinstance(parsed, dict):
        raise ResponseValidationError(
            "Evaluator response must be a JSON object, got "
            f"{type(parsed).__name__}.",
            code=RESPONSE_NOT_JSON,
            details={"parsed_type": type(parsed).__name__},
        )

    return parsed
