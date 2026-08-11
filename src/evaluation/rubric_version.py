"""
EVAL-003 — Rubric versioning.

Owner: Member 4 (RAG Pipeline + AI Evaluation Engine)

Every evaluation result is tagged with the rubric that produced it, so results
graded under different rubric revisions are never silently compared in a
benchmark (BENCH-004) or an A/B report.

The tag has two parts:

    pecs-rubric-v1.0+3f9c1a2b7d04
    ^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^
    semantic version  content fingerprint of the live rubric text

The semantic version is bumped by hand when the rubric changes meaningfully.
The fingerprint is derived from the actual rubric bytes, so *any* edit changes
the stored tag even if someone forgets to bump the version — which is what makes
"a rubric edit changes the version tag on new records" true in practice rather
than by convention.
"""

from __future__ import annotations

from typing import Any, Dict

#: Hand-maintained semantic version of the C1-C6 rubric. Bump on any change to
#: weights, bands, criterion definitions, or the system prompt's scoring rules.
RUBRIC_VERSION = "pecs-rubric-v1.0"


def rubric_fingerprint() -> str:
    """Return the content fingerprint of the live rubric text."""
    from src.rag.system_prompt import system_prompt_fingerprint

    return system_prompt_fingerprint()


def rubric_tag() -> str:
    """Return the full ``<version>+<fingerprint>`` rubric tag."""
    return f"{RUBRIC_VERSION}+{rubric_fingerprint()}"


def tag_with_rubric_version(result: Dict[str, Any]) -> Dict[str, Any]:
    """Attach the rubric tag to an evaluation result.

    Args:
        result: An evaluation result dict.

    Returns:
        dict: The same dict (mutated in place and returned, so it can be used
              inline) carrying ``rubric_version`` and ``rubric_tag``.

    Raises:
        TypeError: If ``result`` is not a dict.
    """
    if not isinstance(result, dict):
        raise TypeError(f"result must be a dict, got {type(result).__name__}")

    result["rubric_version"] = RUBRIC_VERSION
    result["rubric_tag"] = rubric_tag()
    return result
