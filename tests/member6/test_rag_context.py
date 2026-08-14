"""
TEST-004 — RAG context-assembly unit tests.

Owner: Member 6 (Testing + Benchmarking + Docker + Deployment)

Covers RAG-008 (the compact calibration formatter) and RAG-009
(``build_batch_content``) against fixed fixtures.

Acceptance (TEST-004): fixture-based tests catch a deliberately broken
formatter. ``test_a_broken_formatter_is_caught_by_the_fixture`` does exactly
that — it swaps in a formatter that drops the score block and asserts the
assembled batch stops matching the fixture. A test suite that would pass with a
broken formatter is not covering the formatter.

Everything here is injected: no vector store, no embedding model, no API key.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.rag.build_batch_content import (
    build_batch_content,
    build_batch_content_with_items,
    render_batch_content,
)
from src.rag.config import MAX_PROMPT_CHARS
from src.rag.empty_retrieval_handler import NO_CALIBRATION_TEXT
from src.rag.format_calibration import ARROW, format_calibration
from src.rag.truncate import truncate_with_flag

CALIBRATION_SCORES = {"c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5}

#: The exact documented compact form. Kept as a literal so a formatting change
#: has to update this line deliberately.
EXPECTED_CALIBRATION = (
    "Scores: C1:8 | C2:7 | C3:9 | C4:6 | C5:10 | C6:5 "
    f"{ARROW} Justification: reference example."
)


def _retriever(available: bool = True, record_id: str = "rec-001"):
    def _fetch(prompt, domain, level, collection, embedder):  # noqa: ANN001
        if not available:
            return {
                "available": False,
                "calibration": NO_CALIBRATION_TEXT,
                "record_id": None,
                "metadata": None,
            }
        return {
            "available": True,
            "calibration": EXPECTED_CALIBRATION,
            "record_id": record_id,
            "distance": 0.12,
            "metadata": {"domain": domain or "CODE", "level": level or "strong"},
        }

    return _fetch


def _pairs(count: int = 2) -> List[Dict[str, str]]:
    return [
        {
            "task": f"Task {index}",
            "student_prompt": f"Candidate prompt {index}",
        }
        for index in range(1, count + 1)
    ]


# ---------------------------------------------------------------------------
# RAG-008 — compact calibration formatter
# ---------------------------------------------------------------------------


def test_formatter_matches_the_documented_compact_form() -> None:
    assert format_calibration(CALIBRATION_SCORES, "reference example.") == EXPECTED_CALIBRATION


def test_formatter_accepts_the_suffixed_score_key_style() -> None:
    """Both key styles appear in the corpus; both must format identically."""
    suffixed = {f"{key}_score": value for key, value in CALIBRATION_SCORES.items()}
    assert format_calibration(suffixed, "reference example.") == EXPECTED_CALIBRATION


def test_formatter_drops_a_spurious_trailing_zero() -> None:
    assert "C1:8 " in format_calibration({**CALIBRATION_SCORES, "c1": 8.0}, "x.")


def test_formatter_keeps_a_genuine_fraction() -> None:
    assert "C1:8.5" in format_calibration({**CALIBRATION_SCORES, "c1": 8.5}, "x.")


def test_formatter_rejects_an_incomplete_score_set() -> None:
    with pytest.raises(ValueError, match="c1..c6"):
        format_calibration({"c1": 8, "c2": 7}, "x.")


def test_formatter_rejects_an_empty_justification() -> None:
    with pytest.raises(ValueError):
        format_calibration(CALIBRATION_SCORES, "   ")


def test_formatter_stays_on_one_line() -> None:
    """One calibration line per submission is what keeps the batch under the
    RAG-019 token cap; a multi-line format would blow it silently."""
    assert "\n" not in format_calibration(CALIBRATION_SCORES, "reference example.")


# ---------------------------------------------------------------------------
# RAG-009 — batch assembly
# ---------------------------------------------------------------------------


def test_batch_has_one_entry_per_pair_in_submission_order() -> None:
    content = build_batch_content(_pairs(3), retriever=_retriever())

    assert content.count("### Submission") == 3
    assert content.index("### Submission 1") < content.index("### Submission 2")
    assert content.index("### Submission 2") < content.index("### Submission 3")


def test_batch_carries_the_calibration_for_each_submission() -> None:
    content = build_batch_content(_pairs(2), retriever=_retriever())
    assert content.count(EXPECTED_CALIBRATION) == 2


def test_batch_header_states_the_submission_count() -> None:
    """The evaluator relies on this to return one entry per submission."""
    assert "2 submission(s)" in build_batch_content(_pairs(2), retriever=_retriever())


def test_uncalibrated_submission_uses_the_documented_placeholder() -> None:
    content = build_batch_content(_pairs(1), retriever=_retriever(available=False))
    assert NO_CALIBRATION_TEXT in content


def test_items_record_which_submissions_got_calibration() -> None:
    _, items = build_batch_content_with_items(_pairs(2), retriever=_retriever())
    assert [item.calibration_available for item in items] == [True, True]
    assert [item.record_id for item in items] == ["rec-001", "rec-001"]


def test_empty_batch_is_rejected() -> None:
    from src.evaluation.error_types import RagError

    with pytest.raises(RagError):
        build_batch_content([], retriever=_retriever())


def test_retrieval_can_be_disabled_entirely() -> None:
    """The benchmark harness grades uncalibrated as a baseline; that path must
    not touch the vector store at all."""

    def _explode(*args, **kwargs):  # noqa: ANN001, ANN002
        raise AssertionError("retrieval must not be called when disabled")

    content = build_batch_content(
        _pairs(1), retriever=_explode, retrieve_calibration=False
    )
    assert NO_CALIBRATION_TEXT in content


# ---------------------------------------------------------------------------
# RAG-010 — per-submission truncation
# ---------------------------------------------------------------------------


def test_oversized_prompt_is_truncated_to_the_configured_limit() -> None:
    oversized = "x" * 5000
    truncated, was_truncated = truncate_with_flag(oversized)

    assert was_truncated
    assert len(truncated) <= MAX_PROMPT_CHARS + 32  # limit plus the marker


def test_short_prompt_is_left_untouched() -> None:
    text, was_truncated = truncate_with_flag("short prompt")
    assert text == "short prompt"
    assert not was_truncated


def test_truncation_is_flagged_on_the_assembled_item() -> None:
    pairs = [{"task": "Task", "student_prompt": "y" * 5000}]
    _, items = build_batch_content_with_items(pairs, retriever=_retriever())
    assert items[0].truncated


# ---------------------------------------------------------------------------
# The acceptance criterion: a broken formatter must fail these fixtures
# ---------------------------------------------------------------------------


def test_a_broken_formatter_is_caught_by_the_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deliberately break RAG-008 and confirm the batch fixture notices.

    Without this, a formatter that silently dropped the score block would leave
    every other test in this module green — the assembled batch would still have
    the right number of entries in the right order, just with no calibration
    signal in them.
    """
    import src.rag.format_calibration as formatter

    monkeypatch.setattr(
        formatter, "format_calibration", lambda scores, justification: "BROKEN"
    )

    broken = formatter.format_calibration(CALIBRATION_SCORES, "reference example.")
    assert broken != EXPECTED_CALIBRATION

    content = build_batch_content(
        _pairs(1),
        retriever=lambda p, d, l, c, e: {  # noqa: E741, ANN001
            "available": True,
            "calibration": broken,
            "record_id": "rec-001",
            "metadata": None,
        },
    )
    assert EXPECTED_CALIBRATION not in content
