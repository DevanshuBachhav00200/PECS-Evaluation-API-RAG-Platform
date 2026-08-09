import pytest
from src.dataset.annotation.annotation_schema import Annotation


def base():
    return Annotation(
        record_id="R1", prompt_id="P1", c1=1, c2=2, c3=3, c4=4, c5=5, c6=6,
        justification="test", annotator_id="M2", timestamp="2026-01-01T00:00:00Z",
        confidence="high", status="draft"
    )


def test_valid_annotation():
    base().validate()


def test_score_range():
    a = base(); a.c1 = 11
    with pytest.raises(ValueError):
        a.validate()


def test_approved_requires_reviewer():
    a = base(); a.status = "approved"
    with pytest.raises(ValueError):
        a.validate()
