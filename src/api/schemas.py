"""Pydantic request models for PECS Evaluation API (API-002)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class SubmissionRequest(BaseModel):
    """Request model representing a single prompt evaluation submission."""

    task: str = Field(
        ...,
        min_length=1,
        description="Task assignment or problem description.",
        examples=["Write a Python function to sort a list."],
    )
    student_prompt: str = Field(
        ...,
        min_length=1,
        description="The student prompt text to evaluate.",
        examples=["Prompt text for sorting array"],
    )
    domain: Optional[str] = Field(
        default=None,
        description="Optional domain category (e.g. CODE) to scope calibration retrieval.",
        examples=["CODE"],
    )
    level: Optional[str] = Field(
        default=None,
        description="Optional target skill level (e.g. strong, developing, weak).",
        examples=["strong"],
    )

    @field_validator("task", "student_prompt", mode="before")
    @classmethod
    def validate_non_whitespace(cls, value: Any) -> str:
        """Ensure string fields are non-empty and not just whitespace."""
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError("Field cannot be empty or whitespace only.")
            return stripped
        return value

    def to_qa_pair(self) -> Dict[str, Any]:
        """Convert submission request into the dictionary format expected by the evaluation engine."""
        pair: Dict[str, Any] = {
            "task": self.task,
            "student_prompt": self.student_prompt,
        }
        if self.domain is not None:
            pair["domain"] = self.domain
        if self.level is not None:
            pair["level"] = self.level
        return pair


class EvaluationRequest(BaseModel):
    """Request model for batch evaluation of submissions."""

    submissions: List[SubmissionRequest] = Field(
        ...,
        min_length=1,
        description="Non-empty list of submissions to evaluate.",
    )

    def to_qa_pairs(self) -> List[Dict[str, Any]]:
        """Convert all submissions in the request into a list of QA pair dictionaries for the evaluation engine."""
        return [sub.to_qa_pair() for sub in self.submissions]
