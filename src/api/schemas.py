"""Pydantic request and response models for PECS Evaluation API (API-002, API-003)."""

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

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "submissions": [
                        {
                            "task": "Write a Python function to sort a list of integers.",
                            "student_prompt": "def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(0, len(arr)-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
                            "domain": "CODE",
                            "level": "strong",
                        }
                    ]
                }
            ]
        }
    }

    def to_qa_pairs(self) -> List[Dict[str, Any]]:
        """Convert all submissions in the request into a list of QA pair dictionaries for the evaluation engine."""
        return [sub.to_qa_pair() for sub in self.submissions]



# ---------------------------------------------------------------------------
# Response Models (API-003)
# ---------------------------------------------------------------------------


class EthicsGateResponse(BaseModel):
    """C5 ethics gate evaluation audit details."""

    mode: str = Field(
        ...,
        description="Ethics handling mode (e.g. weighted_only, hard_gate).",
        examples=["weighted_only"],
    )
    triggered: bool = Field(
        ...,
        description="Whether hard-gate ethics veto triggered.",
        examples=[False],
    )
    c5_score: float = Field(
        ...,
        description="The C5 ethics criterion score.",
        examples=[9.0],
    )
    threshold: float = Field(
        ...,
        description="Ethics gate threshold value.",
        examples=[3.0],
    )
    capped_score: Optional[float] = Field(
        default=None,
        description="Capped composite score if ethics gate triggered, else None.",
        examples=[None],
    )
    pre_gate_composite: float = Field(
        ...,
        description="Composite score before ethics gate application.",
        examples=[7.05],
    )
    policy_confirmation: str = Field(
        ...,
        description="Status of Alliance Compliance policy confirmation.",
        examples=["pending_alliance_compliance_confirmation"],
    )


class SubmissionEvaluationResponse(BaseModel):
    """Evaluation result for a single prompt submission."""

    submission_index: int = Field(
        ...,
        ge=1,
        description="1-based position of the submission in the batch.",
        examples=[1],
    )
    task: str = Field(
        ...,
        description="Task assignment or problem description.",
        examples=["Write a Python function"],
    )
    scores: Dict[str, float] = Field(
        ...,
        description="Per-criterion scores for criteria c1 through c6.",
        examples=[{"c1": 8.0, "c2": 7.0, "c3": 6.0, "c4": 7.0, "c5": 9.0, "c6": 5.0}],
    )
    composite_score: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description="Weighted composite score on a 0.0 to 10.0 scale.",
        examples=[7.05],
    )
    skill_level: str = Field(
        ...,
        description="Mapped skill level band (weak, developing, strong).",
        examples=["strong"],
    )
    ethics_gate: EthicsGateResponse = Field(
        ...,
        description="Ethics gate audit block detailing mode and triggering status.",
    )
    justifications: Dict[str, str] = Field(
        ...,
        description="Per-criterion justification text for criteria c1 through c6.",
    )
    justification_truncated: List[str] = Field(
        default_factory=list,
        description="Criteria whose justification text exceeded word cap and was truncated.",
    )
    prompt_truncated: bool = Field(
        default=False,
        description="Whether student prompt was truncated during RAG assembly.",
    )
    calibration_used: bool = Field(
        default=False,
        description="Whether a calibration example was retrieved and included.",
    )
    calibration_record_id: Optional[str] = Field(
        default=None,
        description="Record ID of the calibration example used, if available.",
        examples=["rec-001"],
    )
    domain: Optional[str] = Field(
        default=None,
        description="Optional domain category scope applied at retrieval.",
    )
    level: Optional[str] = Field(
        default=None,
        description="Optional target skill level scope applied at retrieval.",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Evaluator confidence score normalized to 0.0-1.0 scale.",
    )
    confidence_label: Optional[str] = Field(
        default=None,
        description="Categorical confidence label (low, medium, high).",
    )
    confidence_status: str = Field(
        ...,
        description="Status of confidence signal parsing (e.g. not_implemented_in_mvp).",
        examples=["not_implemented_in_mvp"],
    )
    model_version: str = Field(
        ...,
        description="Evaluator model version used for this submission.",
        examples=["claude-haiku-4-5-20251001"],
    )
    rubric_version: str = Field(
        ...,
        description="Rubric version identifier.",
        examples=["pecs-rubric-v1.0"],
    )
    rubric_tag: str = Field(
        ...,
        description="Rubric tag with fingerprint.",
    )


class OverallSummaryResponse(BaseModel):
    """Batch-level summary of strengths, weaknesses, and improvement areas."""

    strength: str = Field(
        ...,
        description="Batch strength summary narrative.",
    )
    weakness: str = Field(
        ...,
        description="Batch weakness summary narrative.",
    )
    improvement: str = Field(
        ...,
        description="Batch improvement recommendations.",
    )
    source: str = Field(
        ...,
        description="Source of summary text (evaluator, derived, mixed).",
        examples=["evaluator"],
    )
    truncated: List[str] = Field(
        default_factory=list,
        description="Summary fields that were truncated.",
    )
    criterion_averages: Dict[str, float] = Field(
        ...,
        description="Average score across submissions for each criterion.",
        examples=[{"c1": 8.0, "c2": 7.0, "c3": 6.0, "c4": 7.0, "c5": 9.0, "c6": 5.0}],
    )


class UsageInfoResponse(BaseModel):
    """Token usage and estimated cost metrics for the evaluation batch."""

    input_tokens: int = Field(
        ...,
        ge=0,
        description="Total input tokens consumed by the LLM call.",
        examples=[1200],
    )
    output_tokens: int = Field(
        ...,
        ge=0,
        description="Total output tokens generated by the LLM call.",
        examples=[300],
    )
    cache_creation_input_tokens: int = Field(
        default=0,
        ge=0,
        description="Input tokens written to prompt cache.",
    )
    cache_read_input_tokens: int = Field(
        default=0,
        ge=0,
        description="Input tokens read from prompt cache.",
    )
    attempts: int = Field(
        default=1,
        ge=1,
        description="Total LLM call attempts made.",
        examples=[1],
    )
    estimated_input_tokens: int = Field(
        default=0,
        ge=0,
        description="Pre-flight estimated input tokens.",
        examples=[900],
    )
    cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated total evaluation cost in USD.",
        examples=[0.0027],
    )


class EvaluationMetaResponse(BaseModel):
    """Batch metadata including submissions count, weights, configuration, and caching."""

    submissions: int = Field(
        ...,
        ge=1,
        description="Total number of submissions evaluated in this batch.",
        examples=[1],
    )
    calibrated_submissions: int = Field(
        ...,
        ge=0,
        description="Number of submissions evaluated with calibration context.",
        examples=[1],
    )
    weights: Dict[str, float] = Field(
        ...,
        description="Criteria weight distribution used for composite scoring.",
    )
    config: Dict[str, Any] = Field(
        ...,
        description="Snapshot of effective evaluation configuration settings.",
    )
    caching: Dict[str, Any] = Field(
        ...,
        description="Diagnostics detailing prompt caching status.",
    )


class EvaluationResponse(BaseModel):
    """Successful evaluation batch response model matching Member 4 output structure."""

    evaluations: List[SubmissionEvaluationResponse] = Field(
        ...,
        description="List of per-submission evaluation results.",
    )
    overall_summary: OverallSummaryResponse = Field(
        ...,
        description="Batch-level overall summary.",
    )
    usage: UsageInfoResponse = Field(
        ...,
        description="Token usage and cost diagnostics for the batch.",
    )
    rubric_version: str = Field(
        ...,
        description="Rubric version identifier.",
        examples=["pecs-rubric-v1.0"],
    )
    rubric_tag: str = Field(
        ...,
        description="Rubric version tag with fingerprint.",
    )
    model_version: str = Field(
        ...,
        description="Model version used for batch evaluation.",
        examples=["claude-haiku-4-5-20251001"],
    )
    meta: EvaluationMetaResponse = Field(
        ...,
        description="Batch-level evaluation metadata.",
    )


class ErrorDetailResponse(BaseModel):
    """Structured error detail payload matching Member 4 error schema."""

    code: str = Field(
        ...,
        description="PECS error code (e.g. RAG_TOKEN_SAFETY_ABORT, EVAL_INVALID_INPUT).",
        examples=["RAG_TOKEN_SAFETY_ABORT"],
    )
    layer: str = Field(
        ...,
        description="Layer where error occurred (rag or evaluation).",
        examples=["rag"],
    )
    message: str = Field(
        ...,
        description="Operator-facing error description without stack trace.",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Machine-readable error details and context.",
    )
    retryable: bool = Field(
        default=False,
        description="Whether the request may be retried by caller.",
        examples=[False],
    )


class EvaluationErrorResponse(BaseModel):
    """Top-level error response model for API failure responses."""

    error: ErrorDetailResponse = Field(
        ...,
        description="Structured error details payload.",
    )


class HealthCheckResponse(BaseModel):
    """Health check response model for API-004."""

    status: str = Field(
        default="healthy",
        description="Health status of the PECS Evaluation API service.",
        examples=["healthy"],
    )
    version: str = Field(
        default="1.0.0",
        description="API service version.",
        examples=["1.0.0"],
    )
    embedder_loaded: bool = Field(
        default=True,
        description="Whether the sentence-transformer embedding model is warm-loaded and ready.",
        examples=[True],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy",
                    "version": "1.0.0",
                    "embedder_loaded": True,
                }
            ]
        }
    }



