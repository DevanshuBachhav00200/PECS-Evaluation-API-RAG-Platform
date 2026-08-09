from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Literal

Confidence = Literal["low", "medium", "high"]
Status = Literal["draft", "reviewed", "approved", "rejected"]
GoldStatus = Literal["ai_only", "needs_sme_review", "gold_approved", "rejected"]

CRITERIA = ("c1", "c2", "c3", "c4", "c5", "c6")
WEIGHTS = {"c1": 0.15, "c2": 0.20, "c3": 0.20, "c4": 0.20, "c5": 0.15, "c6": 0.10}


@dataclass
class Annotation:
    """Canonical Member 2 annotation record.

    `record_id` is the stable downstream key. `prompt_id` is retained because
    source datasets may reuse it and must not be silently rewritten.
    """

    record_id: str
    prompt_id: str
    c1: float
    c2: float
    c3: float
    c4: float
    c5: float
    c6: float
    justification: str
    annotator_id: str
    timestamp: str
    confidence: Confidence
    status: Status = "draft"
    reviewer_id: str | None = None
    annotation_version: str = "pecs-annotations-v0.1"
    source: str = "unknown"
    gold_status: GoldStatus = "ai_only"

    def validate(self) -> None:
        if not self.record_id.strip():
            raise ValueError("record_id cannot be empty")
        if not self.prompt_id.strip():
            raise ValueError("prompt_id cannot be empty")
        if not self.annotator_id.strip():
            raise ValueError("annotator_id cannot be empty")
        if not self.justification.strip():
            raise ValueError("justification cannot be empty")
        if self.confidence not in {"low", "medium", "high"}:
            raise ValueError("confidence must be low, medium, or high")
        if self.status not in {"draft", "reviewed", "approved", "rejected"}:
            raise ValueError("invalid annotation status")
        if self.gold_status not in {"ai_only", "needs_sme_review", "gold_approved", "rejected"}:
            raise ValueError("invalid gold_status")
        for criterion in CRITERIA:
            value = getattr(self, criterion)
            if not 1 <= float(value) <= 10:
                raise ValueError(f"{criterion} must be between 1 and 10; got {value}")
        if self.status == "approved" and not self.reviewer_id:
            raise ValueError("approved annotation requires reviewer_id")
        if self.gold_status == "gold_approved" and self.status != "approved":
            raise ValueError("gold_approved requires approved annotation")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
