from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path


def append_audit_event(
    path: str | Path,
    prompt_id: str,
    field: str,
    old_value,
    new_value,
    changed_by: str,
    reason: str,
    *,
    record_id: str | None = None,
    annotation_version: str = "pecs-annotations-v0.1",
) -> None:
    """Append one immutable audit event as JSONL."""
    if not changed_by.strip():
        raise ValueError("changed_by is required")
    if not reason.strip():
        raise ValueError("reason is required")
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "record_id": record_id,
        "prompt_id": prompt_id,
        "field": field,
        "old_value": old_value,
        "new_value": new_value,
        "changed_by": changed_by,
        "reason": reason,
        "annotation_version": annotation_version,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
