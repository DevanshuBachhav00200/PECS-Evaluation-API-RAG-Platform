from __future__ import annotations

import pandas as pd


def approve_annotation(row: dict, reviewer_id: str) -> dict:
    if not reviewer_id or not reviewer_id.strip():
        raise ValueError("reviewer_id is required")
    annotator_id = str(row.get("annotator_id", "")).strip()
    if annotator_id and annotator_id == reviewer_id.strip():
        raise ValueError("reviewer and annotator must be different for second-pass review")
    out = dict(row)
    out["reviewer_id"] = reviewer_id.strip()
    out["status"] = "approved"
    return out


def require_review_for_gold(df: pd.DataFrame) -> None:
    required = {"reviewer_id", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required review columns: {sorted(missing)}")
    bad = df[(df.status == "approved") & (df.reviewer_id.isna() | (df.reviewer_id.astype(str).str.strip() == ""))]
    if not bad.empty:
        raise ValueError(f"{len(bad)} approved rows have no reviewer_id")
