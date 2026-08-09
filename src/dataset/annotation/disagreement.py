from __future__ import annotations

import pandas as pd

CRITERIA = [f"c{i}" for i in range(1, 7)]


def disagreement_report(df: pd.DataFrame, left_suffix="_annotator", right_suffix="_reviewer") -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        for c in CRITERIA:
            a, b = r[f"{c}{left_suffix}"], r[f"{c}{right_suffix}"]
            if pd.notna(a) and pd.notna(b) and float(a) != float(b):
                rows.append({
                    "prompt_id": r["prompt_id"],
                    "criterion": c.upper(),
                    "annotator_score": float(a),
                    "reviewer_score": float(b),
                    "absolute_difference": abs(float(a) - float(b)),
                })
    return pd.DataFrame(rows, columns=["prompt_id", "criterion", "annotator_score", "reviewer_score", "absolute_difference"])
