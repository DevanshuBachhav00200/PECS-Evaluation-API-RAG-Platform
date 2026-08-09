from __future__ import annotations

import pandas as pd


def flag_consistency(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flags = []
    for _, r in out.iterrows():
        reasons = []
        if bool(r.get("score_range_error", False)):
            reasons.append("score_out_of_range")
        if bool(r.get("final_score_error", False)):
            reasons.append("final_score_formula_mismatch")
        if bool(r.get("skill_level_mismatch", False)):
            reasons.append("skill_level_mismatch")
        if "justification" in out.columns and not str(r.get("justification", "") or "").strip():
            reasons.append("missing_justification")
        if "annotator_id" in out.columns and not str(r.get("annotator_id", "") or "").strip():
            reasons.append("missing_annotator_id")
        scores = [r.get(f"c{i}_score") for i in range(1, 7)]
        declared = str(r.get("skill_level", "")).lower()
        if all(pd.notna(s) and float(s) >= 9 for s in scores) and declared == "weak":
            reasons.append("extreme_score_level_conflict")
        if all(pd.notna(s) and float(s) <= 2 for s in scores) and declared == "strong":
            reasons.append("extreme_score_level_conflict")
        flags.append(";".join(reasons) if reasons else "OK")
    out["consistency_flag"] = flags
    out["needs_review"] = out.consistency_flag != "OK"
    return out
