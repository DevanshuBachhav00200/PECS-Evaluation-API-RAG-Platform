from __future__ import annotations
from src.common.normalization import normalize_text
from difflib import SequenceMatcher
import re
import pandas as pd

from .annotation_schema import CRITERIA
from .scoring import skill_level, weighted_score


def normalize_prompt(text: str) -> str:
    return normalize_text(text)


def validate_annotation_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = ["prompt_id", "skill_level", "domain", "prompt_text", *[f"{c}_score" for c in CRITERIA]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    errors = []
    for c in CRITERIA:
        col = f"{c}_score"
        numeric = pd.to_numeric(out[col], errors="coerce")
        bad = numeric.isna() | (numeric < 1) | (numeric > 10)
        errors.append(bad)
        out[col] = numeric
    any_bad = errors[0].copy()
    for e in errors[1:]:
        any_bad |= e
    out["score_range_error"] = any_bad

    expected = []
    for _, row in out.iterrows():
        try:
            expected.append(weighted_score({c: row[f"{c}_score"] for c in CRITERIA}))
        except Exception:
            expected.append(float("nan"))
    out["computed_final_score"] = expected
    if "final_score" in out.columns:
        out["final_score_error"] = (pd.to_numeric(out["final_score"], errors="coerce") - out["computed_final_score"]).abs() > 0.011
    else:
        out["final_score_error"] = False

    out["expected_skill_level"] = out["computed_final_score"].map(lambda x: skill_level(x) if pd.notna(x) else None)
    out["skill_level_mismatch"] = out.apply(
        lambda r: str(r.get("skill_level", "")).lower() != str(r.get("expected_skill_level", "")).lower()
        if r["expected_skill_level"] is not None else True,
        axis=1,
    )
    out["normalized_prompt"] = out["prompt_text"].map(normalize_prompt)
    return out


def leakage_report(df: pd.DataFrame, id_col: str = "prompt_id", similarity_threshold: float = 0.90) -> pd.DataFrame:
    if "dataset_split" not in df.columns:
        raise ValueError("dataset_split is required for leakage checks")
    rows = []
    for split_a, split_b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        a = df.loc[df.dataset_split == split_a, [id_col, "normalized_prompt"]].copy()
        b = df.loc[df.dataset_split == split_b, [id_col, "normalized_prompt"]].copy()
        id_overlap = len(set(a[id_col].astype(str)) & set(b[id_col].astype(str)))
        exact_text_overlap = len(set(a.normalized_prompt) & set(b.normalized_prompt))
        near = []
        for _, ra in a.iterrows():
            for _, rb in b.iterrows():
                ratio = SequenceMatcher(None, ra.normalized_prompt, rb.normalized_prompt).ratio()
                if ratio >= similarity_threshold:
                    near.append((ra[id_col], rb[id_col], ratio))
        near.sort(key=lambda x: x[2], reverse=True)
        rows.append({
            "split_a": split_a,
            "split_b": split_b,
            "prompt_id_overlap_count": id_overlap,
            "exact_prompt_text_overlap_count": exact_text_overlap,
            "near_duplicate_count": len(near),
            "max_similarity": round(near[0][2], 4) if near else 0.0,
            "near_duplicate_examples": ";".join(f"{a_id}|{b_id}|{score:.3f}" for a_id, b_id, score in near[:10]),
            "leakage_free_at_threshold": exact_text_overlap == 0 and len(near) == 0,
            "note": "prompt_id overlaps are identifier reuse and are not treated as leakage when prompt text is distinct",
        })
    return pd.DataFrame(rows)
