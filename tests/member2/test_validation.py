import pandas as pd
from src.dataset.annotation.validation import validate_annotation_frame


def test_validation_computes_score():
    df = pd.DataFrame([{
        "prompt_id": "P1", "prompt_text": "x", "skill_level": "strong", "domain": "CODE",
        "c1_score": 8, "c2_score": 7, "c3_score": 9, "c4_score": 6, "c5_score": 10, "c6_score": 5,
        "final_score": 7.6
    }])
    out = validate_annotation_frame(df)
    assert out.loc[0, "computed_final_score"] == 7.6
    assert not bool(out.loc[0, "final_score_error"])
