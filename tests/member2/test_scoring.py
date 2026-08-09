from src.dataset.annotation.scoring import weighted_score, skill_level


def test_weighted_score():
    scores = {"c1": 8, "c2": 7, "c3": 9, "c4": 6, "c5": 10, "c6": 5}
    assert weighted_score(scores) == 7.6


def test_skill_boundaries():
    assert skill_level(3.99) == "weak"
    assert skill_level(4.0) == "developing"
    assert skill_level(6.99) == "developing"
    assert skill_level(7.0) == "strong"
