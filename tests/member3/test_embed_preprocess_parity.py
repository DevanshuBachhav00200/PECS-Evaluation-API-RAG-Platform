from src.dataset.annotation.validation import normalize_prompt
from src.embeddings.embed_preprocess import preprocess_for_embedding


def test_emb002_matches_ds005_on_plain_text():
    raw = "Simple prompt text"

    assert preprocess_for_embedding(raw) == normalize_prompt(raw)


def test_emb002_matches_ds005_on_mixed_case_and_whitespace():
    raw = "  MiXeD   CaSe\t\tWith\n\nWhitespace  "

    assert preprocess_for_embedding(raw) == normalize_prompt(raw)


def test_emb002_matches_ds005_on_non_string_input():
    raw = 98765

    assert preprocess_for_embedding(raw) == normalize_prompt(raw)


def test_emb002_matches_ds005_on_already_normalized_text():
    raw = "already normalized text"

    assert preprocess_for_embedding(raw) == normalize_prompt(raw)