from src.embeddings.embed_preprocess import preprocess_for_embedding


def test_representative_text_is_normalized():
    raw = "  Hello   WORLD  \n"

    expected = "hello world"

    assert preprocess_for_embedding(raw) == expected


def test_lowercase_applied():
    raw = "MiXeD CaSe TeXt"

    assert preprocess_for_embedding(raw) == "mixed case text"


def test_leading_and_trailing_whitespace_stripped():
    raw = "   leading and trailing   "

    assert preprocess_for_embedding(raw) == "leading and trailing"


def test_internal_whitespace_runs_collapsed():
    raw = "a\t\tb\n\nc   d"

    assert preprocess_for_embedding(raw) == "a b c d"


def test_same_input_produces_same_output_repeatedly():
    raw = "Some Prompt Text"

    result_a = preprocess_for_embedding(raw)
    result_b = preprocess_for_embedding(raw)

    assert result_a == result_b


def test_idempotent_preprocessing():
    raw = "  Multiple   Spaces   And CAPS  "

    once = preprocess_for_embedding(raw)
    twice = preprocess_for_embedding(once)

    assert once == twice


def test_non_string_input_is_coerced_to_string():
    raw = 12345

    assert preprocess_for_embedding(raw) == "12345"