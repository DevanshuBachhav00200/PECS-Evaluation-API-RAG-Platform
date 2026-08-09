from src.common.normalization import normalize_text


def test_normalize_text_lowercases_text():
    assert normalize_text("HELLO WORLD") == "hello world"


def test_normalize_text_strips_leading_and_trailing_whitespace():
    assert normalize_text("   leading and trailing   ") == "leading and trailing"


def test_normalize_text_collapses_internal_whitespace():
    assert normalize_text("a\t\tb\n\nc   d") == "a b c d"


def test_normalize_text_is_deterministic():
    raw = "Some Prompt Text"

    result_a = normalize_text(raw)
    result_b = normalize_text(raw)

    assert result_a == result_b


def test_normalize_text_is_idempotent():
    raw = "  Multiple   Spaces   And CAPS  "

    once = normalize_text(raw)
    twice = normalize_text(once)

    assert once == twice


def test_normalize_text_coerces_non_string_input():
    assert normalize_text(12345) == "12345"