"""
EMB-007 — Tests for embedding input error handling.

The tests verify that empty and oversized inputs are handled through
structured results instead of unhandled exceptions.

The existing EMB-002 preprocessing function is reused through the
production validation module, ensuring that normalization behavior
remains centralized.
"""

import pytest

from src.embeddings.embed_error_handling import (
    DEFAULT_MAX_TEXT_CHARS,
    prepare_embedding_input,
    validate_embedding_input,
)


# ---------------------------------------------------------------------------
# Valid input
# ---------------------------------------------------------------------------


def test_valid_input_returns_successful_structured_result():
    result = validate_embedding_input("Hello World")

    assert result.is_valid is True
    assert result.text == "hello world"
    assert result.error_code is None
    assert result.error_message is None


def test_valid_input_uses_emb002_normalization():
    result = validate_embedding_input(
        "  HELLO   WORLD\t\n"
    )

    assert result.is_valid is True
    assert result.text == "hello world"


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_empty_string_returns_structured_error():
    result = validate_embedding_input("")

    assert result.is_valid is False
    assert result.text is None
    assert result.error_code == "EMPTY_INPUT"
    assert result.error_message is not None


def test_whitespace_only_input_returns_structured_error():
    result = validate_embedding_input("   \t\n   ")

    assert result.is_valid is False
    assert result.text is None
    assert result.error_code == "EMPTY_INPUT"
    assert "empty" in result.error_message.lower()


def test_empty_input_does_not_raise_exception():
    """
    Core EMB-007 acceptance criterion:
    empty input must return a structured error rather than crash.
    """

    try:
        result = validate_embedding_input("")
    except Exception as exc:
        pytest.fail(
            f"Empty embedding input raised an unhandled exception: {exc}"
        )

    assert result.is_valid is False
    assert result.error_code == "EMPTY_INPUT"


# ---------------------------------------------------------------------------
# Oversized input
# ---------------------------------------------------------------------------


def test_oversized_input_returns_structured_error():
    result = validate_embedding_input(
        "a" * 101,
        max_chars=100,
    )

    assert result.is_valid is False
    assert result.text is None
    assert result.error_code == "INPUT_TOO_LARGE"
    assert result.error_message is not None


def test_input_at_maximum_size_is_accepted():
    text = "a" * 100

    result = validate_embedding_input(
        text,
        max_chars=100,
    )

    assert result.is_valid is True
    assert result.text == text


def test_oversized_input_does_not_raise_exception():
    """
    Oversized text must be rejected gracefully through the structured
    result rather than producing an unhandled exception.
    """

    try:
        result = validate_embedding_input(
            "x" * 101,
            max_chars=100,
        )
    except Exception as exc:
        pytest.fail(
            f"Oversized embedding input raised an unhandled exception: {exc}"
        )

    assert result.is_valid is False
    assert result.error_code == "INPUT_TOO_LARGE"


# ---------------------------------------------------------------------------
# Configurability
# ---------------------------------------------------------------------------


def test_default_maximum_is_defined():
    assert isinstance(DEFAULT_MAX_TEXT_CHARS, int)
    assert DEFAULT_MAX_TEXT_CHARS > 0


def test_custom_maximum_can_be_configured():
    result = validate_embedding_input(
        "hello",
        max_chars=4,
    )

    assert result.is_valid is False
    assert result.error_code == "INPUT_TOO_LARGE"


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def test_zero_max_chars_raises_configuration_error():
    with pytest.raises(ValueError, match="max_chars"):
        validate_embedding_input("hello", max_chars=0)


def test_negative_max_chars_raises_configuration_error():
    with pytest.raises(ValueError, match="max_chars"):
        validate_embedding_input("hello", max_chars=-1)


def test_non_integer_max_chars_raises_configuration_error():
    with pytest.raises(ValueError, match="max_chars"):
        validate_embedding_input("hello", max_chars=100.5)


# ---------------------------------------------------------------------------
# JSON/API-friendly structured result
# ---------------------------------------------------------------------------


def test_prepare_embedding_input_returns_plain_dict_for_empty_input():
    result = prepare_embedding_input("")

    assert isinstance(result, dict)
    assert result["is_valid"] is False
    assert result["text"] is None
    assert result["error_code"] == "EMPTY_INPUT"
    assert result["error_message"] is not None


def test_prepare_embedding_input_returns_plain_dict_for_valid_input():
    result = prepare_embedding_input("  Hello WORLD  ")

    assert result == {
        "is_valid": True,
        "text": "hello world",
        "error_code": None,
        "error_message": None,
    }


def test_prepare_embedding_input_returns_plain_dict_for_oversized_input():
    result = prepare_embedding_input(
        "a" * 51,
        max_chars=50,
    )

    assert result["is_valid"] is False
    assert result["text"] is None
    assert result["error_code"] == "INPUT_TOO_LARGE"