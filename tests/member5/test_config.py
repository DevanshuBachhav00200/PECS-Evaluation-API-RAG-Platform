"""Unit tests for API-012 environment-based settings."""

import pytest

from src.api import ConfigurationError, Settings, get_settings


def test_config_all_required_variables_present():
    """Verify Settings initializes and validates correctly when all required variables are present."""
    env = {
        "ANTHROPIC_API_KEY": "test-anthropic-key-dummy",
        "API_KEY": "test-server-api-key-dummy",
        "CHROMA_DB_PATH": "Database/pels_vector_db",
    }
    settings = Settings(env=env)
    settings.validate()

    assert settings.anthropic_api_key == "test-anthropic-key-dummy"
    assert settings.api_key == "test-server-api-key-dummy"
    assert settings.chroma_db_path == "Database/pels_vector_db"


def test_config_missing_anthropic_api_key():
    """Verify ConfigurationError is raised when ANTHROPIC_API_KEY is missing or empty."""
    env = {
        "API_KEY": "test-server-api-key-dummy",
        "CHROMA_DB_PATH": "Database/pels_vector_db",
    }
    settings = Settings(env=env)

    with pytest.raises(ConfigurationError) as exc_info:
        _ = settings.anthropic_api_key
    assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    with pytest.raises(ConfigurationError):
        settings.validate()


def test_config_missing_api_key():
    """Verify ConfigurationError is raised when API_KEY is missing or empty."""
    env = {
        "ANTHROPIC_API_KEY": "test-anthropic-key-dummy",
        "CHROMA_DB_PATH": "Database/pels_vector_db",
    }
    settings = Settings(env=env)

    with pytest.raises(ConfigurationError) as exc_info:
        _ = settings.api_key
    assert "API_KEY" in str(exc_info.value)

    with pytest.raises(ConfigurationError):
        settings.validate()


def test_config_missing_chroma_db_path():
    """Verify ConfigurationError is raised when CHROMA_DB_PATH is missing or empty."""
    env = {
        "ANTHROPIC_API_KEY": "test-anthropic-key-dummy",
        "API_KEY": "test-server-api-key-dummy",
    }
    settings = Settings(env=env)

    with pytest.raises(ConfigurationError) as exc_info:
        _ = settings.chroma_db_path
    assert "CHROMA_DB_PATH" in str(exc_info.value)

    with pytest.raises(ConfigurationError):
        settings.validate()


def test_config_secrets_never_exposed_in_repr():
    """Verify that sensitive API key values are masked and never exposed in string representations."""
    env = {
        "ANTHROPIC_API_KEY": "super-secret-anthropic-key-12345",
        "API_KEY": "super-secret-server-api-key-67890",
        "CHROMA_DB_PATH": "Database/pels_vector_db",
    }
    settings = Settings(env=env)
    repr_str = repr(settings)

    assert "super-secret-anthropic-key-12345" not in repr_str
    assert "super-secret-server-api-key-67890" not in repr_str
    assert "***" in repr_str
    assert "Database/pels_vector_db" in repr_str


def test_get_settings_factory():
    """Verify get_settings factory helper function."""
    env = {
        "ANTHROPIC_API_KEY": "key1",
        "API_KEY": "key2",
        "CHROMA_DB_PATH": "path1",
    }
    s = get_settings(env=env)
    assert s.anthropic_api_key == "key1"
    assert s.api_key == "key2"
    assert s.chroma_db_path == "path1"
