"""Environment-based settings and configuration validation for PECS Evaluation API (API-012)."""

import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

# Resolve repository root path
REPO_ROOT = Path(__file__).resolve().parents[2]

# Automatically load environment variables from .env if present at repo root
load_dotenv(REPO_ROOT / ".env")


class ConfigurationError(RuntimeError):
    """Raised when required environment settings are missing or invalid."""
    pass


class Settings:
    """Environment-based API configuration settings (API-012).

    Loads and validates required configuration from environment variables:
    - ANTHROPIC_API_KEY: Credential for evaluator LLM calls (Member 4 RAG engine).
    - API_KEY: Server-side secret key for client x-api-key authentication (API-007).
    - CHROMA_DB_PATH: Path to the ChromaDB vector database storage directory.
    """

    def __init__(self, env: Optional[Dict[str, str]] = None) -> None:
        self._env = env

    @property
    def env_source(self) -> Dict[str, str]:
        """Return target environment dict source."""
        return self._env if self._env is not None else dict(os.environ)

    @property
    def anthropic_api_key(self) -> str:
        key = self.env_source.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise ConfigurationError(
                "Missing required environment variable: 'ANTHROPIC_API_KEY'."
            )
        return key

    @property
    def api_key(self) -> str:
        key = self.env_source.get("API_KEY", "").strip()
        if not key:
            raise ConfigurationError(
                "Missing required environment variable: 'API_KEY'."
            )
        return key

    @property
    def chroma_db_path(self) -> str:
        path = self.env_source.get("CHROMA_DB_PATH", "").strip()
        if not path:
            raise ConfigurationError(
                "Missing required environment variable: 'CHROMA_DB_PATH'."
            )
        return path

    def validate(self) -> None:
        """Validate that all required environment settings are present.

        Raises:
            ConfigurationError: If any required setting is missing or empty.
        """
        _ = self.anthropic_api_key
        _ = self.api_key
        _ = self.chroma_db_path

    def __repr__(self) -> str:
        """Return a string representation without exposing secret values."""
        chroma_val = self.env_source.get("CHROMA_DB_PATH", "<unset>")
        return (
            f"<Settings anthropic_api_key='***' api_key='***' "
            f"chroma_db_path={chroma_val!r}>"
        )


def get_settings(env: Optional[Dict[str, str]] = None) -> Settings:
    """Factory function returning verified application Settings instance."""
    settings = Settings(env=env)
    return settings
