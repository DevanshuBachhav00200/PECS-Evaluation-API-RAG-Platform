"""Authentication dependency for PECS Evaluation API (API-007)."""

import os
import secrets
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.api.config import ConfigurationError, Settings

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """Validate request x-api-key header against server API_KEY environment variable.

    Raises:
        HTTPException(500): If server API_KEY environment variable is not configured.
        HTTPException(403): If request x-api-key header is missing or invalid.
    """
    try:
        settings = Settings()
        server_key = settings.api_key
    except ConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key authentication is not configured on the server.",
        ) from exc


    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key header is missing.",
        )

    if not secrets.compare_digest(api_key, server_key):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return api_key
