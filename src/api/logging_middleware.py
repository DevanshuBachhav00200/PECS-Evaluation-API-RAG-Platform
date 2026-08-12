"""Request and response logging middleware for PECS Evaluation API (API-011)."""

import itertools
import json
import logging
import time
from pathlib import Path
from threading import Lock
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# Dedicated log directory and file path
LOG_DIR = Path("logs")
LOG_FILE_PATH = LOG_DIR / "api_requests.log"

logger = logging.getLogger("pecs.api.request")


def setup_request_logger() -> None:
    """Ensure dedicated log directory exists and FileHandler is attached."""
    if not LOG_DIR.exists():
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Avoid duplicate handlers if setup is invoked multiple times
    has_file_handler = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "").endswith("api_requests.log")
        for h in logger.handlers
    )

    if not has_file_handler:
        file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)


setup_request_logger()


class RequestCounter:
    """Thread-safe auto-incrementing request log ID counter."""

    def __init__(self, start: int = 1):
        self._start = start
        self._counter = itertools.count(start)
        self._lock = Lock()

    def next_id(self) -> int:
        with self._lock:
            return next(self._counter)

    def reset(self, start: int = 1) -> None:
        with self._lock:
            self._start = start
            self._counter = itertools.count(start)


request_counter = RequestCounter(start=1)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware logging exactly one traceable line per request with an auto-incrementing log ID."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        log_id = request_counter.next_id()
        start_time = time.time()

        request.state.log_id = log_id

        try:
            response = await call_next(request)
            duration_ms = round((time.time() - start_time) * 1000, 2)
            status_code = response.status_code
        except Exception as exc:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            status_code = 500
            log_record = {
                "log_id": log_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "client_host": request.client.host if request.client else None,
                "error": str(exc),
            }
            logger.error("%s", json.dumps(log_record, sort_keys=True))
            raise exc

        log_record = {
            "log_id": log_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "client_host": request.client.host if request.client else None,
        }
        logger.info("%s", json.dumps(log_record, sort_keys=True))

        response.headers["X-Log-ID"] = str(log_id)
        return response
