"""Global in-memory sliding-window rate limiter for PECS Evaluation API (API-009)."""

import time
from collections import deque
from threading import Lock
from typing import Optional

from fastapi import HTTPException, status


class SlidingWindowRateLimiter:
    """Global in-memory sliding-window rate limiter.

    Tracks request timestamps within a rolling window across all clients.
    """

    def __init__(self, max_requests: int = 30, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = Lock()

    def check_rate_limit(self, current_time: Optional[float] = None) -> None:
        """Enforce rate limit based on current timestamp.

        Args:
            current_time: Optional explicit timestamp override (used in tests).

        Raises:
            HTTPException(429): If request count reaches or exceeds max_requests within rolling window.
        """
        now = time.time() if current_time is None else float(current_time)
        cutoff = now - self.window_seconds

        with self._lock:
            # Evict timestamps older than the rolling window cutoff
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()

            if len(self._timestamps) >= self.max_requests:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Global rate limit exceeded. Maximum 30 requests per 60 seconds.",
                )

            self._timestamps.append(now)

    def reset(self) -> None:
        """Reset the rate limiter state (clears all recorded timestamps)."""
        with self._lock:
            self._timestamps.clear()

    def get_current_count(self, current_time: Optional[float] = None) -> int:
        """Return number of active request timestamps within rolling window."""
        now = time.time() if current_time is None else float(current_time)
        cutoff = now - self.window_seconds
        with self._lock:
            while self._timestamps and self._timestamps[0] <= cutoff:
                self._timestamps.popleft()
            return len(self._timestamps)



# Global rate limiter instance: 30 requests per 60 seconds
global_rate_limiter = SlidingWindowRateLimiter(max_requests=30, window_seconds=60.0)


def rate_limit_dependency() -> None:
    """FastAPI dependency wrapper for global rate limiting."""
    global_rate_limiter.check_rate_limit()
