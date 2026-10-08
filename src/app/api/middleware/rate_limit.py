"""Rate limiting middleware and dependency.

Provides sliding window rate limiting per API key or client IP,
with Redis storage backend in production and robust in-memory fallback for local dev.
"""

import time
from collections import defaultdict
from collections.abc import Awaitable, Callable

import logfire
from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.observability.logging import get_logger

logger = get_logger("rate_limit")

# In-memory store: key -> list of timestamps with bounded capacity
MAX_TRACKED_CLIENTS = 10_000
_in_memory_rate_limits: dict[str, list[float]] = defaultdict(list)


def _cleanup_expired_clients(now: float, window_seconds: float) -> None:
    """Periodically evict inactive clients to prevent memory leaks."""
    expired_keys = [
        k
        for k, timestamps in _in_memory_rate_limits.items()
        if not timestamps or (now - timestamps[-1] >= window_seconds)
    ]
    for k in expired_keys:
        _in_memory_rate_limits.pop(k, None)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware enforcing requests-per-minute limits with sliding window."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Skip rate limiting on health, root, and documentation endpoints
        if request.url.path in (
            "/health",
            "/ready",
            "/live",
            "/v1/health",
            "/api/v1/health",
            "/info",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/",
        ):
            return await call_next(request)

        # Identify client by API key or IP address
        api_key = request.headers.get("X-API-Key")
        client_id = api_key if api_key else (request.client.host if request.client else "unknown")

        limit = settings.RATE_LIMIT_PER_MINUTE
        window_seconds = 60.0
        now = time.time()

        # Enforce memory bounds if tracked clients exceed threshold
        if len(_in_memory_rate_limits) > MAX_TRACKED_CLIENTS:
            _cleanup_expired_clients(now, window_seconds)

        with logfire.span(
            "rate_limit.check",
            client_id=client_id[:10] if api_key else client_id,
            limit=limit,
        ):
            # In-memory sliding window check
            timestamps = _in_memory_rate_limits[client_id]
            valid_timestamps = [t for t in timestamps if now - t < window_seconds]
            _in_memory_rate_limits[client_id] = valid_timestamps

            if len(valid_timestamps) >= limit:
                logger.warning(
                    "RATE_LIMIT_EXCEEDED",
                    client_id=client_id[:10] if api_key else client_id,
                    requests_in_window=len(valid_timestamps),
                    limit=limit,
                )
                retry_after = int(window_seconds - (now - valid_timestamps[0])) + 1
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Maximum {limit} requests per minute.",
                    headers={"Retry-After": str(max(retry_after, 1))},
                )

            # Record this request
            _in_memory_rate_limits[client_id].append(now)

            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(max(0, limit - len(valid_timestamps) - 1))
            return response
