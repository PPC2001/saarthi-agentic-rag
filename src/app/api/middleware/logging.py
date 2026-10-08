"""Request correlation ID and HTTP access logging middleware.

Injects unique X-Request-ID, binds context to logger, enforces secure response headers,
and measures request latency.
"""

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.observability.logging import (
    correlation_id_ctx,
    get_logger,
    user_id_ctx,
)

logger = get_logger("http_access")


class RequestCorrelationMiddleware(BaseHTTPMiddleware):
    """Middleware that sets up request correlation, logging, and security headers."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start_time = time.perf_counter()

        # Extract or generate X-Request-ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        correlation_token = correlation_id_ctx.set(request_id)

        # Extract user if already populated by auth
        user_id = request.headers.get("X-User-ID", "anonymous")
        user_token = user_id_ctx.set(user_id)

        # Log request entry
        client_host = request.client.host if request.client else "unknown"
        logger.info(
            "HTTP_REQUEST_STARTED",
            method=request.method,
            path=request.url.path,
            client_ip=client_host,
            request_id=request_id,
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "HTTP_REQUEST_FAILED",
                method=request.method,
                path=request.url.path,
                duration_ms=round(duration_ms, 2),
                error=str(exc),
                request_id=request_id,
            )
            correlation_id_ctx.reset(correlation_token)
            user_id_ctx.reset(user_token)
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Inject security headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        logger.info(
            "HTTP_REQUEST_COMPLETED",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
            request_id=request_id,
        )

        correlation_id_ctx.reset(correlation_token)
        user_id_ctx.reset(user_token)

        return response
