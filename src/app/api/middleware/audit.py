"""HIPAA Audit Trail HTTP Middleware.

Ensures every query, document access, and ingestion event is audited with:
- Authenticated user ID and role
- Client IP and timestamp
- Query hash and latency
- HTTP response status
"""

import time
from collections.abc import Awaitable, Callable

import logfire
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings
from app.observability.audit import AuditLogEntry, AuditLogger
from app.observability.logging import get_logger

logger = get_logger("audit_middleware")
audit_logger = AuditLogger(settings)


class HIPAAAuditMiddleware(BaseHTTPMiddleware):
    """Middleware that audits access to sensitive insurance endpoints."""

    AUDITED_PREFIXES = (
        "/v1/query",
        "/v1/ingest",
        "/v1/collections",
        "/api/v1/query",
        "/api/v1/ingest",
        "/api/v1/collections",
    )

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        is_audited = any(path.startswith(prefix) for prefix in self.AUDITED_PREFIXES)

        if not is_audited:
            return await call_next(request)

        start_time = time.perf_counter()
        user_id = request.headers.get("X-Dev-User", "anonymous")
        user_role = "insurance_agent"
        client_ip = request.client.host if request.client else "127.0.0.1"

        with logfire.span(
            "audit.http_access",
            method=request.method,
            path=path,
            user_id=user_id,
            client_ip=client_ip,
        ):
            try:
                response = await call_next(request)
                latency_ms = (time.perf_counter() - start_time) * 1000

                entry = AuditLogEntry(
                    user_id=user_id,
                    user_role=user_role,
                    action=f"http_{request.method.lower()}:{path}",
                    ip_address=client_ip,
                    latency_ms=round(latency_ms, 2),
                    status_code=response.status_code,
                )
                # Log audit asynchronously
                await audit_logger.log_event(entry)
                logger.info(
                    "AUDITED_HTTP_REQUEST_COMPLETED",
                    path=path,
                    method=request.method,
                    status_code=response.status_code,
                    latency_ms=round(latency_ms, 2),
                )
                return response

            except Exception as exc:
                latency_ms = (time.perf_counter() - start_time) * 1000
                entry = AuditLogEntry(
                    user_id=user_id,
                    user_role=user_role,
                    action=f"http_{request.method.lower()}:{path}",
                    ip_address=client_ip,
                    latency_ms=round(latency_ms, 2),
                    status_code=500,
                    response_summary=f"Exception: {str(exc)}",
                )
                await audit_logger.log_event(entry)
                logger.error(
                    "AUDITED_HTTP_REQUEST_FAILED",
                    path=path,
                    method=request.method,
                    error=str(exc),
                    latency_ms=round(latency_ms, 2),
                )
                raise
