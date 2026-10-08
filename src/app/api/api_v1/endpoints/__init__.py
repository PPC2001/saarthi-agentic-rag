"""API v1 endpoint handlers."""

from app.api.api_v1.endpoints import (
    admin,
    auth,
    chat,
    collections,
    health,
    info,
    ingest,
    meta,
    query,
    session_attachment,
)

__all__ = [
    "admin",
    "auth",
    "chat",
    "collections",
    "health",
    "info",
    "ingest",
    "meta",
    "query",
    "session_attachment",
]
