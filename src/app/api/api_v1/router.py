"""API v1 Central Router.

Aggregates all version 1 endpoints into a unified APIRouter, modeled after
production microservice architecture.
"""

from fastapi import APIRouter

from app.api.api_v1.endpoints import (
    admin,
    auth,
    chat,
    collections,
    health,
    info,
    ingest,
    meta,
    metrics,
    query,
    session_attachment,
)

api_router = APIRouter()

# Include versioned sub-routers
api_router.include_router(meta.router)          # /meta — public, no auth
api_router.include_router(auth.router)          # /auth/*
api_router.include_router(chat.router)          # /chat/stream — SSE
api_router.include_router(query.router)         # /query — synchronous RAG
api_router.include_router(ingest.router)
api_router.include_router(session_attachment.router)
api_router.include_router(collections.router)
api_router.include_router(admin.router)
api_router.include_router(health.router)
api_router.include_router(info.router)
api_router.include_router(metrics.router)

__all__ = ["api_router"]
