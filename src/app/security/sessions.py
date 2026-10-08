"""Redis-backed session store for Saarthi AI.

Replaces the plaintext cookie with a signed, opaque session token stored in
Redis.  The token is a URL-safe random string; the session data is JSON-
serialised in Redis with an idle TTL (sliding window) and an absolute TTL.

In a full production deployment this module is used by:
  - Mock login: after picking an account, POST /auth/mock/login creates a session
  - Real Google OIDC (future): the OIDC callback creates the same session shape
  - Every authenticated request: get_session() validates the cookie and returns
    the user context

Design:
  - Session key: ``saarthi:session:<sid>`` (Redis hash)
  - ``sid`` is a 32-byte URL-safe random token (256 bits of entropy)
  - Session data stored: user_id, email, name, roles, departments, auth_provider,
    created_at, last_seen (updated on every access for idle timeout)
"""

from __future__ import annotations

import json
import secrets
import time
from typing import Any

from redis.asyncio import Redis

from app.observability.logging import get_logger

logger = get_logger("sessions")

SESSION_KEY_PREFIX = "saarthi:session:"
COOKIE_NAME = "saarthi_session"


def _key(sid: str) -> str:
    return f"{SESSION_KEY_PREFIX}{sid}"


async def create_session(
    redis: Redis,
    *,
    user_id: str,
    email: str,
    name: str,
    roles: list[str],
    departments: list[str],
    auth_provider: str,
    idle_seconds: int = 28_800,   # 8 hours
    absolute_seconds: int = 86_400,  # 24 hours
) -> str:
    """Create a new session and return the opaque session ID.

    The caller should set this as an HttpOnly cookie on the response.
    """
    sid = secrets.token_urlsafe(32)
    now = time.time()
    data: dict[str, Any] = {
        "user_id": user_id,
        "email": email,
        "name": name,
        "roles": roles,
        "departments": departments,
        "auth_provider": auth_provider,
        "created_at": now,
        "last_seen": now,
        "absolute_expires_at": now + absolute_seconds,
    }
    await redis.set(
        _key(sid),
        json.dumps(data),
        ex=idle_seconds,  # idle TTL — refreshed on each touch
    )
    logger.info("SESSION_CREATED", user_id=user_id, auth_provider=auth_provider)
    return sid


async def get_session(
    redis: Redis,
    sid: str,
    idle_seconds: int = 28_800,
) -> dict[str, Any] | None:
    """Look up and refresh an existing session.  Returns ``None`` if expired or missing."""
    raw = await redis.get(_key(sid))
    if not raw:
        return None

    data: dict[str, Any] = json.loads(raw)
    now = time.time()

    # Absolute TTL check (even if Redis idle TTL has not fired yet)
    if now > data.get("absolute_expires_at", 0):
        await redis.delete(_key(sid))
        logger.info("SESSION_ABSOLUTE_EXPIRED", sid=sid[:8])
        return None

    # Slide the idle TTL
    data["last_seen"] = now
    await redis.set(_key(sid), json.dumps(data), ex=idle_seconds)
    return data


async def destroy_session(redis: Redis, sid: str) -> None:
    """Invalidate a session immediately (logout / offboarding)."""
    await redis.delete(_key(sid))
    logger.info("SESSION_DESTROYED", sid=sid[:8])


async def destroy_all_user_sessions(redis: Redis, user_id: str) -> int:
    """Delete all sessions belonging to a user (admin offboarding action).

    Warning: this performs a SCAN which is O(n) over all session keys.
    In a large deployment consider maintaining a ``user_id → set of sids``
    secondary index instead.
    """
    count = 0
    async for key in redis.scan_iter(f"{SESSION_KEY_PREFIX}*"):
        raw = await redis.get(key)
        if raw:
            data = json.loads(raw)
            if data.get("user_id") == user_id:
                await redis.delete(key)
                count += 1
    logger.info("SESSION_BULK_DESTROYED", user_id=user_id, count=count)
    return count
