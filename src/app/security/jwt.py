"""JWT Token issuance and verification using PyJWT."""
from __future__ import annotations

import time
from typing import Any

import jwt
from pydantic import BaseModel

from app.config import get_settings


class TokenClaims(BaseModel):
    sub: str
    email: str
    name: str
    roles: list[str]
    departments: list[str] = []
    auth_provider: str = "google"
    iss: str = "https://accounts.google.com"
    aud: str = "saarthi-ai"
    iat: int
    exp: int


def create_access_token(
    user_id: str,
    email: str,
    name: str,
    roles: list[str],
    departments: list[str] | None = None,
    auth_provider: str = "google",
    expires_in_hours: int | None = None,
) -> str:
    """Create a signed JWT token mirroring Google OIDC / enterprise claims."""
    settings = get_settings()
    now = int(time.time())
    ttl = (expires_in_hours or settings.JWT_EXPIRATION_HOURS) * 3600

    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "roles": roles,
        "departments": departments or [],
        "auth_provider": auth_provider,
        "iss": "https://accounts.google.com",
        "aud": "saarthi-ai",
        "iat": now,
        "exp": now + ttl,
    }

    return jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_access_token(token: str) -> dict[str, Any] | None:
    """Validate and decode a signed JWT token. Returns claims dict or None."""
    settings = get_settings()
    try:
        claims = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            audience="saarthi-ai",
        )
        return claims
    except Exception:
        # Also decode if audience is omitted
        try:
            return jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
                options={"verify_aud": False},
            )
        except Exception:
            return None
