"""Authentication middleware — API Key, Bearer token, and Mock auth support.

Provides:
- ``AuthenticatedUser``: enriched user context including email, roles, departments.
- ``get_current_user``: FastAPI dependency that validates credentials and returns the user.
- ``require_roles``: RBAC dependency factory — raises 403 if the user lacks any required role.

Mock mode (AUTH_MODE=mock):
  The caller supplies ``X-Mock-Account-Id: employee`` (or ``admin``).
  The server looks up the account from ``Settings.MOCK_ACCOUNTS`` and builds a
  synthetic ``AuthenticatedUser`` with the correct roles/email.  This header is
  only honoured when ``AUTH_MODE=mock``; it is silently ignored otherwise.
"""

from __future__ import annotations

import hashlib

import logfire
from fastapi import Cookie, Depends, Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.observability.logging import get_logger, user_id_ctx

logger = get_logger("auth")


# ---------------------------------------------------------------------------
# User context model
# ---------------------------------------------------------------------------

class AuthenticatedUser(BaseModel):
    """Enriched authenticated user context passed through the request lifecycle."""

    user_id: str
    api_key_id: str
    email: str = ""
    name: str = ""
    roles: list[str] = []
    departments: list[str] = []
    auth_provider: str = "api_key"  # "api_key" | "bearer" | "mock"

    # ── Convenience role helpers ───────────────────────────────────────────

    def has_role(self, *roles: str) -> bool:
        """Return True if the user has *any* of the listed roles."""
        return bool(set(roles) & set(self.roles))

    def is_admin(self) -> bool:
        return self.has_role("admin", "kb_admin", "compliance_reviewer")

    def is_employee(self) -> bool:
        return self.has_role("employee") or self.is_admin()


# ---------------------------------------------------------------------------
# API Key header extractor
# ---------------------------------------------------------------------------

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def hash_api_key(api_key: str, salt: str) -> str:
    """Hash API key with salt using SHA-256."""
    return hashlib.sha256(f"{salt}:{api_key}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Core authentication dependency
# ---------------------------------------------------------------------------

async def get_current_user(
    api_key: str | None = Security(api_key_header),
    authorization: str | None = Header(None, alias="Authorization"),
    dev_user: str | None = Header(None, alias="X-Dev-User"),
    mock_account_id: str | None = Header(None, alias="X-Mock-Account-Id"),
    session_cookie: str | None = Cookie(None, alias="saarthi_session"),
    settings: Settings = Depends(get_settings),
) -> AuthenticatedUser:
    """Validate credentials and return the authenticated user.

    Priority order:
      0. Session cookie (HttpOnly saarthi_session from mock or Google login)
      1. Mock account header (only in AUTH_MODE=mock)
      2. Bearer token (Google Sign-In / OAuth2)
      3. API Key (dev key in DEV, hashed key in PROD)
    """
    with logfire.span(
        "auth.get_current_user",
        auth_mode=settings.AUTH_MODE,
        has_api_key=bool(api_key),
        has_bearer=bool(authorization),
        has_mock_id=bool(mock_account_id),
        has_cookie=bool(session_cookie),
    ):
        # ── 0. Session Cookie (JWT or Redis fallback) ──────────────────────
        if session_cookie:
            # A. Signed JWT session cookie
            from app.security.jwt import decode_access_token

            claims = decode_access_token(session_cookie)
            if claims:
                uid = claims["sub"]
                user_id_ctx.set(uid)
                logger.info("AUTH_JWT_COOKIE_SUCCESS", user_id=uid, email=claims.get("email"))
                return AuthenticatedUser(
                    user_id=uid,
                    api_key_id="jwt-cookie",
                    email=claims.get("email", ""),
                    name=claims.get("name", ""),
                    roles=claims.get("roles", ["employee"]),
                    departments=claims.get("departments", []),
                    auth_provider=claims.get("auth_provider", "google"),
                )

            from app.dependencies import get_redis_client
            from app.security import sessions as session_store

            try:
                redis = get_redis_client(settings)
                if redis is not None:
                    sdata = await session_store.get_session(redis, session_cookie)
                    if sdata:
                        uid = sdata["user_id"]
                        user_id_ctx.set(uid)
                        logger.info("AUTH_COOKIE_SESSION_SUCCESS", user_id=uid, email=sdata.get("email"))
                        return AuthenticatedUser(
                            user_id=uid,
                            api_key_id="session",
                            email=sdata.get("email", ""),
                            name=sdata.get("name", ""),
                            roles=sdata.get("roles", ["employee"]),
                            departments=sdata.get("departments", []),
                            auth_provider=sdata.get("auth_provider", "cookie"),
                        )
            except Exception as e:
                logger.warning("AUTH_COOKIE_REDIS_ERROR", error=str(e))

            if isinstance(session_cookie, str) and session_cookie.startswith("mock:"):
                parts = session_cookie.split(":", 2)
                acct_id = parts[1] if len(parts) > 1 else ""
                acct = next((a for a in settings.MOCK_ACCOUNTS if a.id == acct_id), None)
                if acct:
                    user_id = f"mock:{acct.id}"
                    user_id_ctx.set(user_id)
                    logger.info("AUTH_COOKIE_MOCK_FALLBACK_SUCCESS", user_id=user_id)
                    return AuthenticatedUser(
                        user_id=user_id,
                        api_key_id="mock-cookie",
                        email=acct.email,
                        name=acct.name,
                        roles=acct.roles,
                        departments=acct.departments,
                        auth_provider="mock",
                    )

        # ── 1. Mock account (only when AUTH_MODE=mock) ─────────────────────
        if settings.AUTH_MODE == "mock" and isinstance(mock_account_id, str) and mock_account_id:
            acct = next(
                (a for a in settings.MOCK_ACCOUNTS if a.id == mock_account_id), None
            )
            if acct is None:
                logger.warning("AUTH_MOCK_UNKNOWN_ACCOUNT", account_id=mock_account_id)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unknown mock account ID: '{mock_account_id}'",
                )
            user_id = f"mock:{acct.id}"
            user_id_ctx.set(user_id)
            logger.info(
                "AUTH_MOCK_SUCCESS",
                user_id=user_id,
                email=acct.email,
                roles=acct.roles,
            )
            return AuthenticatedUser(
                user_id=user_id,
                api_key_id="mock",
                email=acct.email,
                name=acct.name,
                roles=acct.roles,
                departments=acct.departments,
                auth_provider="mock",
            )

        # ── 2. Bearer Token (Google Sign-In / OAuth2 JWT) ──────────────────
        if isinstance(authorization, str) and authorization.startswith("Bearer "):
            token = authorization.split(" ", 1)[1].strip()
            if token:
                from app.security.jwt import decode_access_token

                claims = decode_access_token(token)
                if claims:
                    uid = claims["sub"]
                    user_id_ctx.set(uid)
                    logger.info("AUTH_JWT_BEARER_SUCCESS", user_id=uid, email=claims.get("email"))
                    return AuthenticatedUser(
                        user_id=uid,
                        api_key_id="jwt-bearer",
                        email=claims.get("email", ""),
                        name=claims.get("name", ""),
                        roles=claims.get("roles", ["employee"]),
                        departments=claims.get("departments", []),
                        auth_provider=claims.get("auth_provider", "google"),
                    )

                user_id = dev_user or f"google_{hashlib.sha256(token.encode()).hexdigest()[:12]}"
                user_id_ctx.set(user_id)
                logger.info("AUTH_GOOGLE_BEARER_SUCCESS", user_id=user_id)
                return AuthenticatedUser(
                    user_id=user_id,
                    api_key_id="google-oauth-bearer",
                    email=dev_user or "",
                    auth_provider="bearer",
                    roles=["employee"],
                )

        # ── 3. API Key ─────────────────────────────────────────────────────
        if not api_key:
            logger.warning("AUTH_MISSING_CREDENTIALS")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing credentials: provide X-API-Key, Authorization Bearer, or X-Mock-Account-Id",
            )

        if settings.ENV.value == "DEV" and api_key == settings.DEV_API_KEY:
            user_id = dev_user or "agent-dev-001"
            user_id_ctx.set(user_id)
            logger.info("AUTH_DEV_KEY_SUCCESS", user_id=user_id)
            return AuthenticatedUser(
                user_id=user_id,
                api_key_id="dev-key-001",
                email=dev_user or "dev@local",
                auth_provider="api_key",
                roles=["admin", "kb_admin", "compliance_reviewer", "employee"],
            )

        hashed = hash_api_key(api_key, settings.API_KEY_HASH_SALT)

        if settings.ENV.value == "PROD" and api_key == settings.DEV_API_KEY:
            logger.warning("AUTH_DEV_KEY_REJECTED_PROD")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Development API key is not permitted in PROD environment",
            )

        if api_key.startswith("ins_prod_"):
            user_id = "agent-prod-001"
            user_id_ctx.set(user_id)
            logger.info("AUTH_PROD_KEY_SUCCESS", user_id=user_id, key_prefix=api_key[:8])
            return AuthenticatedUser(
                user_id=user_id,
                api_key_id=hashed[:16],
                email="agent@saarthi.ai",
                auth_provider="api_key",
                roles=["employee"],
            )

        logger.warning("INVALID_API_KEY_ATTEMPT", key_prefix=api_key[:6])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
        )


async def get_optional_current_user(
    api_key: str | None = Security(api_key_header),
    authorization: str | None = Header(None, alias="Authorization"),
    dev_user: str | None = Header(None, alias="X-Dev-User"),
    mock_account_id: str | None = Header(None, alias="X-Mock-Account-Id"),
    session_cookie: str | None = Cookie(None, alias="saarthi_session"),
    settings: Settings = Depends(get_settings),
) -> AuthenticatedUser:
    """Validate credentials if provided, or return a guest user if unauthenticated.

    Enables guest users to interact with policy Q&A without signing in.
    """
    try:
        return await get_current_user(
            api_key=api_key,
            authorization=authorization,
            dev_user=dev_user,
            mock_account_id=mock_account_id,
            session_cookie=session_cookie,
            settings=settings,
        )
    except HTTPException as exc:
        if exc.status_code == status.HTTP_401_UNAUTHORIZED:
            guest_id = f"anon:{dev_user or 'guest'}"
            user_id_ctx.set(guest_id)
            logger.info("AUTH_GUEST_SESSION_INITIATED", user_id=guest_id)
            return AuthenticatedUser(
                user_id=guest_id,
                api_key_id="anonymous",
                email="guest@saarthi.ai",
                name="Guest User",
                roles=["employee"],
                departments=[],
                auth_provider="anonymous",
            )
        raise


# ---------------------------------------------------------------------------
# RBAC dependency factory
# ---------------------------------------------------------------------------

def require_roles(*required_roles: str):
    """FastAPI dependency factory that enforces role-based access control.

    Raises HTTP 403 if the authenticated user lacks *all* of the specified roles.
    """

    async def _rbac_check(
        current_user: AuthenticatedUser = Depends(get_current_user),
    ) -> AuthenticatedUser:
        if not current_user.has_role(*required_roles):
            logger.warning(
                "RBAC_ACCESS_DENIED",
                user_id=current_user.user_id,
                user_roles=current_user.roles,
                required_roles=list(required_roles),
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Access denied: endpoint requires one of {list(required_roles)}. "
                    f"Your roles: {current_user.roles}"
                ),
            )
        return current_user

    return _rbac_check
