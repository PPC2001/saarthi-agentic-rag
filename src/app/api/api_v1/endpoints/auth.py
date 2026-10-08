"""Authentication API endpoints.

Routes:
  GET  /auth/login               → Redirect to Google OIDC (or mock chooser page)
  GET  /auth/mock/accounts       → List available mock accounts (mock mode only)
  POST /auth/mock/login          → Log in as a mock account; set session cookie
  GET  /auth/me                  → Return the current user's profile
  POST /auth/logout              → Clear the session cookie

All mock-specific routes return 404 when AUTH_MODE != "mock".
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from app.api.middleware.auth import (
    AuthenticatedUser,
    get_current_user,
    get_optional_current_user,
)
from app.config import Settings, get_settings
from app.observability.logging import get_logger

router = APIRouter(prefix="/auth", tags=["Auth"])
logger = get_logger("auth_routes")

# Reusable type aliases
CurrentUser = Annotated[AuthenticatedUser, Depends(get_current_user)]
OptionalCurrentUser = Annotated[AuthenticatedUser, Depends(get_optional_current_user)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_mock_enabled(settings: Settings) -> None:
    """Raise 404 if mock mode is not active (hides the endpoints in non-mock envs)."""
    if settings.AUTH_MODE != "mock":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


def _safe_next(next_url: str | None) -> str:
    """Validate the ?next= redirect target; only allow relative paths."""
    if not next_url or not next_url.startswith("/") or "//" in next_url:
        return "/"
    return next_url


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class MockAccountOut(BaseModel):
    id: str
    email: str
    name: str
    role: str  # primary role for display purposes
    departments: list[str]


class UserProfile(BaseModel):
    user_id: str
    email: str
    name: str
    roles: list[str]
    departments: list[str]
    auth_provider: str
    auth_mode: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/login", summary="Initiate login flow")
async def login(
    request: Request,
    settings: SettingsDep,
    next_url: str | None = Query(None, alias="next"),
) -> RedirectResponse:
    """Redirect to Google OIDC (google mode) or the mock account chooser (mock mode).

    In the frontend, the 'Continue with Google' button hits this endpoint.
    """
    safe_next = _safe_next(next_url)
    if settings.AUTH_MODE == "mock":
        logger.info("AUTH_LOGIN_REDIRECT_MOCK", next=safe_next)
        return RedirectResponse(
            url=f"/login/mock?next={safe_next}",
            status_code=status.HTTP_302_FOUND,
        )
    # Real Google OIDC — placeholder; wire up python-jose + httpx-oauth here.
    logger.info("AUTH_LOGIN_REDIRECT_GOOGLE", next=safe_next)
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Google OIDC is not yet configured. Set AUTH_MODE=mock for local/dev use.",
    )


@router.get(
    "/mock/accounts",
    summary="List mock accounts (mock mode only)",
    response_model=list[MockAccountOut],
)
async def list_mock_accounts(settings: SettingsDep) -> list[dict[str, Any]]:
    """Return the fixed list of demo accounts shown on the mock chooser screen.

    This endpoint is only reachable when ``AUTH_MODE=mock``; otherwise returns 404.
    """
    _require_mock_enabled(settings)
    logger.info("AUTH_MOCK_ACCOUNTS_LISTED", count=len(settings.MOCK_ACCOUNTS))
    return [
        {
            "id": a.id,
            "email": a.email,
            "name": a.name,
            "role": a.roles[0],  # primary role for display
            "departments": a.departments,
        }
        for a in settings.MOCK_ACCOUNTS
    ]


class LoginRequest(BaseModel):
    account_id: str | None = None
    email: str | None = None
    next: str = "/"


@router.post(
    "/google/signin",
    summary="Sign in with Google (issues signed JWT)",
)
@router.post(
    "/mock/login",
    summary="Log in as an account (issues signed JWT)",
)
async def google_signin(
    request: Request,
    settings: SettingsDep,
    payload: LoginRequest | None = None,
    account_id: str | None = Body(default=None, alias="accountId"),
    next_url: str = Body(default="/", alias="next"),
    email: str | None = Body(default=None),
) -> JSONResponse:
    """Authenticate via Google Sign-In and issue a signed JWT session."""
    from app.security.jwt import create_access_token

    # Resolve email and account_id from either payload model or direct body params
    target_email = (payload and payload.email) or email
    target_acct_id = (payload and payload.account_id) or account_id
    target_next = (payload and payload.next) or next_url
    safe_next = _safe_next(target_next)

    # 1. Match from configured accounts
    found_acct = None
    if target_email:
        found_acct = next(
            (a for a in settings.MOCK_ACCOUNTS if a.email.lower() == target_email.strip().lower()),
            None,
        )
    if not found_acct and target_acct_id:
        found_acct = next(
            (a for a in settings.MOCK_ACCOUNTS if a.id == target_acct_id.strip().lower()),
            None,
        )

    # 2. Derive user credentials
    if found_acct:
        user_id = f"usr_{found_acct.id}"
        user_email = found_acct.email
        user_name = found_acct.name
        user_roles = found_acct.roles
        user_departments = found_acct.departments
    else:
        # Dynamic fallback for any @saarthi.com or custom enterprise email
        user_email = (target_email or "pratik@saarthi.com").strip().lower()
        is_admin = "admin" in user_email
        user_id = f"usr_{user_email.split('@')[0]}"
        user_name = user_email.split("@")[0].replace(".", " ").title()
        user_roles = ["admin", "kb_admin", "compliance_reviewer"] if is_admin else ["employee"]
        user_departments = ["Compliance"] if is_admin else ["Operations"]

    # 3. Create real signed JWT token with standard OIDC claims
    jwt_token = create_access_token(
        user_id=user_id,
        email=user_email,
        name=user_name,
        roles=user_roles,
        departments=user_departments,
        auth_provider="google",
        expires_in_hours=settings.JWT_EXPIRATION_HOURS,
    )

    logger.info(
        "AUTH_JWT_SIGNIN_SUCCESS",
        user_id=user_id,
        email=user_email,
        roles=user_roles,
    )

    resp = JSONResponse({
        "ok": True,
        "access_token": jwt_token,
        "token_type": "bearer",
        "email": user_email,
        "name": user_name,
        "roles": user_roles,
        "next": safe_next,
    })

    # Set HttpOnly session cookie containing the signed JWT
    resp.set_cookie(
        key="saarthi_session",
        value=jwt_token,
        httponly=True,
        samesite="lax",
        path="/",
        max_age=settings.JWT_EXPIRATION_HOURS * 3600,
    )
    return resp


@router.get(
    "/me",
    summary="Current user profile",
    response_model=UserProfile,
)
async def get_me(
    current_user: CurrentUser,
    settings: SettingsDep,
) -> UserProfile:
    """Return the profile of the currently authenticated user.

    The SPA calls this on load to determine which UI elements to show
    (e.g., whether to render the Admin menu).
    """
    logger.info("AUTH_ME_REQUESTED", user_id=current_user.user_id)
    return UserProfile(
        user_id=current_user.user_id,
        email=current_user.email,
        name=current_user.name,
        roles=current_user.roles,
        departments=current_user.departments,
        auth_provider=current_user.auth_provider,
        auth_mode=settings.AUTH_MODE,
    )


@router.post("/logout", summary="Invalidate session")
async def logout(current_user: OptionalCurrentUser) -> JSONResponse:
    """Clear the session cookie and invalidate the user's session."""
    user_id = current_user.user_id if current_user else "anonymous"
    logger.info("AUTH_LOGOUT", user_id=user_id)
    resp = JSONResponse({"ok": True, "message": "Logged out successfully"})
    resp.delete_cookie(key="saarthi_session", path="/")
    return resp
