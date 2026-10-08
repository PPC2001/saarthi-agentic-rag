"""``/meta`` endpoint — runtime discovery for the SPA.

Returns the API version, auth mode, environment, and feature flags so the
frontend never needs per-environment build variables.  Called on app load
before authentication so the SPA knows whether to show the mock chooser or
the Google button.

Spec §6.4: GET /api/meta → {apiVersion, minClientVersion, gitSha, env, authMode, features}
"""

from __future__ import annotations

import os

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.observability.logging import get_logger

router = APIRouter(tags=["Meta"])
logger = get_logger("meta")


class MetaResponse(BaseModel):
    """Runtime metadata returned to the SPA on every app load."""

    api_version: str
    min_client_version: str
    git_sha: str
    env: str
    auth_mode: str
    # Feature flags — SPA hides UI elements when False
    features: dict[str, bool]


@router.get(
    "/meta",
    response_model=MetaResponse,
    summary="Runtime API metadata",
    description=(
        "Returns the API version, auth mode, and feature flags. "
        "Called on SPA load so the frontend requires no per-environment build variables."
    ),
)
async def get_meta() -> MetaResponse:
    """Return runtime metadata for SPA bootstrap.

    This endpoint is intentionally **public** (no auth required) so the SPA
    can decide which login screen to render before the user is authenticated.
    """
    settings: Settings = get_settings()
    logger.info("META_REQUESTED", env=settings.ENV.value, auth_mode=settings.AUTH_MODE)
    return MetaResponse(
        api_version=settings.APP_VERSION,
        min_client_version="0.0.0",  # bump when a breaking SPA change ships
        git_sha=os.environ.get("GIT_SHA", "dev"),
        env=settings.ENV.value.lower(),
        auth_mode=settings.AUTH_MODE,
        features={
            "mock_login": settings.AUTH_MODE == "mock",
            "web_research": False,      # not yet wired; toggle when ready
            "uploads": True,
            "admin_console": True,
            "streaming": True,
        },
    )
