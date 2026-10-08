"""Application info and metadata endpoints."""

from fastapi import APIRouter

from app.api.deps import SettingsDep

router = APIRouter(tags=["Info"])


@router.get("/")
@router.get("/info")
async def get_service_info(settings: SettingsDep) -> dict[str, str]:
    """Root and service metadata endpoint."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENV.value,
        "docs": "/docs",
        "health": "/health",
    }
