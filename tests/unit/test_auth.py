"""Unit tests for authentication and role-based access control."""

import pytest
from fastapi import HTTPException

from app.api.middleware.auth import (
    AuthenticatedUser,
    get_current_user,
    hash_api_key,
)
from app.config import Settings


def test_hash_api_key():
    """Verify SHA-256 key hashing."""
    salt = "test-salt"
    key = "secret-key-123"
    hashed = hash_api_key(key, salt)
    assert len(hashed) == 64
    assert hashed == hash_api_key(key, salt)


@pytest.mark.asyncio
async def test_get_current_user_missing_key():
    """Verify missing API key raises 401."""
    settings = Settings()
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(api_key=None, dev_user=None, settings=settings)
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_valid_dev_key():
    """Verify valid dev key authenticates user in DEV environment."""
    settings = Settings()
    user = await get_current_user(
        api_key=settings.DEV_API_KEY,
        dev_user="agent-user-01",
        settings=settings,
    )
    assert user.user_id == "agent-user-01"
    assert user.api_key_id == "dev-key-001"


def test_authenticated_user_model():
    """Verify AuthenticatedUser schema."""
    user = AuthenticatedUser(user_id="u1", api_key_id="k1")
    assert user.user_id == "u1"
    assert user.api_key_id == "k1"
