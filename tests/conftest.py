"""Pytest configuration and shared test fixtures."""

import os

import pytest
from fastapi.testclient import TestClient

# Ensure test environment is DEV
os.environ["ENV"] = "DEV"
os.environ["HIPAA_MODE"] = "false"
os.environ["DEV_API_KEY"] = "dev-test-key-12345"
os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
os.environ.setdefault("AWS_REGION", "ap-southeast-2")
os.environ.setdefault("S3_BUCKET_NAME", "saarthi-dev-uploads")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-mock-key-for-tests")
os.environ.setdefault("PORTKEY_API_KEY", "test-portkey-key")

from app.config import get_settings
from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Create FastAPI TestClient instance."""
    with TestClient(app, base_url="http://127.0.0.1:8000") as c:
        yield c


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Return default authenticated headers for development agent."""
    settings = get_settings()
    return {
        "X-API-Key": settings.DEV_API_KEY,
        "X-Dev-User": "agent-test-001",
    }


@pytest.fixture
def admin_headers() -> dict[str, str]:
    """Return authenticated headers for admin role."""
    settings = get_settings()
    return {
        "X-API-Key": settings.DEV_API_KEY,
        "X-Dev-User": "admin-test-001",
    }
