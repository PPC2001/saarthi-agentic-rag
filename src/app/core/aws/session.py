"""AWS Session and credential management for Saarthi AI.

Provides reusable, shared boto3 and aioboto3 sessions configured with
proper connection pooling, retry policies, and environment credential handling.
Automatically picks up AWS export credentials:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_SESSION_TOKEN
- AWS_REGION / AWS_DEFAULT_REGION
"""

import os

import aioboto3
import boto3
from botocore.config import Config

from app.observability.logging import get_logger

logger = get_logger("aws.session")

_boto3_session: boto3.Session | None = None
_aioboto3_session: aioboto3.Session | None = None

# Default retry and connection configuration for resilient AWS SDK calls
DEFAULT_BOTO_CONFIG = Config(
    retries={
        "max_attempts": 3,
        "mode": "standard",
    },
    connect_timeout=5,
    read_timeout=30,
)


def has_aws_credentials() -> bool:
    """Check if valid AWS credentials or environment variables are available."""
    # Fast check for pasted environment variables
    if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        return True

    # Fallback check through botocore credential provider chain
    try:
        session = boto3.Session()
        credentials = session.get_credentials()
        return credentials is not None and credentials.access_key is not None
    except Exception as e:
        logger.debug("AWS_CREDENTIALS_CHECK_FAILED", error=str(e))
        return False


def get_aws_region(default: str = "ap-southeast-2") -> str:
    """Resolve active AWS region from settings, environment or default."""
    try:
        from app.config import get_settings
        settings = get_settings()
        if settings.AWS_REGION:
            return settings.AWS_REGION
    except Exception:
        pass
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or default


def get_boto3_session(region_name: str | None = None) -> boto3.Session:
    """Get or create singleton synchronous boto3.Session."""
    global _boto3_session
    region = region_name or get_aws_region()
    if _boto3_session is None:
        _boto3_session = boto3.Session(region_name=region)
        logger.info("AWS_BOTO3_SESSION_INITIALIZED", region=region)
    return _boto3_session


def get_aioboto3_session(region_name: str | None = None) -> aioboto3.Session:
    """Get or create singleton asynchronous aioboto3.Session."""
    global _aioboto3_session
    region = region_name or get_aws_region()
    if _aioboto3_session is None:
        _aioboto3_session = aioboto3.Session(region_name=region)
        logger.info("AWS_AIOBOTO3_SESSION_INITIALIZED", region=region)
    return _aioboto3_session


def reset_sessions() -> None:
    """Reset cached sessions (e.g. after updating credentials)."""
    global _boto3_session, _aioboto3_session
    _boto3_session = None
    _aioboto3_session = None
    logger.info("AWS_SESSIONS_RESET")
