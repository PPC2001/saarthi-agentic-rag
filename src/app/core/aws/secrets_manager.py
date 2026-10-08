"""AWS Secrets Manager integration for Saarthi AI.

Loads secrets dynamically from AWS Secrets Manager at startup and injects them
into the environment, enabling secure production key rotation and zero-hardcoded secrets.
"""

import json
import os
from typing import Any

from botocore.exceptions import ClientError

from app.core.aws.session import get_boto3_session, has_aws_credentials
from app.observability.logging import get_logger

logger = get_logger("aws.secrets_manager")


def get_secrets_from_manager(
    secret_name: str,
    region_name: str | None = None,
) -> dict[str, Any]:
    """Retrieve secrets dictionary from AWS Secrets Manager.

    Args:
        secret_name: Name or ARN of the secret in AWS Secrets Manager.
        region_name: AWS region name (defaults to active region).

    Returns:
        Dictionary of key-value secrets, or empty dict if not found or on error.
    """
    if not has_aws_credentials():
        logger.debug(
            "AWS_SECRETS_SKIPPED_NO_CREDENTIALS",
            secret_name=secret_name,
            reason="No AWS credentials found in environment",
        )
        return {}

    try:
        session = get_boto3_session(region_name=region_name)
        client = session.client("secretsmanager")

        response = client.get_secret_value(SecretId=secret_name)
        secret_string = response.get("SecretString")

        if not secret_string:
            logger.warning(
                "AWS_SECRET_STRING_EMPTY",
                secret_name=secret_name,
            )
            return {}

        try:
            secrets = json.loads(secret_string)
            if isinstance(secrets, dict):
                logger.info(
                    "AWS_SECRETS_LOADED_SUCCESS",
                    secret_name=secret_name,
                    keys_count=len(secrets),
                )
                return secrets
            return {"SECRET_VALUE": secrets}
        except json.JSONDecodeError:
            # Secret is raw plaintext string
            return {"RAW_SECRET": secret_string}

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        if error_code == "ResourceNotFoundException":
            logger.info(
                "AWS_SECRET_NOT_FOUND",
                secret_name=secret_name,
                detail="Secret does not exist in AWS Secrets Manager. Using environment variables.",
            )
        elif error_code in ("AccessDeniedException", "UnrecognizedClientException"):
            logger.warning(
                "AWS_SECRET_ACCESS_DENIED",
                secret_name=secret_name,
                error_code=error_code,
                detail="IAM permission missing or credentials expired.",
            )
        else:
            logger.warning(
                "AWS_SECRETS_FETCH_ERROR",
                secret_name=secret_name,
                error=str(e),
            )
        return {}
    except Exception as e:
        logger.error("AWS_SECRETS_UNEXPECTED_ERROR", secret_name=secret_name, error=str(e))
        return {}


def fetch_and_apply_secrets(secret_name: str, region_name: str | None = None) -> int:
    """Fetch secrets from Secrets Manager and apply them to os.environ if not already set.

    Returns:
        Number of secrets injected into os.environ.
    """
    secrets = get_secrets_from_manager(secret_name=secret_name, region_name=region_name)
    applied_count = 0
    for key, value in secrets.items():
        if value is not None and not os.getenv(key):
            os.environ[key] = str(value)
            applied_count += 1

    if applied_count > 0:
        logger.info(
            "AWS_SECRETS_APPLIED_TO_ENV",
            applied_count=applied_count,
            secret_name=secret_name,
        )
    return applied_count
