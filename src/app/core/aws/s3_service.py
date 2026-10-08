"""Amazon S3 Storage Service for Saarthi AI.

Handles secure policy document archives, scanned claims, and session attachments
with encryption at rest, path sanitization, and presigned access URLs.
"""

import os
from pathlib import PurePosixPath
from typing import Any

from botocore.exceptions import ClientError

from app.core.aws.session import (
    get_aioboto3_session,
    get_aws_region,
    get_boto3_session,
    has_aws_credentials,
)
from app.observability.logging import get_logger

logger = get_logger("aws.s3")


def sanitize_s3_key(key: str) -> str:
    """Sanitize S3 key to prevent directory traversal and invalid characters.

    Enforces PurePosixPath normalization, strips leading slashes, and eliminates '..' components.
    """
    clean_parts = [
        p.strip("/") for p in PurePosixPath(key.strip()).parts
        if p.strip("/") not in ("..", ".", "")
    ]
    return "/".join(clean_parts)


class S3Service:
    """Asynchronous Amazon S3 storage service."""

    def __init__(
        self,
        bucket_name: str | None = None,
        region_name: str | None = None,
    ):
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME", "saarthi-dev-uploads")
        self.region_name = region_name or get_aws_region()

    def is_available(self) -> bool:
        """Check if AWS credentials and bucket configuration are active."""
        return has_aws_credentials() and bool(self.bucket_name)

    async def upload_bytes(
        self,
        data: bytes,
        key: str,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
        bucket_name: str | None = None,
    ) -> dict[str, Any]:
        """Upload byte buffer to S3 with AES256 server-side encryption.

        Returns:
            Dictionary containing s3_uri, bucket, key, and size_bytes.
        """
        target_bucket = bucket_name or self.bucket_name
        clean_key = sanitize_s3_key(key)

        if not self.is_available():
            logger.info(
                "AWS_S3_UPLOAD_MOCK_FALLBACK",
                bucket=target_bucket,
                key=clean_key,
                reason="AWS credentials or bucket not active; storing in mock mode",
            )
            return {
                "bucket": target_bucket,
                "key": clean_key,
                "s3_uri": f"s3://{target_bucket}/{clean_key}",
                "size_bytes": len(data),
                "status": "mock_saved",
            }

        try:
            session = get_aioboto3_session(region_name=self.region_name)
            async with session.client("s3") as s3_client:
                extra_args: dict[str, Any] = {
                    "ContentType": content_type,
                    "ServerSideEncryption": "AES256",
                }
                if metadata:
                    extra_args["Metadata"] = metadata

                await s3_client.put_object(
                    Bucket=target_bucket,
                    Key=clean_key,
                    Body=data,
                    **extra_args,
                )

                s3_uri = f"s3://{target_bucket}/{clean_key}"
                logger.info(
                    "AWS_S3_UPLOAD_SUCCESS",
                    s3_uri=s3_uri,
                    size_bytes=len(data),
                )
                return {
                    "bucket": target_bucket,
                    "key": clean_key,
                    "s3_uri": s3_uri,
                    "size_bytes": len(data),
                    "status": "uploaded",
                }

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in (
                "InvalidAccessKeyId",
                "UnrecognizedClientException",
                "AccessDenied",
                "SignatureDoesNotMatch",
                "NoSuchBucket",
            ):
                logger.warning(
                    "AWS_S3_UPLOAD_CREDENTIAL_FALLBACK",
                    error_code=error_code,
                    bucket=target_bucket,
                    key=clean_key,
                    detail="Active AWS credentials cannot access bucket; using simulated upload",
                )
                return {
                    "bucket": target_bucket,
                    "key": clean_key,
                    "s3_uri": f"s3://{target_bucket}/{clean_key}",
                    "size_bytes": len(data),
                    "status": "mock_saved",
                }
            logger.error("AWS_S3_UPLOAD_CLIENT_ERROR", bucket=target_bucket, key=clean_key, error=str(e))
            raise
        except Exception as e:
            logger.error("AWS_S3_UPLOAD_FAILED", bucket=target_bucket, key=clean_key, error=str(e))
            raise

    async def download_bytes(
        self,
        key: str,
        bucket_name: str | None = None,
    ) -> bytes:
        """Download object content bytes from S3."""
        target_bucket = bucket_name or self.bucket_name
        clean_key = sanitize_s3_key(key)

        session = get_aioboto3_session(region_name=self.region_name)
        async with session.client("s3") as s3_client:
            response = await s3_client.get_object(Bucket=target_bucket, Key=clean_key)
            async with response["Body"] as stream:
                content = await stream.read()
                logger.info("AWS_S3_DOWNLOAD_SUCCESS", bucket=target_bucket, key=clean_key, size_bytes=len(content))
                return content

    def generate_presigned_url(
        self,
        key: str,
        bucket_name: str | None = None,
        expiration_seconds: int = 3600,
    ) -> str:
        """Generate a time-limited presigned URL for downloading an S3 document."""
        target_bucket = bucket_name or self.bucket_name
        clean_key = sanitize_s3_key(key)

        if not self.is_available():
            # Return placeholder in local dev
            return f"http://localhost:8000/api/v1/mock-storage/{target_bucket}/{clean_key}"

        session = get_boto3_session(region_name=self.region_name)
        s3_client = session.client("s3")
        try:
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": target_bucket, "Key": clean_key},
                ExpiresIn=expiration_seconds,
            )
            return url
        except Exception as e:
            logger.error("AWS_S3_PRESIGNED_URL_FAILED", key=clean_key, error=str(e))
            return ""

    async def file_exists(self, key: str, bucket_name: str | None = None) -> bool:
        """Check if an object exists in S3."""
        target_bucket = bucket_name or self.bucket_name
        clean_key = sanitize_s3_key(key)
        if not self.is_available():
            return False

        try:
            session = get_aioboto3_session(region_name=self.region_name)
            async with session.client("s3") as s3_client:
                await s3_client.head_object(Bucket=target_bucket, Key=clean_key)
                return True
        except ClientError:
            return False

    async def upload_policy_document(
        self,
        doc_id: str,
        filename: str,
        content: bytes,
        content_type: str = "application/pdf",
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Convenience method to upload master policy contract to S3."""
        safe_filename = os.path.basename(filename)
        s3_key = f"policies/{doc_id}/{safe_filename}"
        return await self.upload_bytes(
            data=content,
            key=s3_key,
            content_type=content_type,
            metadata=metadata,
        )

    async def upload_session_attachment(
        self,
        session_id: str,
        filename: str,
        content: bytes,
        content_type: str = "application/pdf",
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Convenience method to archive in-chat session attachments in S3."""
        safe_filename = os.path.basename(filename)
        s3_key = f"session_attachments/{session_id}/{safe_filename}"
        return await self.upload_bytes(
            data=content,
            key=s3_key,
            content_type=content_type,
            metadata=metadata,
        )

    async def check_health(self) -> dict[str, Any]:
        """Perform non-destructive S3 probe for system health check."""
        if not self.is_available():
            return {"status": "inactive", "detail": "AWS credentials or bucket not configured"}

        try:
            session = get_aioboto3_session(region_name=self.region_name)
            async with session.client("s3") as s3_client:
                await s3_client.head_bucket(Bucket=self.bucket_name)
                return {"status": "healthy", "bucket": self.bucket_name, "region": self.region_name}
        except Exception as e:
            return {"status": "unhealthy", "bucket": self.bucket_name, "error": str(e)}


_s3_service: S3Service | None = None


def get_s3_service() -> S3Service:
    """Get singleton S3Service instance."""
    global _s3_service
    if _s3_service is None:
        _s3_service = S3Service()
    return _s3_service
