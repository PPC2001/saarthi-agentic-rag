"""Common utilities and helper functions for API handlers and requests."""

import os
import re
from typing import Any

from fastapi import HTTPException, Request, UploadFile, status


def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded file name to prevent path traversal and shell injection.

    Strips directory separators, removes non-whitelisted characters, and enforces length limits.
    """
    clean_name = os.path.basename(filename).strip()
    # Replace any character that is not alphanumeric, dot, underscore, or hyphen
    clean_name = re.sub(r"[^a-zA-Z0-9._-]", "_", clean_name)
    # Ensure it's not empty
    if not clean_name:
        clean_name = "document"
    return clean_name[:150]


def validate_file_upload(
    file: UploadFile,
    allowed_extensions: set[str],
    max_size_bytes: int = 25 * 1024 * 1024,
) -> str:
    """Validate an uploaded document against allowed extensions and size limit.

    Returns the sanitized filename if valid, otherwise raises HTTPException.
    """
    raw_name = file.filename or "unknown"
    file_ext = "." + raw_name.rsplit(".", 1)[-1].lower() if "." in raw_name else ""

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file format '{file_ext}'. Allowed formats: "
                f"{', '.join(sorted(allowed_extensions))}"
            ),
        )

    # Check file size if available via content length or seek
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size exceeds maximum allowed limit of {max_size_bytes // (1024 * 1024)} MB "
                f"(received {file_size / (1024 * 1024):.1f} MB)."
            ),
        )

    return sanitize_filename(raw_name)


def extract_client_ip(request: Request) -> str:
    """Extract client IP address from proxy headers or direct connection."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


def format_error_detail(code: str, message: str, field: str | None = None) -> dict[str, Any]:
    """Format standardized error detail dictionary."""
    result: dict[str, Any] = {"code": code, "message": message}
    if field:
        result["field"] = field
    return result
