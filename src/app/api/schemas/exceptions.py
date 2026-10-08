"""Domain Exceptions and RFC 7807 Problem Details with Trace ID Links.

Inspired by enterprise production microservice standards (Cinépolis architecture):
- Strongly-typed domain exceptions inheriting from BaseDomainException
- RFC 7807 (Problem Details for HTTP APIs) compliant payload structure
- Dynamic trace correlation link injection ('instance' field pointing to Logfire / X-Ray trace)
"""

from datetime import UTC, datetime
from typing import Any

from fastapi import status
from pydantic import BaseModel, Field

from app.observability.logging import correlation_id_ctx


class ProblemDetails(BaseModel):
    """RFC 7807 Problem Details representation with distributed trace link."""

    title: str = Field(..., description="Short, human-readable summary of the problem type")
    status: int = Field(..., description="HTTP status code")
    detail: str = Field(..., description="Detailed explanation specific to this occurrence")
    code: str = Field(..., description="Application-specific business error code")
    instance: str = Field(
        ..., description="URI / Trace reference identifying the specific occurrence"
    )
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Contextual debugging metadata"
    )


class BaseDomainException(Exception):  # noqa: N818
    """Base class for domain-level business and operational exceptions."""

    title: str = "InternalDomainError"
    code: str = "ERR_DOMAIN_INTERNAL"
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR

    def __init__(
        self,
        detail: str,
        title: str | None = None,
        code: str | None = None,
        status_code: int | None = None,
        metadata: dict[str, Any] | None = None,
        instance: str | None = None,
    ):
        super().__init__(detail)
        self.detail = detail
        if title:
            self.title = title
        if code:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.metadata = metadata or {}

        # Resolve trace correlation link
        req_id = correlation_id_ctx.get() or "unknown-trace"
        self.instance = instance or f"/traces/{req_id}"

    def to_problem_details(self) -> ProblemDetails:
        """Convert exception to RFC 7807 ProblemDetails payload."""
        return ProblemDetails(
            title=self.title,
            status=self.status_code,
            detail=self.detail,
            code=self.code,
            instance=self.instance,
            metadata=self.metadata,
        )


class PolicyNotFoundException(BaseDomainException):
    """Raised when a requested policy, document, or clause cannot be found."""

    title = "PolicyNotFound"
    code = "ERR_POLICY_NOT_FOUND"
    status_code = status.HTTP_404_NOT_FOUND


class PolicyDocumentParseException(BaseDomainException):
    """Raised when an uploaded policy PDF, DOCX, or scan fails structural clause extraction."""

    title = "DocumentParseFailure"
    code = "ERR_DOC_PARSE_FAILED"
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT


class ActuarialCalculationException(BaseDomainException):
    """Raised when deterministic actuarial math parameters are invalid or incompatible."""

    title = "ActuarialCalculationError"
    code = "ERR_ACTUARIAL_CALCULATION"
    status_code = status.HTTP_400_BAD_REQUEST


class DisallowedDomainException(BaseDomainException):
    """Raised when regulatory web search attempts to fetch from a non-allowlisted domain."""

    title = "DisallowedWebDomain"
    code = "ERR_REGULATORY_DOMAIN_DISALLOWED"
    status_code = status.HTTP_403_FORBIDDEN


class PromptInjectionException(BaseDomainException):
    """Raised when input guardrails detect prompt injection or adversarial jailbreaks."""

    title = "PromptInjectionDetected"
    code = "ERR_SECURITY_PROMPT_INJECTION"
    status_code = status.HTTP_400_BAD_REQUEST


class OffTopicQueryException(BaseDomainException):
    """Raised when user query is outside Indian life insurance domain scope."""

    title = "OffTopicQueryRestricted"
    code = "ERR_GUARDRAIL_OFF_TOPIC"
    status_code = status.HTTP_400_BAD_REQUEST


class RateLimitExceededException(BaseDomainException):
    """Raised when a client exceeds allowed request frequency."""

    title = "RateLimitExceeded"
    code = "ERR_RATE_LIMIT_EXCEEDED"
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
