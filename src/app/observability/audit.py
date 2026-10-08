"""HIPAA compliance audit trail logging.

Maintains an immutable record of every policy query, document access,
and system response to meet HIPAA Security Rule and insurance regulatory requirements.
"""

import hashlib
import uuid
from datetime import UTC, datetime

import logfire
from pydantic import BaseModel, Field

from app.config import Settings
from app.observability.logging import get_logger, mask_phi

logger = get_logger("audit")


class AuditLogEntry(BaseModel):
    """Schema for HIPAA-compliant audit log entries."""

    audit_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    user_id: str = "anonymous"
    user_role: str = "insurance_agent"
    api_key_id: str = "unknown"
    action: str = "policy_query"
    query_text: str = ""
    query_hash: str = ""
    response_summary: str = ""
    documents_accessed: list[str] = Field(default_factory=list)
    model_used: str = ""
    tokens_used: dict[str, int] = Field(default_factory=lambda: {"input": 0, "output": 0})
    latency_ms: float = 0.0
    guardrail_actions: list[str] = Field(default_factory=list)
    phi_detected: bool = False
    ip_address: str = "127.0.0.1"
    source_application: str = "policy-portal"
    status_code: int = 200


class AuditLogger:
    """Records audit entries for all policy interactions."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.hipaa_mode = settings.HIPAA_MODE
        self._dynamodb_client = None

    def _get_dynamo_client(self):
        """Lazy load boto3 client for DynamoDB."""
        if self._dynamodb_client is None and self.settings.ENV.value == "PROD":
            import boto3

            self._dynamodb_client = boto3.client(
                "dynamodb",
                region_name=self.settings.AWS_REGION,
            )
        return self._dynamodb_client

    async def log_event(self, entry: AuditLogEntry) -> None:
        """Record an audit entry to the audit log."""
        with logfire.span(
            "audit.log_event",
            action=entry.action,
            user_id=entry.user_id,
            status_code=entry.status_code,
            latency_ms=entry.latency_ms,
        ):
            # Calculate SHA256 of query text for deduplication/forensics
            if entry.query_text and not entry.query_hash:
                entry.query_hash = hashlib.sha256(entry.query_text.encode("utf-8")).hexdigest()

            # In HIPAA mode, mask any PHI in the query_text logged to stdout
            safe_query_text = mask_phi(entry.query_text) if self.hipaa_mode else entry.query_text

            logger.info(
                "AUDIT_RECORD",
                audit_id=entry.audit_id,
                user_id=entry.user_id,
                user_role=entry.user_role,
                action=entry.action,
                query_hash=entry.query_hash,
                safe_query=safe_query_text[:100],
                documents_count=len(entry.documents_accessed),
                model_used=entry.model_used,
                latency_ms=entry.latency_ms,
                status_code=entry.status_code,
                phi_detected=entry.phi_detected,
            )

            # Persist audit entry to DynamoDB audit-log table
            try:
                from app.core.aws.dynamodb_service import get_dynamodb_service
                dynamo = get_dynamodb_service()
                await dynamo.put_audit_entry({
                    "audit_id": entry.audit_id,
                    "timestamp": entry.timestamp,
                    "user_id": entry.user_id,
                    "user_role": entry.user_role,
                    "action": entry.action,
                    "query_hash": entry.query_hash,
                    "model_used": entry.model_used,
                    "latency_ms": str(entry.latency_ms),
                    "status_code": str(entry.status_code),
                    "phi_detected": entry.phi_detected,
                    "ip_address": entry.ip_address,
                })
            except Exception as e:
                # Never crash the user request on audit write failure, but alarm!
                logger.error(
                    "Failed to write audit entry to DynamoDB",
                    error=str(e),
                    audit_id=entry.audit_id,
                )
