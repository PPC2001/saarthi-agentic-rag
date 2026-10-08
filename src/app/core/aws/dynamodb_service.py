"""Amazon DynamoDB Service for Saarthi AI.

Provides persistent, managed NoSQL storage for:
1. Policy Document Registry (saarthi-ai-documents)
2. HIPAA & Compliance Audit Trail (saarthi-ai-audit-log)
3. Persistent Chat Session History (saarthi-ai-chat-sessions)
"""

import os
import uuid
from datetime import UTC, datetime
from typing import Any

from app.core.aws.session import get_aioboto3_session, get_aws_region, has_aws_credentials
from app.observability.logging import get_logger

logger = get_logger("aws.dynamodb")


class DynamoDBService:
    """Asynchronous DynamoDB service with multi-table support."""

    def __init__(
        self,
        table_prefix: str | None = None,
        region_name: str | None = None,
    ):
        self.table_prefix = table_prefix or os.getenv("DYNAMODB_TABLE_PREFIX", "saarthi-ai")
        self.region_name = region_name or get_aws_region()

        self.table_documents = os.getenv("DYNAMODB_TABLE_DOCUMENTS", f"{self.table_prefix}-documents")
        self.table_audit = os.getenv("DYNAMODB_TABLE_AUDIT", f"{self.table_prefix}-audit-log")
        self.table_sessions = os.getenv("DYNAMODB_TABLE_SESSIONS", f"{self.table_prefix}-chat-sessions")

        # In-memory mock storage when AWS credentials are not active
        self._mock_documents: dict[str, dict[str, Any]] = {}
        self._mock_audit: list[dict[str, Any]] = []
        self._mock_sessions: dict[str, list[dict[str, Any]]] = {}

    def is_available(self) -> bool:
        """Check if AWS credentials are active."""
        return has_aws_credentials()

    # ── 1. Document Registry Operations ──────────────────────────────────────────

    async def put_document(self, record: dict[str, Any]) -> None:
        """Register policy document metadata in DynamoDB."""
        doc_id = record.get("doc_id")
        if not doc_id:
            raise ValueError("Document record must contain 'doc_id'")

        now_iso = datetime.now(UTC).isoformat()
        record.setdefault("created_at", now_iso)
        record.setdefault("updated_at", now_iso)
        record.setdefault("status", "PENDING")

        if not self.is_available():
            self._mock_documents[doc_id] = record
            logger.debug("DYNAMODB_DOC_SAVED_MOCK", doc_id=doc_id)
            return

        try:
            session = get_aioboto3_session(region_name=self.region_name)
            async with session.resource("dynamodb") as dynamo:
                table = await dynamo.Table(self.table_documents)
                await table.put_item(Item=record)
                logger.info("DYNAMODB_DOC_SAVED_SUCCESS", doc_id=doc_id, table=self.table_documents)
        except Exception as e:
            logger.warning("DYNAMODB_DOC_PUT_ERROR", doc_id=doc_id, error=str(e))
            self._mock_documents[doc_id] = record

    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Fetch document record by doc_id."""
        if not self.is_available():
            return self._mock_documents.get(doc_id)

        try:
            session = get_aioboto3_session(region_name=self.region_name)
            async with session.resource("dynamodb") as dynamo:
                table = await dynamo.Table(self.table_documents)
                response = await table.get_item(Key={"doc_id": doc_id})
                item = response.get("Item")
                return item or self._mock_documents.get(doc_id)
        except Exception as e:
            logger.warning("DYNAMODB_DOC_GET_ERROR", doc_id=doc_id, error=str(e))
            return self._mock_documents.get(doc_id)

    async def update_document_status(
        self,
        doc_id: str,
        status: str,
        chunks_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update indexing status of a document."""
        now_iso = datetime.now(UTC).isoformat()
        # Update mock
        if doc_id in self._mock_documents:
            self._mock_documents[doc_id]["status"] = status
            self._mock_documents[doc_id]["updated_at"] = now_iso
            if chunks_count is not None:
                self._mock_documents[doc_id]["chunks_count"] = chunks_count
            if error_message is not None:
                self._mock_documents[doc_id]["error_message"] = error_message

        if not self.is_available():
            return

        try:
            session = get_aioboto3_session(region_name=self.region_name)
            async with session.resource("dynamodb") as dynamo:
                table = await dynamo.Table(self.table_documents)

                update_expr = "SET #st = :status, updated_at = :updated_at"
                expr_names = {"#st": "status"}
                expr_values: dict[str, Any] = {
                    ":status": status,
                    ":updated_at": now_iso,
                }

                if chunks_count is not None:
                    update_expr += ", chunks_count = :chunks"
                    expr_values[":chunks"] = chunks_count
                if error_message is not None:
                    update_expr += ", error_message = :err"
                    expr_values[":err"] = error_message

                await table.update_item(
                    Key={"doc_id": doc_id},
                    UpdateExpression=update_expr,
                    ExpressionAttributeNames=expr_names,
                    ExpressionAttributeValues=expr_values,
                )
                logger.info("DYNAMODB_DOC_STATUS_UPDATED", doc_id=doc_id, status=status)
        except Exception as e:
            logger.warning("DYNAMODB_DOC_UPDATE_ERROR", doc_id=doc_id, error=str(e))

    async def list_documents(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recently ingested policy documents."""
        if not self.is_available():
            return list(self._mock_documents.values())[:limit]

        try:
            session = get_aioboto3_session(region_name=self.region_name)
            async with session.resource("dynamodb") as dynamo:
                table = await dynamo.Table(self.table_documents)
                response = await table.scan(Limit=limit)
                items = response.get("Items", [])
                return items or list(self._mock_documents.values())[:limit]
        except Exception as e:
            logger.warning("DYNAMODB_DOC_LIST_ERROR", error=str(e))
            return list(self._mock_documents.values())[:limit]

    # ── 2. Audit Trail Operations ────────────────────────────────────────────────

    async def put_audit_entry(self, entry: dict[str, Any]) -> None:
        """Write audit entry to DynamoDB audit table."""
        entry.setdefault("audit_id", str(uuid.uuid4()))
        timestamp = entry.get("timestamp") or datetime.now(UTC).isoformat()
        entry.setdefault("timestamp", timestamp)

        if not self.is_available():
            self._mock_audit.append(entry)
            return

        try:
            session = get_aioboto3_session(region_name=self.region_name)
            async with session.resource("dynamodb") as dynamo:
                table = await dynamo.Table(self.table_audit)
                await table.put_item(Item=entry)
        except Exception as e:
            logger.warning("DYNAMODB_AUDIT_PUT_ERROR", error=str(e))
            self._mock_audit.append(entry)

    # ── 3. Chat Session State Operations ─────────────────────────────────────────

    async def save_session_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: str = "anonymous",
        citations: list[dict[str, Any]] | None = None,
        model_used: str | None = None,
    ) -> None:
        """Persist a chat message turn to DynamoDB."""
        timestamp = datetime.now(UTC).isoformat()
        item = {
            "session_id": session_id,
            "timestamp": timestamp,
            "user_id": user_id,
            "role": role,
            "content": content,
            "citations": citations or [],
            "model_used": model_used or "unknown",
        }

        if not self.is_available():
            self._mock_sessions.setdefault(session_id, []).append(item)
            return

        try:
            session = get_aioboto3_session(region_name=self.region_name)
            async with session.resource("dynamodb") as dynamo:
                table = await dynamo.Table(self.table_sessions)
                await table.put_item(Item=item)
        except Exception as e:
            logger.warning("DYNAMODB_SESSION_SAVE_ERROR", session_id=session_id, error=str(e))
            self._mock_sessions.setdefault(session_id, []).append(item)

    async def get_session_history(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieve recent conversation history for a session."""
        if not self.is_available():
            return self._mock_sessions.get(session_id, [])[-limit:]

        try:
            from boto3.dynamodb.conditions import Key
            session = get_aioboto3_session(region_name=self.region_name)
            async with session.resource("dynamodb") as dynamo:
                table = await dynamo.Table(self.table_sessions)
                response = await table.query(
                    KeyConditionExpression=Key("session_id").eq(session_id),
                    Limit=limit,
                    ScanIndexForward=True,  # Chronological order
                )
                items = response.get("Items", [])
                return items or self._mock_sessions.get(session_id, [])[-limit:]
        except Exception as e:
            logger.warning("DYNAMODB_SESSION_QUERY_ERROR", session_id=session_id, error=str(e))
            return self._mock_sessions.get(session_id, [])[-limit:]

    async def check_health(self) -> dict[str, Any]:
        """Verify connection to DynamoDB tables."""
        if not self.is_available():
            return {"status": "inactive", "detail": "AWS credentials not configured"}

        try:
            session = get_aioboto3_session(region_name=self.region_name)
            async with session.client("dynamodb") as client:
                res = await client.describe_table(TableName=self.table_documents)
                table_status = res.get("Table", {}).get("TableStatus", "UNKNOWN")
                return {
                    "status": "healthy" if table_status == "ACTIVE" else "degraded",
                    "table_documents": self.table_documents,
                    "table_status": table_status,
                }
        except Exception as e:
            return {"status": "unhealthy", "table": self.table_documents, "error": str(e)}


_dynamodb_service: DynamoDBService | None = None


def get_dynamodb_service() -> DynamoDBService:
    """Get singleton DynamoDBService instance."""
    global _dynamodb_service
    if _dynamodb_service is None:
        _dynamodb_service = DynamoDBService()
    return _dynamodb_service
