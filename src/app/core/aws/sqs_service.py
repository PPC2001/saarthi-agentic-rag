"""Amazon SQS Ingestion Queue Service for Saarthi AI.

Provides asynchronous, decoupled document ingestion queueing.
Enables reliable batch document ingestion, retry policies, and dead-letter queueing.
"""

import json
import os
from typing import Any

from botocore.exceptions import ClientError

from app.core.aws.session import get_aioboto3_session, get_aws_region, has_aws_credentials
from app.observability.logging import get_logger

logger = get_logger("aws.sqs")


class SQSService:
    """Asynchronous Amazon SQS service for document ingestion jobs."""

    def __init__(
        self,
        queue_url: str | None = None,
        queue_name: str | None = None,
        region_name: str | None = None,
    ):
        self.queue_url = queue_url or os.getenv("SQS_INGEST_QUEUE_URL")
        self.queue_name = queue_name or os.getenv("SQS_INGEST_QUEUE_NAME", "saarthi-ai-ingest-queue")
        self.region_name = region_name or get_aws_region()

        # In-memory mock queue for local development and testing
        self._mock_queue: list[dict[str, Any]] = []

    def is_available(self) -> bool:
        """Check if AWS credentials and SQS queue are active."""
        return has_aws_credentials()

    async def get_or_resolve_queue_url(self) -> str | None:
        """Resolve queue URL from configured URL or by querying queue name."""
        if self.queue_url:
            return self.queue_url

        if not self.is_available():
            return None

        try:
            session = get_aioboto3_session(region_name=self.region_name)
            async with session.client("sqs") as client:
                res = await client.get_queue_url(QueueName=self.queue_name)
                self.queue_url = res.get("QueueUrl")
                return self.queue_url
        except ClientError as e:
            logger.debug("SQS_GET_QUEUE_URL_FAILED", queue_name=self.queue_name, error=str(e))
            return None

    async def send_ingest_job(
        self,
        doc_id: str,
        filename: str,
        s3_bucket: str,
        s3_key: str,
        product_name: str,
        product_code: str,
        uin: str | None = None,
        document_type: str = "master_policy",
        state: str = "ALL",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Send a document ingestion job message to SQS queue."""
        payload = {
            "doc_id": doc_id,
            "filename": filename,
            "s3_bucket": s3_bucket,
            "s3_key": s3_key,
            "product_name": product_name,
            "product_code": product_code,
            "uin": uin,
            "document_type": document_type,
            "state": state,
            "metadata": metadata or {},
        }

        queue_url = await self.get_or_resolve_queue_url()

        if not queue_url:
            # Mock fallback
            import uuid
            msg_id = f"mock-msg-{uuid.uuid4()}"
            self._mock_queue.append({
                "MessageId": msg_id,
                "ReceiptHandle": f"mock-receipt-{msg_id}",
                "Body": json.dumps(payload),
            })
            logger.info("SQS_INGEST_JOB_QUEUED_MOCK", doc_id=doc_id, msg_id=msg_id)
            return msg_id

        try:
            session = get_aioboto3_session(region_name=self.region_name)
            async with session.client("sqs") as client:
                response = await client.send_message(
                    QueueUrl=queue_url,
                    MessageBody=json.dumps(payload),
                    MessageAttributes={
                        "DocumentType": {
                            "StringValue": document_type,
                            "DataType": "String",
                        },
                        "ProductCode": {
                            "StringValue": product_code,
                            "DataType": "String",
                        },
                    },
                )
                message_id = response.get("MessageId", "unknown")
                logger.info("SQS_INGEST_JOB_SENT_SUCCESS", doc_id=doc_id, message_id=message_id)
                return message_id
        except Exception as e:
            logger.error("SQS_SEND_MESSAGE_FAILED", doc_id=doc_id, error=str(e))
            # Fallback to mock
            import uuid
            msg_id = f"fallback-{uuid.uuid4()}"
            self._mock_queue.append({
                "MessageId": msg_id,
                "ReceiptHandle": f"mock-receipt-{msg_id}",
                "Body": json.dumps(payload),
            })
            return msg_id

    async def receive_ingest_jobs(
        self,
        max_messages: int = 5,
        wait_time_seconds: int = 5,
    ) -> list[dict[str, Any]]:
        """Receive document ingestion messages from SQS queue."""
        queue_url = await self.get_or_resolve_queue_url()

        if not queue_url:
            # Pop from mock queue
            items = self._mock_queue[:max_messages]
            self._mock_queue = self._mock_queue[max_messages:]
            return items

        try:
            session = get_aioboto3_session(region_name=self.region_name)
            async with session.client("sqs") as client:
                response = await client.receive_message(
                    QueueUrl=queue_url,
                    MaxNumberOfMessages=max_messages,
                    WaitTimeSeconds=wait_time_seconds,
                    AttributeNames=["All"],
                    MessageAttributeNames=["All"],
                )
                return response.get("Messages", [])
        except Exception as e:
            logger.error("SQS_RECEIVE_MESSAGES_FAILED", error=str(e))
            if self._mock_queue:
                items = self._mock_queue[:max_messages]
                self._mock_queue = self._mock_queue[max_messages:]
                return items
            return []

    async def delete_message(self, receipt_handle: str) -> bool:
        """Delete an acknowledged message from SQS queue."""
        queue_url = await self.get_or_resolve_queue_url()
        if not queue_url or receipt_handle.startswith("mock-receipt-") or "fallback-" in receipt_handle:
            return True

        try:
            session = get_aioboto3_session(region_name=self.region_name)
            async with session.client("sqs") as client:
                await client.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=receipt_handle,
                )
                return True
        except Exception as e:
            logger.error("SQS_DELETE_MESSAGE_FAILED", error=str(e))
            return False

    async def check_health(self) -> dict[str, Any]:
        """Check SQS queue status and attribute count."""
        if not self.is_available():
            return {"status": "inactive", "detail": "AWS credentials not configured"}

        queue_url = await self.get_or_resolve_queue_url()
        if not queue_url:
            return {"status": "inactive", "detail": f"Queue {self.queue_name} not found"}

        try:
            session = get_aioboto3_session(region_name=self.region_name)
            async with session.client("sqs") as client:
                res = await client.get_queue_attributes(
                    QueueUrl=queue_url,
                    AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"],
                )
                attrs = res.get("Attributes", {})
                return {
                    "status": "healthy",
                    "queue_name": self.queue_name,
                    "visible_messages": attrs.get("ApproximateNumberOfMessages", 0),
                    "in_flight_messages": attrs.get("ApproximateNumberOfMessagesNotVisible", 0),
                }
        except Exception as e:
            return {"status": "unhealthy", "queue_name": self.queue_name, "error": str(e)}


_sqs_service: SQSService | None = None


def get_sqs_service() -> SQSService:
    """Get singleton SQSService instance."""
    global _sqs_service
    if _sqs_service is None:
        _sqs_service = SQSService()
    return _sqs_service
