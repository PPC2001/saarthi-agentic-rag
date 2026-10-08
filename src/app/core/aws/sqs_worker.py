"""Background worker process for consuming SQS document ingestion queue.

Pulls pending document ingestion tasks from Amazon SQS, fetches the document from
Amazon S3, extracts semantic clauses, indexes embeddings into Qdrant,
and updates status in Amazon DynamoDB.
"""

import asyncio
import json
from typing import Any

from app.config import get_settings
from app.core.aws.dynamodb_service import get_dynamodb_service
from app.core.aws.s3_service import get_s3_service
from app.core.aws.sqs_service import get_sqs_service
from app.core.document_processing.session_docs import parse_uploaded_document
from app.core.retrieval.qdrant_service import get_qdrant_client, insert_document_chunks
from app.dependencies import get_embedder
from app.observability.logging import get_logger

logger = get_logger("aws.sqs_worker")


async def process_single_message(message: dict[str, Any]) -> bool:
    """Process a single SQS ingestion task.

    Returns:
        True if successfully processed and acknowledged, False otherwise.
    """
    settings = get_settings()
    s3 = get_s3_service()
    dynamo = get_dynamodb_service()
    sqs = get_sqs_service()

    receipt_handle = message.get("ReceiptHandle")
    body_str = message.get("Body", "{}")

    try:
        payload = json.loads(body_str)
    except Exception as e:
        logger.error("SQS_INVALID_JSON_BODY", error=str(e), body=body_str[:100])
        if receipt_handle:
            await sqs.delete_message(receipt_handle)
        return False

    doc_id = payload.get("doc_id")
    filename = payload.get("filename", "unknown.pdf")
    s3_key = payload.get("s3_key")
    s3_bucket = payload.get("s3_bucket")
    product_name = payload.get("product_name", "Unknown Product")
    product_code = payload.get("product_code", "UNKNOWN")
    uin = payload.get("uin")
    document_type = payload.get("document_type", "master_policy")
    state = payload.get("state", "ALL")

    if not doc_id or not s3_key:
        logger.error("SQS_MESSAGE_MISSING_REQUIRED_FIELDS", payload=payload)
        if receipt_handle:
            await sqs.delete_message(receipt_handle)
        return False

        logger.info("SQS_PROCESSING_DOCUMENT_JOB", doc_id=doc_id, s3_key=s3_key)
        await dynamo.update_document_status(doc_id=doc_id, status="PROCESSING")

        try:
            # 1. Download document from Amazon S3
            doc_bytes = await s3.download_bytes(key=s3_key, bucket_name=s3_bucket)
            if not doc_bytes:
                raise ValueError(f"Document content empty from S3 key: {s3_key}")

            # 2. Extract clauses and chunks
            chunks = parse_uploaded_document(file_bytes=doc_bytes, filename=filename)
            chunks_count = len(chunks) if chunks else 0

            # 3. Vector Embed & Index into Qdrant
            if chunks:
                qdrant = get_qdrant_client(settings)
                embedder = get_embedder(settings)
                await insert_document_chunks(
                    client=qdrant,
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    embedder=embedder,
                    chunks=chunks,
                    product_code=product_code,
                    product_name=product_name,
                    uin=uin,
                    document_type=document_type,
                    state=state,
                )

            # 4. Update DynamoDB status to INDEXED
            await dynamo.update_document_status(
                doc_id=doc_id,
                status="INDEXED",
                chunks_count=chunks_count,
            )

            # 5. Delete message from SQS
            if receipt_handle:
                await sqs.delete_message(receipt_handle)

            logger.info(
                "SQS_DOCUMENT_INDEXED_SUCCESS",
                doc_id=doc_id,
                chunks_count=chunks_count,
                filename=filename,
            )
            return True

        except Exception as e:
            logger.error("SQS_DOCUMENT_PROCESSING_FAILED", doc_id=doc_id, error=str(e))
            await dynamo.update_document_status(
                doc_id=doc_id,
                status="FAILED",
                error_message=str(e),
            )
            return False


async def run_sqs_worker(max_iterations: int | None = None, poll_interval_seconds: int = 5) -> None:
    """Run SQS ingestion worker loop."""
    sqs = get_sqs_service()
    logger.info("SQS_WORKER_STARTED", queue_name=sqs.queue_name)

    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        iterations += 1
        try:
            messages = await sqs.receive_ingest_jobs(max_messages=5, wait_time_seconds=poll_interval_seconds)
            if messages:
                logger.info("SQS_MESSAGES_RECEIVED", count=len(messages))
                for msg in messages:
                    await process_single_message(msg)
            else:
                await asyncio.sleep(poll_interval_seconds)
        except asyncio.CancelledError:
            logger.info("SQS_WORKER_CANCELLED")
            break
        except Exception as e:
            logger.error("SQS_WORKER_ERROR", error=str(e))
            await asyncio.sleep(poll_interval_seconds)


if __name__ == "__main__":
    asyncio.run(run_sqs_worker())
