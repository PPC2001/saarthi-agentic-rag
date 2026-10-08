"""Unit tests for AWS services integration (S3, DynamoDB, SQS, Secrets Manager, ElastiCache)."""

import pytest

from app.core.aws.dynamodb_service import DynamoDBService
from app.core.aws.s3_service import S3Service, sanitize_s3_key
from app.core.aws.secrets_manager import get_secrets_from_manager
from app.core.aws.sqs_service import SQSService


def test_sanitize_s3_key():
    """Verify path traversal prevention on S3 keys."""
    assert sanitize_s3_key("../../../etc/passwd") == "etc/passwd"
    assert sanitize_s3_key("policies/123/../../secret.pdf") == "policies/123/secret.pdf"
    assert sanitize_s3_key("/policies/doc.pdf") == "policies/doc.pdf"
    assert sanitize_s3_key("policies/doc.pdf") == "policies/doc.pdf"


@pytest.mark.asyncio
async def test_s3_service_mock_mode():
    """Test S3Service in fallback/mock mode when AWS credentials are not set."""
    s3 = S3Service(bucket_name="test-bucket", region_name="ap-south-1")
    # In mock mode, upload_bytes returns simulated record without crashing
    res = await s3.upload_bytes(
        data=b"%PDF-1.4 test document content",
        key="policies/test-doc/contract.pdf",
        content_type="application/pdf",
    )
    assert res["bucket"] == "test-bucket"
    assert res["key"] == "policies/test-doc/contract.pdf"
    assert res["size_bytes"] == len(b"%PDF-1.4 test document content")
    assert "s3://" in res["s3_uri"]

    # Presigned URL generation
    url = s3.generate_presigned_url("policies/test-doc/contract.pdf")
    assert "contract.pdf" in url


@pytest.mark.asyncio
async def test_dynamodb_service_document_registry():
    """Test DynamoDBService document registration, status update, and retrieval."""
    dynamo = DynamoDBService(table_prefix="test-saarthi")

    doc_id = "test-doc-12345"
    record = {
        "doc_id": doc_id,
        "filename": "sample_term_policy.pdf",
        "s3_bucket": "test-bucket",
        "s3_key": f"policies/{doc_id}/sample_term_policy.pdf",
        "product_code": "TERM-01",
        "product_name": "Secure Life Term",
        "status": "PROCESSING",
    }

    # Put document
    await dynamo.put_document(record)

    # Get document
    retrieved = await dynamo.get_document(doc_id)
    assert retrieved is not None
    assert retrieved["doc_id"] == doc_id
    assert retrieved["product_code"] == "TERM-01"
    assert retrieved["status"] == "PROCESSING"

    # Update document status
    await dynamo.update_document_status(
        doc_id=doc_id,
        status="INDEXED",
        chunks_count=12,
    )

    updated = await dynamo.get_document(doc_id)
    assert updated["status"] == "INDEXED"
    assert updated["chunks_count"] == 12

    # List documents
    docs = await dynamo.list_documents(limit=10)
    assert len(docs) >= 1
    assert any(d["doc_id"] == doc_id for d in docs)


@pytest.mark.asyncio
async def test_dynamodb_service_chat_sessions():
    """Test DynamoDB chat session persistence."""
    dynamo = DynamoDBService(table_prefix="test-saarthi")
    session_id = "session-abc-999"

    # Save turn
    await dynamo.save_session_message(
        session_id=session_id,
        role="user",
        content="What is the grace period for premium payment?",
        user_id="user-1",
    )
    await dynamo.save_session_message(
        session_id=session_id,
        role="assistant",
        content="The grace period is 30 days for annual premium payments.",
        user_id="system",
        model_used="claude-sonnet",
    )

    history = await dynamo.get_session_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    assert history[1]["model_used"] == "claude-sonnet"


@pytest.mark.asyncio
async def test_sqs_service_queue():
    """Test SQSService message queueing and retrieval."""
    sqs = SQSService(queue_name="test-ingest-queue")

    doc_id = "doc-sqs-456"
    msg_id = await sqs.send_ingest_job(
        doc_id=doc_id,
        filename="health_policy.pdf",
        s3_bucket="test-bucket",
        s3_key="policies/doc-sqs-456/health_policy.pdf",
        product_name="Health Protect",
        product_code="HEALTH-99",
    )
    assert msg_id is not None

    messages = await sqs.receive_ingest_jobs(max_messages=1, wait_time_seconds=1)
    assert len(messages) == 1
    assert "health_policy.pdf" in messages[0]["Body"]

    # Delete message
    receipt_handle = messages[0]["ReceiptHandle"]
    success = await sqs.delete_message(receipt_handle)
    assert success is True


def test_secrets_manager_graceful_without_credentials():
    """Verify Secrets Manager returns empty dict gracefully without throwing unhandled errors."""
    secrets = get_secrets_from_manager("non-existent-secret-name")
    assert isinstance(secrets, dict)
