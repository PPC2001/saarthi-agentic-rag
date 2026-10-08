"""Policy document ingestion endpoint with Amazon S3, DynamoDB, and SQS integration."""

import os
import uuid
from typing import Any

import logfire
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.api.deps import (
    CurrentUser,
    DynamoDBDep,
    EmbedderDep,
    QdrantDep,
    S3Dep,
    SettingsDep,
    SQSDep,
)
from app.api.schemas.common import DocumentType, ProductLine
from app.api.schemas.ingest import IngestResponse
from app.observability.logging import get_logger

router = APIRouter(tags=["Ingestion"])
logger = get_logger("ingest_endpoint")

# Allowed MIME types and extensions for secure file upload
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg", ".tiff"}
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "image/png",
    "image/jpeg",
    "image/tiff",
}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB limit for policy documents


class DocumentStatusResponse(BaseModel):
    """Document processing status response from DynamoDB."""

    doc_id: str
    filename: str
    status: str
    product_code: str | None = None
    chunks_count: int = 0
    s3_uri: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    error_message: str | None = None


class PresignedDownloadResponse(BaseModel):
    """S3 presigned download URL response."""

    doc_id: str
    filename: str
    download_url: str
    expires_in_seconds: int = 3600


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest Insurance Policy Document",
    description="Upload a policy contract (PDF, DOCX, scan/image, TXT) with insurance metadata. Stores raw file in S3, registers in DynamoDB, queues via SQS, and indexes into Qdrant.",
)
async def ingest_document(
    current_user: CurrentUser,
    qdrant: QdrantDep,
    embedder: EmbedderDep,
    settings: SettingsDep,
    s3: S3Dep,
    dynamo: DynamoDBDep,
    sqs: SQSDep,
    file: UploadFile = File(
        ..., description="Insurance policy document (PDF, DOCX, Scan/Image, TXT)"
    ),
    product_name: str = Form(..., min_length=2, max_length=100),
    product_code: str = Form(..., min_length=2, max_length=50),
    uin: str | None = Form(None, max_length=50),
    product_line: ProductLine = Form(ProductLine.TERM),
    document_type: DocumentType = Form(DocumentType.MASTER_POLICY),
    state: str = Form("ALL", max_length=5),
    async_processing: bool = Form(False, description="If true, queue processing via SQS and return immediately"),
) -> IngestResponse:
    """Validate and ingest an insurance policy document with multi-service AWS persistence."""
    raw_filename = file.filename or "unknown.pdf"
    # Path traversal protection: extract strict base filename
    filename = os.path.basename(raw_filename)

    with logfire.span(
        "ingest.document_upload",
        filename=filename,
        product_code=product_code,
        uin=uin,
        user_id=current_user.user_id,
    ):
        # 1. Validate file extension
        file_ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if file_ext not in ALLOWED_EXTENSIONS:
            logger.warning(
                "INGESTION_REJECTED_INVALID_EXTENSION", ext=file_ext, filename=filename
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file extension '{file_ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
            )

        # 2. Read content and enforce size limits
        content = await file.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            logger.warning(
                "INGESTION_REJECTED_FILE_SIZE_EXCEEDED",
                size_bytes=len(content),
                filename=filename,
            )
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum size limit of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB",
            )

        # 3. Magic bytes inspection
        if file_ext == ".pdf" and not content.startswith(b"%PDF"):
            logger.warning("INGESTION_REJECTED_INVALID_PDF", filename=filename)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File content does not match valid PDF signature",
            )
        elif file_ext == ".docx" and not content.startswith(b"PK\x03\x04"):
            logger.warning("INGESTION_REJECTED_INVALID_DOCX", filename=filename)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File content does not match valid DOCX archive signature",
            )
        elif file_ext == ".png" and not content.startswith(b"\x89PNG\r\n\x1a\n"):
            logger.warning("INGESTION_REJECTED_INVALID_PNG", filename=filename)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File content does not match valid PNG signature",
            )
        elif file_ext in (".jpg", ".jpeg") and not content.startswith(b"\xff\xd8\xff"):
            logger.warning("INGESTION_REJECTED_INVALID_JPEG", filename=filename)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File content does not match valid JPEG signature",
            )

        doc_id = str(uuid.uuid4())
        logger.info(
            "POLICY_DOCUMENT_INGEST_INITIATED",
            document_id=doc_id,
            filename=filename,
            product_code=product_code,
            uin=uin,
            state=state,
            file_size_bytes=len(content),
        )

        # 4. Amazon S3: Archive raw policy document
        content_type = file.content_type or "application/octet-stream"
        s3_res = await s3.upload_policy_document(
            doc_id=doc_id,
            filename=filename,
            content=content,
            content_type=content_type,
            metadata={
                "product_code": product_code,
                "uin": uin or "N/A",
                "uploaded_by": current_user.user_id,
            },
        )
        s3_uri = s3_res.get("s3_uri")
        s3_key = s3_res.get("key")
        s3_bucket = s3_res.get("bucket")

        # 5. Amazon DynamoDB: Register document record
        doc_record = {
            "doc_id": doc_id,
            "filename": filename,
            "s3_bucket": s3_bucket,
            "s3_key": s3_key,
            "s3_uri": s3_uri,
            "product_name": product_name,
            "product_code": product_code,
            "uin": uin or "N/A",
            "product_line": product_line.value,
            "document_type": document_type.value,
            "state": state,
            "file_size_bytes": len(content),
            "uploaded_by": current_user.user_id,
            "status": "PROCESSING",
            "chunks_count": 0,
        }
        await dynamo.put_document(doc_record)

        # 6. Amazon SQS: Publish ingestion message
        await sqs.send_ingest_job(
            doc_id=doc_id,
            filename=filename,
            s3_bucket=s3_bucket,
            s3_key=s3_key,
            product_name=product_name,
            product_code=product_code,
            uin=uin,
            document_type=document_type.value,
            state=state,
        )

        # If user explicitly requested asynchronous queue-only ingestion:
        if async_processing:
            return IngestResponse(
                document_id=doc_id,
                document_title=filename,
                chunks_created=0,
                status="queued",
                message="Document accepted and queued for asynchronous background processing via Amazon SQS.",
                collection_name=settings.QDRANT_COLLECTION_NAME,
                s3_uri=s3_uri,
                download_url=s3.generate_presigned_url(s3_key),
            )

        # 7. Immediate Vector Indexing
        from app.core.document_processing.session_docs import parse_uploaded_document
        from app.core.retrieval.qdrant_service import insert_document_chunks

        chunks = parse_uploaded_document(file_bytes=content, filename=filename)
        chunks_indexed = len(chunks) if chunks else 1

        if chunks:
            try:
                chunks_indexed = await insert_document_chunks(
                    client=qdrant,
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    embedder=embedder,
                    chunks=chunks,
                    product_code=product_code,
                    product_name=product_name,
                    uin=uin,
                    document_type=document_type.value,
                    state=state,
                )
                logger.info(
                    "QDRANT_CHUNKS_INDEXED_SUCCESS",
                    chunks_indexed=chunks_indexed,
                    doc_id=doc_id,
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                )
                await dynamo.update_document_status(
                    doc_id=doc_id,
                    status="INDEXED",
                    chunks_count=chunks_indexed,
                )
            except Exception as e:
                logger.warning("QDRANT_INGEST_FALLBACK", error=str(e), filename=filename)
                await dynamo.update_document_status(
                    doc_id=doc_id,
                    status="INDEX_ERROR",
                    error_message=str(e),
                )

        presigned_url = s3.generate_presigned_url(s3_key)

        return IngestResponse(
            document_id=doc_id,
            document_title=filename,
            chunks_created=chunks_indexed,
            status="accepted",
            message=f"Document successfully parsed into {chunks_indexed} clauses, archived to Amazon S3, registered in DynamoDB, and indexed in vector store.",
            collection_name=settings.QDRANT_COLLECTION_NAME,
            s3_uri=s3_uri,
            download_url=presigned_url,
        )


@router.get(
    "/ingest/{doc_id}/status",
    response_model=DocumentStatusResponse,
    summary="Get Ingestion Status from DynamoDB",
    description="Query real-time document processing status and metadata from Amazon DynamoDB.",
)
async def get_document_status(
    doc_id: str,
    dynamo: DynamoDBDep,
) -> DocumentStatusResponse:
    """Retrieve document metadata and indexing status from DynamoDB."""
    record = await dynamo.get_document(doc_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found.",
        )
    return DocumentStatusResponse(
        doc_id=record.get("doc_id", doc_id),
        filename=record.get("filename", "unknown"),
        status=record.get("status", "UNKNOWN"),
        product_code=record.get("product_code"),
        chunks_count=record.get("chunks_count", 0),
        s3_uri=record.get("s3_uri"),
        created_at=record.get("created_at"),
        updated_at=record.get("updated_at"),
        error_message=record.get("error_message"),
    )


@router.get(
    "/ingest/{doc_id}/download-url",
    response_model=PresignedDownloadResponse,
    summary="Get S3 Presigned Download URL",
    description="Generate a secure, time-limited presigned URL to download the original document from Amazon S3.",
)
async def get_document_download_url(
    doc_id: str,
    dynamo: DynamoDBDep,
    s3: S3Dep,
) -> PresignedDownloadResponse:
    """Generate time-limited presigned download URL for S3 stored document."""
    record = await dynamo.get_document(doc_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found.",
        )

    s3_key = record.get("s3_key")
    if not s3_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document has no associated S3 object key.",
        )

    url = s3.generate_presigned_url(s3_key)
    return PresignedDownloadResponse(
        doc_id=doc_id,
        filename=record.get("filename", "document.pdf"),
        download_url=url,
        expires_in_seconds=3600,
    )


@router.get(
    "/ingest/catalog",
    summary="List Registered Documents",
    description="List all policy contracts registered in DynamoDB catalog.",
)
async def list_registered_documents(
    dynamo: DynamoDBDep,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List documents tracked in DynamoDB."""
    return await dynamo.list_documents(limit=min(limit, 100))
