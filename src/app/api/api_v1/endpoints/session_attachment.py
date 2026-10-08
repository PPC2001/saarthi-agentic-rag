"""Session attachment upload endpoint for in-chat document RAG."""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.api.deps import OptionalCurrentUser, RedisDep, S3Dep
from app.core.document_processing.session_docs import (
    SessionDocumentStore,
    parse_uploaded_document,
)
from app.observability.logging import get_logger

router = APIRouter(prefix="/session", tags=["Session Attachments"])
logger = get_logger("session_attachments")

ALLOWED_ATTACHMENT_EXTENSIONS = {".pdf", ".docx", ".txt"}
MAX_ATTACHMENT_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB


class SessionAttachmentResponse(BaseModel):
    """Response after parsing and storing in-chat document attachment."""

    status: str = "success"
    session_id: str
    filename: str
    chunks_extracted: int
    message: str
    s3_uri: str | None = None


@router.post(
    "/attach",
    response_model=SessionAttachmentResponse,
    summary="Attach Policy Document to Chat Session",
    description="Upload a policy bond, endorsement, circular, or brochure for instant in-chat clause extraction and citations.",
)
async def attach_document(
    s3: S3Dep,
    file: UploadFile = File(..., description="Policy document (PDF, TXT, DOCX)"),
    session_id: str = Form(..., description="Unique session ID for the active chat"),
) -> SessionAttachmentResponse:
    """Parse and index an attached document into session-scoped memory with S3 backup."""
    raw_filename = file.filename or "uploaded_policy.pdf"
    import os
    filename = os.path.basename(raw_filename)
    file_ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if file_ext not in ALLOWED_ATTACHMENT_EXTENSIONS:
        logger.warning(
            "ATTACHMENT_REJECTED_INVALID_EXTENSION",
            ext=file_ext,
            filename=filename,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported attachment extension '{file_ext}'. Allowed: {list(ALLOWED_ATTACHMENT_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > MAX_ATTACHMENT_SIZE_BYTES:
        logger.warning(
            "ATTACHMENT_REJECTED_FILE_SIZE_EXCEEDED",
            size_bytes=len(content),
            filename=filename,
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Attachment exceeds maximum limit of {MAX_ATTACHMENT_SIZE_BYTES // (1024 * 1024)}MB",
        )

    # Parse document
    chunks = parse_uploaded_document(file_bytes=content, filename=filename)
    if not chunks:
        logger.error("ATTACHMENT_CLAUSE_PARSE_FAILED", filename=filename)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unable to extract text clauses from uploaded file. Please verify file content.",
        )

    # Store in session memory
    SessionDocumentStore.add_document(session_id=session_id, filename=filename, chunks=chunks)

    # Archive attachment to Amazon S3
    s3_res = await s3.upload_session_attachment(
        session_id=session_id,
        filename=filename,
        content=content,
        content_type=file.content_type or "application/octet-stream",
    )
    s3_uri = s3_res.get("s3_uri")

    logger.info(
        "ATTACHMENT_INDEXED_FOR_SESSION",
        session_id=session_id,
        filename=filename,
        chunks=len(chunks),
        size_kb=round(len(content) / 1024, 1),
        s3_uri=s3_uri,
    )

    return SessionAttachmentResponse(
        status="success",
        session_id=session_id,
        filename=filename,
        chunks_extracted=len(chunks),
        message=f"Successfully extracted {len(chunks)} clauses from '{filename}'. You can now ask questions about this document.",
        s3_uri=s3_uri,
    )


@router.delete(
    "/{session_id}/clear",
    summary="Clear Session Attachments",
    description="Release all in-memory attachments associated with a session ID.",
)
async def clear_session(session_id: str) -> dict[str, str]:
    """Clear in-memory attachments for a session."""
    SessionDocumentStore.clear_session(session_id)
    logger.info("SESSION_ATTACHMENTS_CLEARED", session_id=session_id)
    return {"status": "cleared", "session_id": session_id}


# ── Library Document Management ──────────────────────────────────────────

class LibraryDocItem(BaseModel):
    id: str
    name: str
    size: int
    uploaded_at: str
    status: str = "ready"
    chunks: int = 0
    has_pii: bool = False
    session_id: str | None = None


class LinkDocRequest(BaseModel):
    target_session_id: str


@router.get(
    "/library",
    response_model=list[LibraryDocItem],
    summary="List user library documents",
)
async def list_library_documents(
    current_user: OptionalCurrentUser = None,
    redis_client: RedisDep = None,
) -> list[LibraryDocItem]:
    """Return all library documents registered for the current user."""
    user_id = current_user.user_id if current_user else "usr_default"
    items: list[LibraryDocItem] = []
    if redis_client is not None:
        try:
            raw_docs = await redis_client.hgetall(f"saarthi:library:{user_id}:docs")
            for _, raw_val in raw_docs.items():
                try:
                    data = json.loads(raw_val)
                    items.append(LibraryDocItem.model_validate(data))
                except Exception:
                    pass
            items.sort(key=lambda x: x.uploaded_at, reverse=True)
        except Exception as e:
            logger.warning("LIBRARY_FETCH_FAILED", error=str(e))
    return items


@router.post(
    "/library/upload",
    response_model=LibraryDocItem,
    summary="Upload Document to Personal Library",
)
async def upload_library_document(
    file: UploadFile = File(...),
    current_user: OptionalCurrentUser = None,
    redis_client: RedisDep = None,
) -> LibraryDocItem:
    """Upload and index a document in the user's personal library."""
    user_id = current_user.user_id if current_user else "usr_default"
    filename = file.filename or "policy_doc.pdf"
    file_ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if file_ext not in ALLOWED_ATTACHMENT_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported attachment extension '{file_ext}'. Allowed: {list(ALLOWED_ATTACHMENT_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > MAX_ATTACHMENT_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Attachment exceeds maximum limit of {MAX_ATTACHMENT_SIZE_BYTES // (1024 * 1024)}MB",
        )

    chunks = parse_uploaded_document(file_bytes=content, filename=filename)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unable to extract text clauses from uploaded file. Please verify file content.",
        )

    doc_id = "doc_" + uuid.uuid4().hex[:8]
    session_id = f"library_{doc_id}"

    # Store in memory for instant retrieval
    SessionDocumentStore.add_document(
        session_id=session_id,
        filename=filename,
        chunks=chunks,
        user_id=user_id,
    )

    doc_item = LibraryDocItem(
        id=doc_id,
        name=filename,
        size=len(content),
        uploaded_at=datetime.now(UTC).isoformat(),
        status="ready",
        chunks=len(chunks),
        has_pii=False,
        session_id=session_id,
    )

    if redis_client is not None:
        try:
            await redis_client.hset(
                f"saarthi:library:{user_id}:docs",
                doc_id,
                json.dumps(doc_item.model_dump()),
            )
        except Exception as e:
            logger.warning("LIBRARY_SAVE_REDIS_ERROR", error=str(e))

    return doc_item


@router.post(
    "/library/{doc_id}/link",
    summary="Link library document to active chat session",
)
async def link_library_document(
    doc_id: str,
    payload: LinkDocRequest,
    current_user: OptionalCurrentUser = None,
) -> dict[str, Any]:
    """Copy attached document chunks to the active chat session ID."""
    user_id = current_user.user_id if current_user else "usr_default"
    source_session_id = f"library_{doc_id}"
    ok = SessionDocumentStore.link_session(
        from_session_id=source_session_id,
        to_session_id=payload.target_session_id,
        user_id=user_id,
    )
    return {"ok": ok, "session_id": payload.target_session_id}


@router.delete(
    "/library/{doc_id}",
    summary="Delete document from Personal Library",
)
async def delete_library_document(
    doc_id: str,
    current_user: OptionalCurrentUser = None,
    redis_client: RedisDep = None,
) -> dict[str, Any]:
    """Delete document from library and release indexed chunks."""
    user_id = current_user.user_id if current_user else "usr_default"
    if redis_client is not None:
        try:
            await redis_client.hdel(f"saarthi:library:{user_id}:docs", doc_id)
        except Exception as e:
            logger.warning("LIBRARY_DELETE_REDIS_ERROR", error=str(e))
    SessionDocumentStore.clear_session(f"library_{doc_id}")
    return {"ok": True, "deleted_id": doc_id}

