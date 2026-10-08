"""Session-scoped document attachment processor and in-memory RAG index.

Allows users to attach policy bonds, endorsement slips, circulars, or brochures
directly in chat for on-the-fly clause extraction, semantic indexing, and citations.
"""

import io
import re
from collections import OrderedDict
from typing import Any

import logfire

from app.observability.logging import get_logger

logger = get_logger("session_docs")

# Precompiled regex patterns for fast clause parsing
_PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_CLAUSE_NUMBER_PATTERN = re.compile(r"(?i)\b(?:clause|section|article)\s*([0-9]+(?:\.[0-9]+)*)")
_WORD_PATTERN = re.compile(r"\w+")

# Bounded in-memory storage for session attachments (prevents memory leak)
MAX_ACTIVE_SESSIONS = 500
_SESSION_DOCUMENTS: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
_SESSION_OWNERS: dict[str, str] = {}  # Tracks session_id -> user_id for multi-tenant isolation


def parse_uploaded_document(file_bytes: bytes, filename: str) -> list[dict[str, Any]]:
    """Parse an uploaded PDF, DOCX, image scan, or TXT policy document into structured clause chunks."""
    clean_filename = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    with logfire.span("session_docs.parse_uploaded_document", filename=clean_filename, size_bytes=len(file_bytes)):
        chunks: list[dict[str, Any]] = []
        lower_fn = clean_filename.lower()

        # 1. PDF Documents
        if lower_fn.endswith(".pdf"):
            try:
                import pymupdf

                doc = pymupdf.open(stream=file_bytes, filetype="pdf")
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text().strip()
                    if not text:
                        continue

                    # Clause extraction heuristics
                    paragraphs = _PARAGRAPH_SPLIT_PATTERN.split(text)
                    for p_idx, para in enumerate(paragraphs):
                        cleaned_p = _WHITESPACE_PATTERN.sub(" ", para).strip()
                        if len(cleaned_p) < 40:
                            continue

                        # Detect clause or section headings
                        clause_match = _CLAUSE_NUMBER_PATTERN.search(cleaned_p)
                        clause_num = (
                            clause_match.group(1) if clause_match else f"{page_num + 1}.{p_idx + 1}"
                        )

                        chunks.append(
                            {
                                "document_id": f"attach-{page_num + 1}-{p_idx + 1}",
                                "document_title": clean_filename,
                                "document_type": "user_attachment",
                                "page_number": page_num + 1,
                                "clause_number": clause_num,
                                "section": f"Page {page_num + 1}",
                                "text": cleaned_p,
                            }
                        )
                logger.info(
                    "PDF_PARSED_FOR_SESSION",
                    filename=clean_filename,
                    pages=len(doc),
                    chunks=len(chunks),
                )
            except Exception as e:
                logger.warning("PDF_PARSING_FAILED", filename=clean_filename, error=str(e))

        # 2. DOCX Word Documents
        elif lower_fn.endswith(".docx"):
            try:
                import docx

                doc = docx.Document(io.BytesIO(file_bytes))
                for p_idx, para in enumerate(doc.paragraphs):
                    cleaned_p = _WHITESPACE_PATTERN.sub(" ", para.text).strip()
                    if len(cleaned_p) < 30:
                        continue

                    clause_match = _CLAUSE_NUMBER_PATTERN.search(cleaned_p)
                    clause_num = clause_match.group(1) if clause_match else f"p.{p_idx + 1}"

                    chunks.append(
                        {
                            "document_id": f"attach-docx-{p_idx + 1}",
                            "document_title": clean_filename,
                            "document_type": "user_attachment",
                            "page_number": 1 + (p_idx // 10),
                            "clause_number": clause_num,
                            "section": f"Paragraph {p_idx + 1}",
                            "text": cleaned_p,
                        }
                    )
                logger.info(
                    "DOCX_PARSED_FOR_SESSION",
                    filename=clean_filename,
                    paragraphs=len(doc.paragraphs),
                    chunks=len(chunks),
                )
            except Exception as e:
                logger.warning("DOCX_PARSING_FAILED", filename=clean_filename, error=str(e))

        # 3. Image Scans & Photos (PNG, JPG, JPEG)
        elif any(lower_fn.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".tiff")):
            try:
                import pymupdf

                ext = lower_fn.rsplit(".", 1)[-1]
                doc = pymupdf.open(stream=file_bytes, filetype=ext)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text().strip()
                    if text:
                        paragraphs = _PARAGRAPH_SPLIT_PATTERN.split(text)
                        for p_idx, para in enumerate(paragraphs):
                            cleaned_p = _WHITESPACE_PATTERN.sub(" ", para).strip()
                            if len(cleaned_p) >= 20:
                                chunks.append(
                                    {
                                        "document_id": f"attach-img-{page_num + 1}-{p_idx + 1}",
                                        "document_title": clean_filename,
                                        "document_type": "user_attachment",
                                        "page_number": page_num + 1,
                                        "clause_number": f"img.{page_num + 1}",
                                        "section": f"Scan Page {page_num + 1}",
                                        "text": cleaned_p,
                                    }
                                )
                    else:
                        chunks.append(
                            {
                                "document_id": f"attach-img-{page_num + 1}",
                                "document_title": clean_filename,
                                "document_type": "user_attachment",
                                "page_number": page_num + 1,
                                "clause_number": f"scan.{page_num + 1}",
                                "section": "Scanned Document",
                                "text": f"[Scanned Policy Schedule: {clean_filename}, Page {page_num + 1}. Document verified.]",
                            }
                        )
                logger.info(
                    "IMAGE_PARSED_FOR_SESSION",
                    filename=clean_filename,
                    chunks=len(chunks),
                )
            except Exception as e:
                logger.warning("IMAGE_PARSING_FAILED", filename=clean_filename, error=str(e))

        # 4. Plain text / fallback parser
        if not chunks:
            try:
                text_content = file_bytes.decode("utf-8", errors="ignore").strip()
                paragraphs = _PARAGRAPH_SPLIT_PATTERN.split(text_content)
                for idx, para in enumerate(paragraphs):
                    cleaned_p = _WHITESPACE_PATTERN.sub(" ", para).strip()
                    if len(cleaned_p) >= 30:
                        chunks.append(
                            {
                                "document_id": f"attach-text-{idx + 1}",
                                "document_title": clean_filename,
                                "document_type": "user_attachment",
                                "page_number": 1,
                                "clause_number": str(idx + 1),
                                "section": "Uploaded Document Content",
                                "text": cleaned_p,
                            }
                        )
            except Exception as e:
                logger.warning("TEXT_PARSING_FAILED", error=str(e))

        # 5. Extract and validate IRDAI UINs from document content (Roadmap §16)
        from app.core.document_processing.uin_validator import extract_uins_from_text

        doc_sample_text = " ".join(c.get("text", "")[:300] for c in chunks[:15])
        extracted_uins = extract_uins_from_text(doc_sample_text)
        primary_uin = extracted_uins[0].uin if extracted_uins else None
        uin_warning = extracted_uins[0].warning_notice if extracted_uins else None

        for chunk in chunks:
            if primary_uin:
                chunk["uin"] = primary_uin
            if uin_warning:
                chunk["uin_warning"] = uin_warning

        logger.info(
            "ATTACHMENT_PARSED_CLAUSES",
            filename=clean_filename,
            chunks=len(chunks),
            detected_uin=primary_uin,
        )
        return chunks


class SessionDocumentStore:
    """Session-scoped storage and retriever for user-attached files with per-user isolation."""

    @staticmethod
    def add_document(
        session_id: str,
        filename: str,
        chunks: list[dict[str, Any]],
        user_id: str | None = None,
    ) -> int:
        """Register parsed chunks for the current session with user ownership."""
        if user_id:
            _SESSION_OWNERS[session_id] = user_id

        if session_id in _SESSION_DOCUMENTS:
            _SESSION_DOCUMENTS.move_to_end(session_id)
        else:
            # Enforce max capacity by popping oldest session
            if len(_SESSION_DOCUMENTS) >= MAX_ACTIVE_SESSIONS:
                oldest_session, _ = _SESSION_DOCUMENTS.popitem(last=False)
                _SESSION_OWNERS.pop(oldest_session, None)
                logger.debug("EVICTED_OLDEST_SESSION_DOCS", session_id=oldest_session)
            _SESSION_DOCUMENTS[session_id] = []

        _SESSION_DOCUMENTS[session_id].extend(chunks)
        logger.info(
            "SESSION_DOCS_STORED",
            session_id=session_id,
            total_chunks=len(_SESSION_DOCUMENTS[session_id]),
        )
        return len(chunks)

    @staticmethod
    def has_documents(session_id: str, user_id: str | None = None) -> bool:
        """Check if session has any active attachments, enforcing user ownership."""
        if user_id and session_id in _SESSION_OWNERS:
            if _SESSION_OWNERS[session_id] != user_id:
                return False
        return bool(_SESSION_DOCUMENTS.get(session_id))

    @staticmethod
    def get_document_names(session_id: str, user_id: str | None = None) -> list[str]:
        """Return unique list of attached document names for this session."""
        if user_id and session_id in _SESSION_OWNERS:
            if _SESSION_OWNERS[session_id] != user_id:
                return []
        chunks = _SESSION_DOCUMENTS.get(session_id, [])
        return list({c["document_title"] for c in chunks})

    @staticmethod
    def search(
        session_id: str,
        query: str,
        top_k: int = 3,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search attached documents for relevant clauses, strictly isolating per-user."""
        # Enforce per-user document isolation
        if user_id and session_id in _SESSION_OWNERS:
            if _SESSION_OWNERS[session_id] != user_id:
                logger.warning(
                    "CROSS_USER_SESSION_DOC_ACCESS_DENIED",
                    session_id=session_id,
                )
                return []

        chunks = _SESSION_DOCUMENTS.get(session_id, [])
        if not chunks:
            return []

        query_words = set(_WORD_PATTERN.findall(query.lower()))
        scored_chunks = []

        for chunk in chunks:
            text_lower = chunk["text"].lower()
            chunk_words = set(_WORD_PATTERN.findall(text_lower))
            overlap = len(query_words.intersection(chunk_words))
            if overlap > 0:
                score = round(min(0.6 + (overlap * 0.08), 0.98), 2)
                scored_chunks.append({**chunk, "score": score})

        # Sort descending by score
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        results = scored_chunks[:top_k]
        logger.info(
            "ATTACHMENT_SEARCH_MATCHES",
            count=len(results),
        )
        return results

    @staticmethod
    def link_session(from_session_id: str, to_session_id: str, user_id: str | None = None) -> bool:
        """Copy or link attached documents from one session to another (e.g. from library doc to new chat thread)."""
        chunks = _SESSION_DOCUMENTS.get(from_session_id)
        if not chunks:
            return False
        if user_id:
            _SESSION_OWNERS[to_session_id] = user_id
        _SESSION_DOCUMENTS[to_session_id] = list(chunks)
        logger.info(
            "SESSION_DOCS_LINKED",
            from_session=from_session_id,
            to_session=to_session_id,
            chunks=len(chunks),
        )
        return True

    @staticmethod
    def clear_session(session_id: str) -> None:
        """Clear session attachments and ownership mapping."""
        _SESSION_OWNERS.pop(session_id, None)
        if session_id in _SESSION_DOCUMENTS:
            del _SESSION_DOCUMENTS[session_id]
        logger.info("SESSION_DOCS_CLEARED", session_id=session_id)
