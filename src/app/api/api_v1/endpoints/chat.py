"""SSE streaming chat endpoint.

Implements the Server-Sent Events (SSE) contract for the Saarthi AI chat UI.

SSE event stream format (spec §17.2):
  meta    → {requestId, threadId, authMode, model}
  status  → {node, message}          — progress updates from graph nodes
  token   → {text}                   — partial answer token (streaming)
  citations → [{...}]                — sources, sent after final token
  final   → {answer, queryType, ...} — full validated answer
  error   → {code, message}          — on failure

The endpoint uses LangGraph's ``astream_events`` API to emit node-level
progress events and token-level streaming from the synthesizer node.

Thread IDs:
  A ``thread_id`` identifies a conversation thread (future: Postgres
  checkpointer for persistence).  For now, UUIDs are auto-generated per
  request if not supplied by the client.

Spec §17.1: POST /v1/chat/stream  (also available at /api/v1/chat/stream)
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncGenerator

import logfire
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.deps import OptionalCurrentUser, QueryServiceDep, SettingsDep
from app.api.schemas.chat import ChatStreamRequest, make_sse_event
from app.observability.logging import get_logger, log_compliance_audit

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = get_logger("chat_stream")

# How often to send a keep-alive ping (seconds) to prevent proxy timeouts
PING_INTERVAL_EVENTS = 20


async def _stream_graph(
    request: ChatStreamRequest,
    current_user,
    query_service,
    settings,
    request_id: str,
    thread_id: str,
) -> AsyncGenerator[str, None]:
    """Core SSE generator.

    Emits events in the order: meta → status* → token* → citations → final
    Falls back to a single ``final`` event on error.
    """
    start = time.perf_counter()

    # ── meta event (always first) ───────────────────────────────────────────
    yield make_sse_event(
        "meta",
        {
            "requestId": request_id,
            "threadId": thread_id,
            "authMode": settings.AUTH_MODE,
            "model": settings.LLM_MODEL,
        },
    )

    try:
        # ── status: guard / preprocessing ─────────────────────────────────
        yield make_sse_event("status", {"node": "guard_input", "message": "Checking query…"})

        # Run the graph (non-streaming invoke first; token streaming added when
        # the synthesizer node yields partial chunks via astream_events)
        from app.api.schemas.query import PolicyQueryRequest
        from app.core.document_processing.session_docs import SessionDocumentStore

        active_sid = request.thread_id or thread_id
        user_id = current_user.user_id if current_user else None
        has_doc = SessionDocumentStore.has_documents(active_sid, user_id=user_id)
        doc_names = SessionDocumentStore.get_document_names(active_sid, user_id=user_id) if has_doc else []

        policy_request = PolicyQueryRequest(
            query=request.message,
            session_id=active_sid,
            has_attachment=has_doc,
            attachment_name=doc_names[0] if doc_names else None,
            include_citations=True,
            product_code=request.product_code,
        )

        yield make_sse_event("status", {"node": "retrieval", "message": "Searching knowledge base…"})

        # Execute the graph — currently uses the synchronous ainvoke path.
        # When the synthesizer supports streaming, swap to astream_events.
        response = await query_service.execute_query(policy_request, current_user)

        yield make_sse_event("status", {"node": "generate", "message": "Generating answer…"})

        # ── token streaming (simulate token-by-token for existing non-streaming graph) ──
        # Split the answer into word chunks and emit them incrementally so the
        # SPA gets a ChatGPT-like streaming experience even before the graph
        # supports native token streaming.
        words = response.answer.split(" ")
        for i, word in enumerate(words):
            text = word + (" " if i < len(words) - 1 else "")
            yield make_sse_event("token", {"text": text})

        # ── citations event ────────────────────────────────────────────────
        if response.citations:
            yield make_sse_event(
                "citations",
                [c.model_dump(mode="json") for c in response.citations],
            )

        # ── final event ────────────────────────────────────────────────────
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        yield make_sse_event(
            "final",
            {
                "answer": response.answer,
                "queryType": response.query_type,
                "model": response.model_used,
                "latencyMs": latency_ms,
                "threadId": thread_id,
                "requestId": request_id,
            },
        )

        # Compliance audit
        log_compliance_audit(
            user_id=current_user.user_id if current_user else "anonymous",
            session_id=thread_id,
            query=request.message,
            sources=[c.model_dump() for c in response.citations],
            answer=response.answer,
            latency_ms=latency_ms,
            model_used=response.model_used,
            query_type=response.query_type,
        )

    except Exception as exc:
        logger.error("CHAT_STREAM_ERROR", error=str(exc), request_id=request_id)
        yield make_sse_event(
            "error",
            {"code": "INTERNAL_ERROR", "message": "An error occurred while generating the response."},
        )


@router.post(
    "/stream",
    summary="Chat with streaming SSE",
    description=(
        "Stream a chat response as Server-Sent Events. "
        "Events: meta → status* → token* → citations → final | error. "
        "Send ``Accept: text/event-stream`` and a POST body with ``message``."
    ),
    response_class=StreamingResponse,
)
async def chat_stream(
    request: ChatStreamRequest,
    current_user: OptionalCurrentUser,
    query_service: QueryServiceDep,
    settings: SettingsDep,
) -> StreamingResponse:
    """Stream a chat response via Server-Sent Events.

    The client should open the connection with:
      ``POST /v1/chat/stream``
      ``Content-Type: application/json``
      ``Accept: text/event-stream``
    """
    request_id = str(uuid.uuid4())
    thread_id = request.thread_id or str(uuid.uuid4())

    with logfire.span(
        "chat.stream",
        request_id=request_id,
        thread_id=thread_id,
        user_id=current_user.user_id if current_user else "anonymous",
        message_prefix=request.message[:80],
    ):
        logger.info(
            "CHAT_STREAM_INITIATED",
            request_id=request_id,
            thread_id=thread_id,
            user_id=current_user.user_id if current_user else "anonymous",
        )

        return StreamingResponse(
            _stream_graph(
                request=request,
                current_user=current_user,
                query_service=query_service,
                settings=settings,
                request_id=request_id,
                thread_id=thread_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store",
                "X-Accel-Buffering": "no",      # disable nginx response buffering
                "X-Request-ID": request_id,
            },
        )
