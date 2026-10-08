"""Pydantic schemas for Chat (SSE streaming) operations.

Defines:
- ChatStreamRequest  — POST body for /v1/chat/stream
- SSE event helpers  — make_sse_event() formats any dict as a valid SSE data line

SSE wire format (spec §17.2):
  data: <json>\\n\\n

Event types:
  meta      → {requestId, threadId, authMode, model}
  status    → {node, message}
  token     → {text}
  citations → [{document_id, snippet, score, ...}]
  final     → {answer, queryType, model, latencyMs, threadId, requestId}
  error     → {code, message}
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field


class ChatStreamRequest(BaseModel):
    """Request body for the SSE chat stream endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's chat message.",
        examples=["What are the suicide exclusion terms for Saral Term?"],
    )
    thread_id: str | None = Field(
        default=None,
        description=(
            "Conversation thread ID for multi-turn context. "
            "If omitted, the server creates a new UUID thread."
        ),
    )
    product_code: str | None = Field(
        default=None,
        description="Optional product filter (e.g. 'SARAL-TERM').",
    )
    # Future: attachment_ids, preferred_sources, etc.


def make_sse_event(event_type: str, data: dict | list) -> str:
    """Format a Server-Sent Events message.

    Produces:
      event: <type>\\n
      data: <json>\\n
      \\n

    The double newline terminates the event so the browser's EventSource
    parser emits it immediately.
    """
    json_data = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_type}\ndata: {json_data}\n\n"
