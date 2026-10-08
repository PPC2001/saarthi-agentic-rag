"""Pydantic schemas for Policy Query operations."""

from typing import Any

from pydantic import BaseModel, Field

from app.api.schemas.common import DocumentType, ProductLine


class PolicyQueryRequest(BaseModel):
    """Input request for natural language policy inquiries."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=1500,
        description="Natural language question regarding policy terms, riders, exclusions, or IRDAI regulations.",
        examples=["What are the suicide exclusion terms for Saral Term under IRDAI guidelines?"],
    )
    product_code: str | None = Field(
        default=None,
        description="Filter by specific product code, e.g. 'SARAL-TERM' or 'ULIP-WEALTH'.",
    )
    product_line: ProductLine | None = Field(
        default=None,
        description="Filter by product category (term, ulip, savings_endowment, annuity_pension, etc.).",
    )
    state: str | None = Field(
        default=None,
        max_length=5,
        description="Indian state/jurisdiction code (e.g. 'MH', 'DL', 'KA', 'TN') or 'ALL' for Pan-India.",
    )
    document_types: list[DocumentType] | None = Field(
        default=None,
        description="Optional list of document types to search within.",
    )
    include_citations: bool = Field(
        default=True,
        description="Whether to include full clause citations and snippets in the response.",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for multi-turn conversational context.",
    )
    has_attachment: bool = Field(
        default=False,
        description="Flag indicating if a policy document or schedule is attached to this session.",
    )
    attachment_name: str | None = Field(
        default=None,
        description="Filename of the user's uploaded attachment.",
    )


class Citation(BaseModel):
    """Specific citation proving the source of a policy answer."""

    document_id: str
    document_title: str
    document_type: str
    product_code: str
    source_type: str = "contract"  # "contract" | "user_attachment" | "web_search"
    url: str | None = None  # Link to IRDAI or Insurer portal source if web-retrieved
    uin: str | None = None  # IRDAI Unique Identification Number
    state: str = "ALL"
    section: str | None = None
    article: str | None = None
    clause_number: str | None = None
    page_number: int | None = None
    snippet: str
    score: float = 0.0


class PolicyAnswerResponse(BaseModel):
    """Synthesized policy answer with verified legal citations."""

    answer: str
    query_type: str = "general"
    citations: list[Citation] = Field(default_factory=list)
    disclaimer: str = (
        "Statutory Notice: This policy information is provided for advisor, bancassurance, and customer service "
        "reference only in accordance with IRDAI guidelines and Section 45 of the Insurance Act, 1938. "
        "It does not constitute an official claim admission, repudiation, or binding financial advice. "
        "Please refer to the executed policy bond and official product literature."
    )
    latency_ms: float = 0.0
    model_used: str = ""
    session_id: str | None = None
    sub_intent: str | None = None
    quality_score: dict[str, Any] | None = None
    structured_response: dict[str, Any] | None = None

