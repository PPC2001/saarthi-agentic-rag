"""State definition and regex heuristics for LangGraph Agentic RAG."""

import re
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

# Regex heuristics for routing to deterministic actuarial calculator tools
LOAN_CALC_REGEX = re.compile(
    r"(?i)\b(calculate|how\s*much|what\s*is\s*the\s*max(imum)?)\s+(policy\s+)?loan\b"
)
SURRENDER_CALC_REGEX = re.compile(
    r"(?i)\b(calculate|surrender\s*value\s*after|how\s*much\s*surrender)\b"
)
SECTION_45_CALC_REGEX = re.compile(
    r"(?i)\b(incontestable|section\s*45|repudiat|call\s+in\s+question)\b.*\b(years?|since|issued|revived)\b"
)
GRACE_PERIOD_CALC_REGEX = re.compile(
    r"(?i)\b(when\s+does\s+grace\s+period\s+expire|last\s+date\s+to\s+pay|revival\s+deadline|fup)\b"
)


class AgenticRAGState(BaseModel):
    """Execution state passed through the LangGraph agentic workflow.

    Identity fields (user_id, user_email, user_roles, user_departments) are
    propagated from the authenticated HTTP session so every node can apply
    RBAC-aware Qdrant payload filters without reading from the config again.
    """

    # ── Inputs ────────────────────────────────────────────────────────────
    query: str
    thread_id: str | None = None       # persistent conversation thread (LangGraph DynamoDB/Redis checkpointer)

    # ── User identity (populated by QueryService from AuthenticatedUser) ──
    user_id: str = "anonymous"
    user_email: str = ""
    user_roles: list[str] = Field(default_factory=list)
    user_departments: list[str] = Field(default_factory=list)

    # ── Insurance domain filters ──────────────────────────────────────────
    product_code: str | None = None
    state: str | None = "ALL"
    session_id: str | None = None
    has_attachment: bool = False
    include_citations: bool = True

    # ── Processing & Routing ──────────────────────────────────────────────
    sanitized_query: str = ""
    query_type: str = "policy_servicing_lookup"
    sub_intent: str | None = None       # "claim_enquiry", "product_comparison", "renewal_servicing", "regulatory_lookup", "underwriting_query", "tax_query"
    is_blocked: bool = False
    fast_path_answer: str | None = None
    query_embedding: list[float] | None = None  # Pre-computed query embedding to eliminate duplicate vectorization

    # ── Actuarial Tool Outputs ────────────────────────────────────────────
    calculation_result: dict[str, Any] | None = None

    # ── Retrieved Context ─────────────────────────────────────────────────
    retrieved_clauses: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_quality: str = "adequate"  # "adequate", "thin", "insufficient", "empty"

    # ── Synthesis & Output ────────────────────────────────────────────────
    raw_answer: str = ""
    final_answer: str = ""
    model_used: str = ""
    quality_score: dict[str, Any] | None = None            # Post-synthesis quality judge (grounding, completeness, IRDAI compliance)
    structured_response: dict[str, Any] | None = None      # Pydantic structured output for frontend/API consumption

    # ── SSE streaming token buffer (populated by synthesizer for live mode) ──
    stream_tokens: list[str] = Field(default_factory=list)



def get_configurable(config: RunnableConfig | None) -> dict[str, Any]:
    """Helper to safely extract configurable dict from RunnableConfig."""
    if config and isinstance(config, dict):
        return config.get("configurable", {})
    return {}
