"""Vector and session retrieval node with RBAC-aware Qdrant payload filters.

Retrieval priority:
  1. User-uploaded session attachment (highest priority, user's own scope)
  2. Org knowledge base via Qdrant (access-level filter applied from user roles)
  3. Web search results are handled by the web_search subgraph node separately

Access-level mapping (spec §3):
  admin / kb_admin / compliance_reviewer → can see ``restricted`` documents
  employee with department=<X>            → sees ``all_employees`` + ``department:<X>``
  employee (no department)                → sees ``all_employees`` only
"""

from __future__ import annotations

from typing import Any

import logfire
from langchain_core.runnables import RunnableConfig

from app.core.document_processing.session_docs import SessionDocumentStore
from app.core.graphs.retrieval.state import AgenticRAGState, get_configurable
from app.core.retrieval.qdrant_service import search_policy_vectors
from app.observability.logging import get_logger

logger = get_logger("retriever_node")


def _build_access_level_filter(state: AgenticRAGState) -> list[str]:
    """Derive which Qdrant access_level values this user may see.

    Returns a list of allowed ``access_level`` payload values.  The Qdrant
    query will be filtered to only return documents whose ``access_level``
    field is in this list.
    """
    roles = set(state.user_roles)
    departments = state.user_departments

    # Admins and compliance roles see everything
    if roles & {"admin", "kb_admin", "compliance_reviewer"}:
        return ["all_employees", "restricted"] + [f"department:{d}" for d in departments]

    # Regular employees: all_employees + their specific departments
    allowed: list[str] = ["all_employees"]
    for dept in departments:
        allowed.append(f"department:{dept}")
    return allowed


def _determine_top_k(state: AgenticRAGState) -> int:
    """Determine optimal retrieval top_k based on query type and sub-intent (Roadmap §9)."""
    if state.query_type == "general_on_topic":
        return 3
    sub_intent = getattr(state, "sub_intent", None)
    if sub_intent == "product_comparison":
        return 10  # Broad parallel retrieval for balanced comparison synthesis
    if sub_intent == "claim_enquiry":
        return 6   # Prioritize policy T&C + IRDAI claim settlement norms
    if sub_intent in ("renewal_servicing", "tax_query"):
        return 5   # Focused servicing/tax context
    if sub_intent == "regulatory_lookup":
        return 6   # Statutory norms & circular clauses
    return 8


async def retrieval_node(
    state: AgenticRAGState, config: RunnableConfig
) -> dict[str, Any]:
    """Execute vector retrieval against Qdrant and session memory.

    RBAC: the ``access_level`` payload filter is derived from the user's roles
    and departments, so an employee in Operations never sees documents tagged
    ``restricted`` or ``department:Compliance``.
    """
    with logfire.span(
        "graph.retrieval_node",
        query=state.sanitized_query,
        type=state.query_type,
        sub_intent=state.sub_intent,
        user_id=state.user_id,
        user_roles=state.user_roles,
    ):
        citations: list[dict[str, Any]] = []
        retrieved_clauses: list[dict[str, Any]] = []

        # ── 1. User-Uploaded Session Attachment (Highest Priority) ─────────
        if state.session_id:
            attachment_chunks = SessionDocumentStore.search(
                session_id=state.session_id,
                query=state.sanitized_query,
                top_k=3,
                user_id=state.user_id,
            )
            for chunk in attachment_chunks:
                retrieved_clauses.append(chunk)
                citations.append(
                    {
                        "document_id": chunk["document_id"],
                        "document_title": chunk["document_title"],
                        "document_type": "user_attachment",
                        "product_code": "ATTACHMENT",
                        "source_type": "user_attachment",
                        "priority_rank": 1,
                        "uin": chunk.get("uin"),
                        "uin_warning": chunk.get("uin_warning"),
                        "section": chunk.get("section", "Uploaded Document"),
                        "clause_number": chunk.get("clause_number"),
                        "page_number": chunk.get("page_number", 1),
                        "snippet": chunk["text"],
                        "score": chunk.get("score", 0.95),
                    }
                )

        # ── 2. Org KB via Qdrant (RBAC-filtered) ──────────────────────────
        cfg = get_configurable(config)
        qdrant = cfg.get("qdrant")
        embedder = cfg.get("embedder")
        settings = cfg.get("settings")

        if qdrant and embedder and settings:
            # Build the access-level filter from the user's roles and departments
            allowed_access_levels = _build_access_level_filter(state)
            retrieval_top_k = _determine_top_k(state)
            logger.info(
                "RETRIEVAL_ACCESS_FILTER",
                user_id=state.user_id,
                roles=state.user_roles,
                allowed_levels=allowed_access_levels,
                sub_intent=state.sub_intent,
                top_k=retrieval_top_k,
            )

            matched = await search_policy_vectors(
                client=qdrant,
                collection_name=settings.QDRANT_COLLECTION_NAME,
                embedder=embedder,
                query=state.sanitized_query,
                product_code=state.product_code,
                state=state.state,
                top_k=retrieval_top_k,
                query_embedding=state.query_embedding,
                allowed_access_levels=allowed_access_levels,
            )

        else:
            matched = []
            logger.warning(
                "RETRIEVAL_MISSING_CLIENTS",
                has_qdrant=bool(qdrant),
                has_embedder=bool(embedder),
                has_settings=bool(settings),
            )

        for c in matched:
            retrieved_clauses.append(c)
            source_type = c.get("source_type", "contract")
            priority = 2 if source_type == "contract" else 3
            citations.append(
                {
                    "document_id": c["document_id"],
                    "document_title": c["document_title"],
                    "document_type": c.get("document_type", "master_policy"),
                    "product_code": c.get("product_code", state.product_code or "SARAL-TERM"),
                    "source_type": source_type,
                    "priority_rank": priority,
                    "uin": c.get("uin"),
                    "state": c.get("state", state.state or "ALL"),
                    "section": c.get("section", "General Provisions"),
                    "article": c.get("article", ""),
                    "clause_number": c.get("clause_number", ""),
                    "page_number": c.get("page_number", 1),
                    "snippet": c.get("snippet", ""),
                    "score": c.get("score", 0.0),
                    # Access metadata (for audit/tracing, never returned to client)
                    "access_level": c.get("access_level", "all_employees"),
                }
            )

        # Sort strictly by source priority: User Attachment (1) > Official Contract (2) > Regulatory (3)
        citations.sort(key=lambda x: (x.get("priority_rank", 99), -x.get("score", 0.0)))

        logger.info(
            "RETRIEVAL_COMPLETE",
            clause_count=len(retrieved_clauses),
            citation_count=len(citations),
        )

        return {
            "retrieved_clauses": retrieved_clauses,
            "citations": citations,
            "retrieval_quality": (
                "empty" if not retrieved_clauses
                else "insufficient" if len(retrieved_clauses) < 2
                else "adequate"
            ),
        }
