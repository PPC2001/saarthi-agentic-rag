"""Corrective RAG (CRAG) document grading node with FlashRank cross-encoder.

Replaces the original naive score-threshold approach with the FlashRank
cross-encoder reranker from enterprise-rag's ranking_service.py pattern.

Flow:
  1. Retrieved clauses arrive from retrieval_node
  2. FlashRank cross-encoder re-scores each (query, clause) pair
  3. Chunks below MIN_RERANK_SCORE are dropped (relevance gating)
  4. If no chunks survive → "insufficient" → routes to web_search
  5. If chunks survive → "adequate" → routes to synthesis_node

Why this is better than the old score-threshold approach:
  - The old grader checked Qdrant's cosine similarity score (≥0.35)
    but cosine similarity is "fuzzy" — a clause about ULIP structure
    might score 0.40 for a surrender value question (wrong context)
  - FlashRank cross-encodes (query, chunk) pairs jointly → far more
    precise at understanding "this chunk actually answers this query"
  - Same chunk pool is now also sorted by cross-encoder relevance,
    improving synthesis quality (best chunk first)
"""

from typing import Any

import logfire
from langchain_core.runnables import RunnableConfig

from app.core.graphs.retrieval.state import AgenticRAGState
from app.core.retrieval.reranker import rerank_clauses
from app.observability.logging import get_logger
from app.observability.metrics import record_reranker_result

logger = get_logger("grader_node")


async def document_grader_node(
    state: AgenticRAGState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """CRAG Grader: rerank retrieved clauses with FlashRank and gate on relevance.

    Returns updated state with:
      - retrieved_clauses: reranked, filtered clause list
      - citations: reranked citation list (aligned order)
      - retrieval_quality: "adequate" | "insufficient"
    """
    with logfire.span(
        "graph.document_grader",
        clauses_count=len(state.retrieved_clauses),
        query=state.sanitized_query,
    ):
        if not state.retrieved_clauses:
            logger.warning("CRAG_NO_CLAUSES_RETRIEVED_WEB_FALLBACK")
            return {"retrieval_quality": "insufficient"}

        # Run FlashRank cross-encoder reranking
        reranked = await rerank_clauses(
            query=state.sanitized_query,
            clauses=state.retrieved_clauses,
            top_n=5,
        )

        if not reranked:
            record_reranker_result(0, len(state.retrieved_clauses))
            logger.warning(
                "CRAG_ALL_CLAUSES_BELOW_THRESHOLD_WEB_FALLBACK",
                total=len(state.retrieved_clauses),
            )
            return {"retrieval_quality": "insufficient"}

        # Rebuild aligned citations in reranked order
        # Map document_id → citation for fast lookup
        citation_map: dict[str, dict] = {c.get("document_id", ""): c for c in state.citations}

        reranked_citations: list[dict[str, Any]] = []
        for clause in reranked:
            doc_id = clause.get("document_id", "")
            if doc_id in citation_map:
                citation = dict(citation_map[doc_id])
                # Surface the reranker score in the citation for UI display
                if "rerank_score" in clause:
                    citation["rerank_score"] = clause["rerank_score"]
                reranked_citations.append(citation)

        # Include any citations not in the reranked list (e.g. user attachments, actuarial)
        reranked_ids = {c.get("document_id", "") for c in reranked}
        for c in state.citations:
            if c.get("document_id", "") not in reranked_ids:
                reranked_citations.append(c)

        record_reranker_result(len(reranked), len(state.retrieved_clauses))
        logger.info(
            "CRAG_RERANK_SUCCEEDED",
            kept=len(reranked),
            total=len(state.retrieved_clauses),
        )

        quality = "adequate" if len(reranked) >= 2 else "thin"
        return {
            "retrieved_clauses": reranked,
            "citations": reranked_citations,
            "retrieval_quality": quality,
        }
