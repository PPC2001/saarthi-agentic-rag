"""Query orchestration service adhering to SOLID principles.

Encapsulates LangGraph execution, citation extraction, performance telemetry,
and statutory compliance audit logging away from HTTP controllers.
"""

import hashlib
import json
import re
import time
from typing import Any, Protocol, runtime_checkable

import logfire
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from redis.asyncio import Redis

from app.api.middleware.auth import AuthenticatedUser
from app.api.schemas.query import Citation, PolicyAnswerResponse, PolicyQueryRequest
from app.config import Settings
from app.core.graphs.retrieval.builder import build_policy_rag_graph, policy_rag_graph
from app.core.graphs.retrieval.state import AgenticRAGState
from app.core.llm_gateway.embeddings import Embedder
from app.core.retrieval.qdrant_service import AsyncQdrantClient
from app.observability.logging import get_logger, log_compliance_audit
from app.observability.metrics import record_rag_query

logger = get_logger("query_service")


@runtime_checkable
class QueryServiceProtocol(Protocol):
    """Abstract protocol for policy query execution (Interface Segregation / LSP)."""

    async def execute_query(
        self,
        request: PolicyQueryRequest,
        current_user: AuthenticatedUser,
    ) -> PolicyAnswerResponse:
        """Execute a user query and return a structured policy answer."""
        ...


_CANONICAL_QUERY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)\b(grace\s*period|fup|days\s*to\s*pay\s*premium)\b"), "canonical:grace_period"),
    (re.compile(r"(?i)(ulip[\s\w]*lock[\s-]?in|lock[\s-]?in[\s\w]*ulip|5\s*years?[\s\w]*lock[\s-]?in)"), "canonical:ulip_lock_in"),
    (re.compile(r"(?i)\b(free[\s-]?look(\s*period|\s*cancellation|\s*refund)?|cooling[\s-]?off)\b"), "canonical:free_look_period"),
    (re.compile(r"(?i)\b(section\s*45|incontestab\w+|3\s*years?\s*rule\s*(claim|repudiat))\b"), "canonical:section_45_incontestability"),
    (re.compile(r"(?i)\b(suicide\s*clause|suicide\s*exclusion)\b"), "canonical:suicide_clause"),
    (re.compile(r"(?i)\b(paid[\s-]?up\s*value|reduced\s*paid[\s-]?up)\b"), "canonical:reduced_paid_up"),
]



def normalize_canonical_query(query: str, product_code: str | None, state_jurisdiction: str | None) -> str:
    """Normalize frequent regulatory insurance queries to shared canonical cache buckets (Roadmap §15)."""
    q_lower = query.strip().lower()
    # Check if query matches a known high-frequency statutory pattern without specific product
    if not product_code or product_code.upper() in ("ALL", "GENERAL"):
        for pattern, canonical_id in _CANONICAL_QUERY_PATTERNS:
            if pattern.search(q_lower):
                return f"{canonical_id}:{state_jurisdiction or 'ALL'}"
    # Standard normalized token hash
    norm_text = re.sub(r"[^\w\s]", "", q_lower)
    tokens = [w for w in norm_text.split() if w not in ("what", "is", "the", "for", "a", "an", "in", "of", "to", "how", "much", "does")]
    canonical_tokens = " ".join(tokens) if tokens else q_lower
    return f"{product_code or 'ALL'}:{state_jurisdiction or 'ALL'}:{canonical_tokens}"


class QueryService:
    """Concrete implementation of policy querying orchestrated via LangGraph."""


    def __init__(
        self,
        llm: BaseChatModel | None,
        embedder: Embedder,
        qdrant: AsyncQdrantClient,
        redis_client: Redis | None,
        settings: Settings,
        checkpointer: BaseCheckpointSaver | None = None,
        graph: CompiledStateGraph | None = None,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.qdrant = qdrant
        self.redis_client = redis_client
        self.settings = settings
        self.checkpointer = checkpointer
        if graph is not None:
            self.graph = graph
        elif checkpointer is not None:
            self.graph = build_policy_rag_graph(checkpointer=checkpointer)
        else:
            self.graph = policy_rag_graph

    async def execute_query(
        self,
        request: PolicyQueryRequest,
        current_user: AuthenticatedUser,
    ) -> PolicyAnswerResponse:
        """Execute policy query against the LangGraph Agentic RAG pipeline."""
        start_time = time.perf_counter()

        # 0. Check Redis Cache for non-attachment queries (Roadmap §15)
        cache_key: str | None = None
        if self.redis_client is not None and not request.has_attachment:
            canonical_key = normalize_canonical_query(request.query, request.product_code, request.state)
            cache_key = f"saarthi:cache:query:{hashlib.sha256(canonical_key.encode()).hexdigest()}"

            try:
                await self.redis_client.incr("saarthi:cache:lookups")
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    cached_dict = json.loads(cached_data)
                    # Invalidate stale cache if citations were requested but cached entry is empty for a policy query
                    has_citations = bool(cached_dict.get("citations"))
                    is_conversational = cached_dict.get("query_type") in (
                        "greeting",
                        "gratitude",
                        "help",
                        "off_topic",
                        "security_blocked",
                    )
                    if request.include_citations and not has_citations and not is_conversational:
                        logger.info(
                            "SEMANTIC_CACHE_BYPASS_EMPTY_CITATIONS",
                            query_prefix=request.query[:80],
                        )
                    else:
                        await self.redis_client.incr("saarthi:cache:hits")
                        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                        cached_dict["latency_ms"] = latency_ms
                        cached_dict["session_id"] = request.session_id
                        logger.info("SEMANTIC_CACHE_HIT", query_prefix=request.query[:80])
                        record_rag_query(
                            query_type=cached_dict.get("query_type", "cached"),
                            duration_seconds=latency_ms / 1000.0,
                        )
                        return PolicyAnswerResponse.model_validate(cached_dict)
            except Exception as cache_err:
                logger.debug("CACHE_READ_ERROR", error=str(cache_err))

        with logfire.span(
            "rag.agentic_pipeline",
            user_id=current_user.user_id,
            session_id=request.session_id,
            has_attachment=request.has_attachment,
        ) as pipeline_span:
            logger.info(
                "AGENTIC_RAG_PIPELINE_INITIATED",
                user_id=current_user.user_id,
                session_id=request.session_id,
            )

            # 1. Build initial graph state
            initial_state = AgenticRAGState(
                query=request.query,
                thread_id=request.session_id,
                user_id=current_user.user_id,
                user_email=current_user.email,
                user_roles=current_user.roles,
                user_departments=current_user.departments,
                product_code=request.product_code,
                state=request.state or "ALL",
                session_id=request.session_id,
                has_attachment=request.has_attachment,
                include_citations=request.include_citations,
                is_blocked=False,
                fast_path_answer=None,
                final_answer="",
                retrieved_documents=[],
                reranked_documents=[],
                citations=[],
                calculation_result=None,
            )

            config = {
                "configurable": {
                    "thread_id": request.session_id,
                    "checkpoint_ns": "policy_rag",
                    "llm": self.llm,
                    "embedder": self.embedder,
                    "qdrant": self.qdrant,
                    "redis_client": self.redis_client,
                    "settings": self.settings,
                    "user": current_user,
                }
            }

            # 2. Execute workflow
            final_state_dict: dict[str, Any] = await self.graph.ainvoke(
                initial_state, config=config
            )

            # 3. Formulate Citations
            citations: list[Citation] = []
            if request.include_citations and not final_state_dict.get("is_blocked"):
                raw_cits = final_state_dict.get("citations", [])
                for c in raw_cits:
                    try:
                        citations.append(
                            Citation(
                                document_id=c.get("document_id", "doc-1"),
                                document_title=c.get("document_title", "Policy Document"),
                                document_type=c.get("document_type", "master_policy"),
                                product_code=c.get(
                                    "product_code", request.product_code or "SARAL-TERM"
                                ),
                                source_type=c.get("source_type", "contract"),
                                url=c.get("url"),
                                uin=c.get("uin"),
                                state=c.get("state", request.state or "ALL"),
                                section=c.get("section", "General Provisions"),
                                article=c.get("article"),
                                clause_number=c.get("clause_number"),
                                page_number=c.get("page_number", 1),
                                snippet=c.get("snippet", ""),
                                score=c.get("score", 0.0),
                            )
                        )
                    except Exception as cit_err:
                        logger.debug("CITATION_PARSE_SKIPPED", error=str(cit_err))

            # 4. Extract Answer & Metrics
            final_answer = (
                final_state_dict.get("final_answer")
                or final_state_dict.get("fast_path_answer")
                or final_state_dict.get("raw_answer")
                or "Inquiry could not be processed."
            )

            query_type = final_state_dict.get("query_type", "policy_servicing_lookup")
            model_used = final_state_dict.get("model_used")
            if (
                not model_used
                or final_state_dict.get("is_blocked")
                or query_type in ("off_topic", "security_blocked")
            ):
                if final_state_dict.get("is_blocked") or query_type in (
                    "off_topic",
                    "security_blocked",
                ):
                    model_used = "guardrail_filter"
                elif query_type in ("greeting", "help", "gratitude"):
                    model_used = "intent_router"
                else:
                    model_used = self.settings.LLM_MODEL

            latency_ms = (time.perf_counter() - start_time) * 1000

            pipeline_span.set_attribute("latency_ms", round(latency_ms, 2))
            pipeline_span.set_attribute("query_type", query_type)
            pipeline_span.set_attribute("citations_count", len(citations))

            logger.info(
                "AGENTIC_RAG_PIPELINE_COMPLETED",
                latency_ms=round(latency_ms, 2),
                query_type=query_type,
                citations_count=len(citations),
            )

            # Record Prometheus metric
            record_rag_query(
                query_type=query_type,
                duration_seconds=latency_ms / 1000.0,
            )

            # Record Redis metrics for usage & cost analytics
            if self.redis_client is not None:
                try:
                    await self.redis_client.incr("saarthi:metrics:total_queries")
                    depts = current_user.departments if current_user.departments else ["Operations & Servicing"]
                    for d in depts:
                        await self.redis_client.incr(f"saarthi:metrics:dept:{d}")
                    if model_used:
                        await self.redis_client.incr(f"saarthi:metrics:model:{model_used}")
                    est_tokens = max(50, (len(request.query) + len(final_answer)) // 4)
                    await self.redis_client.incrby("saarthi:metrics:total_tokens", est_tokens)
                except Exception as telemetry_err:
                    logger.debug("TELEMETRY_RECORDING_ERROR", error=str(telemetry_err))

            # 5. Immutable Compliance Audit Log
            log_compliance_audit(
                user_id=current_user.user_id,
                session_id=request.session_id,
                query=request.query,
                sources=[c.model_dump() for c in citations],
                answer=final_answer,
                latency_ms=latency_ms,
                model_used=model_used,
                query_type=query_type,
            )

            response = PolicyAnswerResponse(
                answer=final_answer,
                query_type=query_type,
                sub_intent=final_state_dict.get("sub_intent"),
                quality_score=final_state_dict.get("quality_score"),
                structured_response=final_state_dict.get("structured_response"),
                citations=citations,
                latency_ms=round(latency_ms, 2),
                model_used=model_used,
                session_id=request.session_id,
            )


            # 6. Store in Redis Cache for future identical queries
            # Only cache if citations were retrieved when requested, or for conversational intents
            can_cache_citations = not request.include_citations or bool(citations) or query_type in (
                "greeting",
                "gratitude",
                "help",
            )
            if (
                cache_key is not None
                and self.redis_client is not None
                and not final_state_dict.get("is_blocked")
                and query_type not in ("security_blocked", "off_topic")
                and can_cache_citations
            ):
                try:
                    await self.redis_client.set(
                        cache_key,
                        json.dumps(response.model_dump()),
                        ex=self.settings.REDIS_CACHE_TTL_SECONDS,
                    )
                except Exception as cache_set_err:
                    logger.debug("CACHE_WRITE_ERROR", error=str(cache_set_err))

            return response
