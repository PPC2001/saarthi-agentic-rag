"""Unit tests for enterprise standards: Connection-pooled HTTP client and RFC 7807 Domain Exceptions."""

import pytest

from app.api.schemas.exceptions import (
    ActuarialCalculationException,
    DisallowedDomainException,
    OffTopicQueryException,
    PolicyDocumentParseException,
    PolicyNotFoundException,
    ProblemDetails,
    PromptInjectionException,
    RateLimitExceededException,
)
from app.core.http_clients import HTTPRequests, close_http_client, get_http_client
from app.observability.logging import correlation_id_ctx


def test_domain_exception_problem_details():
    """Verify domain exception translates into RFC 7807 problem details with trace instance link."""
    token = correlation_id_ctx.set("test-trace-id-12345")
    try:
        exc = PolicyNotFoundException(
            "Policy 'TERM-001' not found in registry", metadata={"product": "TERM-001"}
        )
        problem = exc.to_problem_details()

        assert isinstance(problem, ProblemDetails)
        assert problem.status == 404
        assert problem.code == "ERR_POLICY_NOT_FOUND"
        assert problem.title == "PolicyNotFound"
        assert problem.detail == "Policy 'TERM-001' not found in registry"
        assert problem.instance == "/traces/test-trace-id-12345"
        assert problem.metadata.get("product") == "TERM-001"
    finally:
        correlation_id_ctx.reset(token)


def test_domain_exception_hierarchy_statuses():
    """Verify specific status codes across all domain exception types."""
    assert PolicyDocumentParseException("parse error").status_code == 422
    assert ActuarialCalculationException("calc error").status_code == 400
    assert DisallowedDomainException("domain error").status_code == 403
    assert PromptInjectionException("injection detected").status_code == 400
    assert OffTopicQueryException("off topic").status_code == 400
    assert RateLimitExceededException("too many requests").status_code == 429


@pytest.mark.asyncio
async def test_http_client_singleton_and_pooling():
    """Verify HTTPRequests connection pool configuration and lifecycle."""
    client = get_http_client()
    assert isinstance(client, HTTPRequests)
    assert not client.is_closed

    # Verify connection limits and default headers
    assert "User-Agent" in client.headers
    assert "Accept" in client.headers
    assert client.timeout.read == 15.0

    # Ensure get_http_client returns the same singleton
    client2 = get_http_client()
    assert client is client2

    # Clean up
    await close_http_client()
    assert client.is_closed


def test_api_v1_versioning_and_router_structure():
    """Verify API v1 router registers endpoints under both /v1 and /api/v1 prefixes."""
    from fastapi.testclient import TestClient

    from app.api.api_v1.router import api_router
    from app.main import app

    client = TestClient(app)

    # 1. Root /health probe for ALB and container runtime
    resp_root = client.get("/health")
    assert resp_root.status_code == 200

    # 2. Versioned /v1/health probe
    resp_v1 = client.get("/v1/health")
    assert resp_v1.status_code == 200

    # 3. Microservice standard /api/v1/health probe
    resp_api_v1 = client.get("/api/v1/health")
    assert resp_api_v1.status_code == 200

    # 4. Root / and /info metadata probe
    resp_info = client.get("/")
    assert resp_info.status_code == 200
    assert "service" in resp_info.json()
    assert "version" in resp_info.json()

    # 5. Verify central router has registered routes
    assert len(api_router.routes) > 0


def test_api_utils_helpers():
    """Verify helper utilities in app.api.utils."""
    from app.api.utils import format_error_detail, sanitize_filename

    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("policy contract [2026]!#.pdf") == "policy_contract__2026___.pdf"
    assert sanitize_filename("") == "document"

    err = format_error_detail("ERR_TEST", "Test error", "field_a")
    assert err == {"code": "ERR_TEST", "message": "Test error", "field": "field_a"}


def test_api_deps_exports():
    """Verify common dependency exports in app.api.deps."""
    from app.api.deps import (
        CurrentUser,
        EmbedderDep,
        QdrantDep,
        QueryServiceDep,
        RedisDep,
        SettingsDep,
        SmartLLMDep,
        get_current_user,
        get_query_service,
        get_settings,
    )

    assert CurrentUser is not None
    assert EmbedderDep is not None
    assert QdrantDep is not None
    assert RedisDep is not None
    assert SettingsDep is not None
    assert SmartLLMDep is not None
    assert QueryServiceDep is not None
    assert callable(get_current_user)
    assert callable(get_query_service)
    assert callable(get_settings)


@pytest.mark.asyncio
async def test_query_service_execution_and_solid_design():
    """Verify QueryService adheres to SRP and DIP by orchestrating query and audit."""
    from unittest.mock import AsyncMock, MagicMock

    from app.api.middleware.auth import AuthenticatedUser
    from app.api.schemas.query import PolicyQueryRequest
    from app.config import get_settings
    from app.services.query_service import QueryService, QueryServiceProtocol

    # Verify Protocol compliance
    assert issubclass(QueryService, QueryServiceProtocol)

    # Mock components for isolated unit testing
    mock_llm = MagicMock()
    mock_embedder = MagicMock()
    mock_qdrant = MagicMock()
    mock_redis = MagicMock()
    settings = get_settings()

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "final_answer": "Policy loan is capped at 90% of surrender value.",
            "query_type": "actuarial_calculation",
            "model_used": "llama-3.3-70b-versatile",
            "citations": [
                {
                    "document_id": "test-doc-1",
                    "document_title": "Saral Jeevan Bima",
                    "section": "Loan Provisions",
                    "snippet": "Maximum loan available is 90% of surrender value.",
                    "score": 0.92,
                }
            ],
            "is_blocked": False,
        }
    )

    service = QueryService(
        llm=mock_llm,
        embedder=mock_embedder,
        qdrant=mock_qdrant,
        redis_client=mock_redis,
        settings=settings,
        graph=mock_graph,
    )

    request = PolicyQueryRequest(query="What is the maximum loan limit?")
    user = AuthenticatedUser(
        user_id="usr-test-1",
        api_key_id="key-test-01",
        email="agent@sbi.co.in",
        role="customer",
    )

    resp = await service.execute_query(request, user)
    assert resp.answer == "Policy loan is capped at 90% of surrender value."
    assert resp.query_type == "actuarial_calculation"
    assert len(resp.citations) == 1
    assert resp.citations[0].document_id == "test-doc-1"
    assert resp.model_used == "llama-3.3-70b-versatile"
    mock_graph.ainvoke.assert_awaited_once()


def test_modular_retrieval_graph_components():
    """Verify modular SRP components in core.graphs.retrieval."""
    from app.core.graphs.retrieval.builder import build_policy_rag_graph, policy_rag_graph
    from app.core.graphs.retrieval.nodes.actuarial import actuarial_calculator_node
    from app.core.graphs.retrieval.nodes.compliance import compliance_guard_node
    from app.core.graphs.retrieval.nodes.grader import document_grader_node
    from app.core.graphs.retrieval.nodes.input_guard import input_guard_and_route_node
    from app.core.graphs.retrieval.nodes.retriever import retrieval_node
    from app.core.graphs.retrieval.nodes.router_node import router_node
    from app.core.graphs.retrieval.nodes.synthesizer import synthesis_node
    from app.core.graphs.retrieval.nodes.web_search import web_search_node
    from app.core.graphs.retrieval.router import (
        route_after_document_grader,
        route_after_input_guard,
        route_after_router_node,
    )
    from app.core.graphs.retrieval.state import AgenticRAGState

    assert callable(actuarial_calculator_node)
    assert callable(compliance_guard_node)
    assert callable(document_grader_node)
    assert callable(input_guard_and_route_node)
    assert callable(router_node)
    assert callable(retrieval_node)
    assert callable(synthesis_node)
    assert callable(web_search_node)
    assert callable(route_after_document_grader)
    assert callable(route_after_input_guard)
    assert callable(route_after_router_node)
    assert callable(build_policy_rag_graph)
    assert policy_rag_graph is not None

    test_state = AgenticRAGState(query="Test query")
    assert test_state.query == "Test query"
    assert test_state.retrieval_quality == "adequate"
