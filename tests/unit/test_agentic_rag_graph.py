"""Unit tests for LangGraph Agentic RAG workflow and Router Node."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.graphs.retrieval.graph import (
    AgenticRAGState,
    build_policy_rag_graph,
    input_guard_and_route_node,
    route_after_document_grader,
    route_after_input_guard,
    route_after_router_node,
    router_node,
)
from app.core.graphs.retrieval.nodes.router_node import classify_query_with_llm


@pytest.mark.asyncio
async def test_input_guard_routing_greetings():
    """Verify greeting query fast-path routes to end_fast_path."""
    state = AgenticRAGState(query="Namaste! How can you help me?")
    result = await input_guard_and_route_node(state)
    assert result["query_type"] == "greeting"
    assert "Saarthi AI" in result["final_answer"]
    assert result["model_used"] == "intent_router"

    updated_state = AgenticRAGState(**{**state.model_dump(), **result})
    assert route_after_input_guard(updated_state) == "end_fast_path"


@pytest.mark.asyncio
async def test_router_node_off_topic():
    """Verify off-topic query routes to end_fast_path with guardrail rejection."""
    state = AgenticRAGState(query="How do I cook butter chicken?")
    result = await router_node(state)
    assert result["query_type"] == "off_topic"
    assert result["is_blocked"] is True
    assert "life insurance" in result["final_answer"].lower()

    updated_state = AgenticRAGState(**{**state.model_dump(), **result})
    assert route_after_router_node(updated_state) == "end_fast_path"


@pytest.mark.asyncio
async def test_router_node_general_on_topic():
    """Verify general on-topic conceptual query routes to retrieval node for grounded KB context."""
    state = AgenticRAGState(
        query="What is the basic difference between a term plan and a ULIP?",
        query_type="general_on_topic",
    )
    # Testing edge routing - general queries are grounded in policy/IRDAI docs
    assert route_after_router_node(state) == "retrieval"


@pytest.mark.asyncio
async def test_input_guard_routing_actuarial_calculation():
    """Verify calculation query routes to actuarial_calculator node."""
    state = AgenticRAGState(
        query="How much policy loan can I calculate on my surrender value of 500000?"
    )
    result = await input_guard_and_route_node(state)
    assert result["query_type"] == "actuarial_calculation"

    updated_state = AgenticRAGState(**{**state.model_dump(), **result})
    assert route_after_input_guard(updated_state) == "actuarial_calculator"


@pytest.mark.asyncio
async def test_router_node_llm_classification_mock():
    """Verify LLM classifier parses 4-taxonomy JSON schema properly."""
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        return_value=MagicMock(
            content='{"intent": "general_on_topic", "reasoning": "Inquiry seeks generic conceptual distinction between ULIP and term plans."}'
        )
    )

    out = await classify_query_with_llm("what is ulip policy ?", mock_llm)
    assert out is not None
    assert out.intent == "general_on_topic"
    assert "ULIP" in out.reasoning


def test_crag_document_grader_routing():
    """Verify CRAG document grader routes to web_search on insufficient retrieval."""
    insufficient_state = AgenticRAGState(
        query="What is the latest IRDAI circular?",
        retrieval_quality="insufficient",
    )
    assert route_after_document_grader(insufficient_state) == "web_search"

    adequate_state = AgenticRAGState(
        query="What is the suicide clause?",
        retrieval_quality="adequate",
    )
    assert route_after_document_grader(adequate_state) == "synthesis"


def test_compiled_graph_structure():
    """Verify LangGraph compiles with expected nodes."""
    graph = build_policy_rag_graph()
    assert graph is not None
    # Verify graph contains core nodes
    nodes = list(graph.nodes.keys())
    assert "router_node" in nodes or "input_guard_and_route" in nodes
    assert "actuarial_calculator" in nodes
    assert "retrieval" in nodes
    assert "document_grader" in nodes
    assert "web_search" in nodes
    assert "synthesis" in nodes
    assert "quality_scorer" in nodes
    assert "compliance_guard" in nodes

