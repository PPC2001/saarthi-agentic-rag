"""Graph construction and compilation for Agentic RAG."""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.core.graphs.retrieval.nodes.actuarial import actuarial_calculator_node
from app.core.graphs.retrieval.nodes.compliance import compliance_guard_node
from app.core.graphs.retrieval.nodes.grader import document_grader_node
from app.core.graphs.retrieval.nodes.quality_scorer import quality_scorer_node
from app.core.graphs.retrieval.nodes.retriever import retrieval_node
from app.core.graphs.retrieval.nodes.router_node import router_node
from app.core.graphs.retrieval.nodes.synthesizer import synthesis_node
from app.core.graphs.retrieval.nodes.web_search import web_search_node
from app.core.graphs.retrieval.router import (
    route_after_document_grader,
    route_after_router_node,
)
from app.core.graphs.retrieval.state import AgenticRAGState


def build_policy_rag_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build and compile the LangGraph Agentic RAG workflow."""
    workflow = StateGraph(AgenticRAGState)

    # 1. Add Nodes
    workflow.add_node("router_node", router_node)
    workflow.add_node("actuarial_calculator", actuarial_calculator_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("document_grader", document_grader_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("quality_scorer", quality_scorer_node)
    workflow.add_node("compliance_guard", compliance_guard_node)

    # 2. Add Edges & Conditional Routing
    workflow.set_entry_point("router_node")

    workflow.add_conditional_edges(
        "router_node",
        route_after_router_node,
        {
            "end_fast_path": END,
            "synthesis": "synthesis",
            "actuarial_calculator": "actuarial_calculator",
            "web_search": "web_search",
            "retrieval": "retrieval",
        },
    )

    workflow.add_edge("actuarial_calculator", "retrieval")
    workflow.add_edge("retrieval", "document_grader")

    workflow.add_conditional_edges(
        "document_grader",
        route_after_document_grader,
        {
            "web_search": "web_search",
            "synthesis": "synthesis",
        },
    )

    workflow.add_edge("web_search", "synthesis")
    workflow.add_edge("synthesis", "quality_scorer")
    workflow.add_edge("quality_scorer", "compliance_guard")
    workflow.add_edge("compliance_guard", END)

    return workflow.compile(checkpointer=checkpointer)



# Singleton compiled graph
policy_rag_graph: CompiledStateGraph = build_policy_rag_graph()
