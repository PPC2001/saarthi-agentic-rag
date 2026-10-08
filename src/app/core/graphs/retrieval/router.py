"""Routing logic and conditional edge definitions for LangGraph."""

from app.core.graphs.retrieval.state import AgenticRAGState


def route_after_router_node(state: AgenticRAGState) -> str:
    """Route after intent classification router node.

    Taxonomy paths:
    - greeting / off_topic / blocked -> end_fast_path
    - general_on_topic -> retrieval (grounded lightweight pass, top_k=3)
    - actuarial_calculation -> actuarial_calculator
    - web_search -> web_search
    - structured_rag / attachment_rag -> retrieval
    """
    if state.is_blocked or state.fast_path_answer is not None:
        return "end_fast_path"
    if state.query_type == "actuarial_calculation":
        return "actuarial_calculator"
    if state.query_type == "web_search":
        return "web_search"
    # Both "general_on_topic" (grounded conceptual lookup) and "structured_rag" route to retrieval
    return "retrieval"



# Backward compatibility alias
route_after_input_guard = route_after_router_node


def route_after_document_grader(state: AgenticRAGState) -> str:
    """Route based on Corrective RAG document relevance grading."""
    if state.retrieval_quality == "insufficient":
        return "web_search"
    return "synthesis"
