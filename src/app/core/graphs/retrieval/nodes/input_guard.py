"""Input guardrail and intent routing node alias (canonical router_node)."""

from app.core.graphs.retrieval.nodes.router_node import router_node

# Direct canonical alias for router_node to eliminate function-call indirection
input_guard_and_route_node = router_node

__all__ = ["input_guard_and_route_node"]
