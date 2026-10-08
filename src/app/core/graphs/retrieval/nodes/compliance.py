"""Compliance guardrail node for statutory disclosures."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from app.core.graphs.retrieval.state import AgenticRAGState
from app.core.guardrails import guardrails_manager


async def compliance_guard_node(
    state: AgenticRAGState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Run output guardrails: PII scrubbing, ULIP risk disclosure, claim settlement notice."""
    guarded = guardrails_manager.validate_output(
        state.raw_answer,
        query=state.sanitized_query,
    )
    return {"final_answer": guarded}
