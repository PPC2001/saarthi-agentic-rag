"""Web search fallback node using official allowlisted domains."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from app.core.graphs.retrieval.state import AgenticRAGState, get_configurable
from app.core.retrieval.web_search import WebRetrievalClient


async def web_search_node(
    state: AgenticRAGState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Search authoritative IRDAI portals and insurer disclosures strictly allowlisted to irdai.gov.in and sbilife.co.in."""
    cfg = get_configurable(config)
    redis_client = cfg.get("redis_client")
    web_client = WebRetrievalClient(redis_client=redis_client)
    sources = await web_client.search_insurance_web(state.sanitized_query, top_k=2)

    web_citations = list(state.citations)
    for ws in sources:
        web_citations.append(
            {
                "document_id": f"web-{abs(hash(ws.url)) % 100000}",
                "document_title": ws.title,
                "document_type": ws.source_type,
                "product_code": "REGULATORY-WEB",
                "source_type": "web_search",
                "priority_rank": 4,
                "url": ws.url,
                "publication_date": ws.publication_date,
                "effective_date": ws.effective_date,
                "snippet": ws.snippet,
                "score": ws.score,
            }
        )

    # Ensure priority ordering is preserved
    web_citations.sort(key=lambda x: (x.get("priority_rank", 99), -x.get("score", 0.0)))

    return {
        "citations": web_citations,
        "query_type": "web_search",
    }
