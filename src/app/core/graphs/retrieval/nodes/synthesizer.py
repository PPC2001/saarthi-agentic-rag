"""LLM synthesis node for legally grounded response generation.

Token-Budget Context Assembly:
  Uses tiktoken to count tokens per chunk and never truncates mid-chunk.
  Always cuts at chunk boundaries to preserve legal clause citation integrity.
  Inspired by enterprise-rag's responder.py _build_context() pattern.
"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.config import Settings
from app.core.graphs.retrieval.state import AgenticRAGState, get_configurable
from app.core.llm_gateway.prompts import build_query_prompt, build_system_prompt
from app.observability.logging import get_logger

logger = get_logger("synthesis_node")

# Maximum tokens for injected context (leaves headroom for system prompt + output)
_MAX_CONTEXT_TOKENS: int = 6_000

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")  # GPT-4 / gpt-oss-120b compatible

    def _count_tokens(text: str) -> int:
        return len(_ENCODING.encode(text))

except ImportError:
    # tiktoken not available — fall back to character-based approximation
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return len(text) // 4  # ~4 chars per token


def _build_context_with_budget(
    context_chunks: list[dict],
    max_tokens: int = _MAX_CONTEXT_TOKENS,
) -> tuple[list[dict], int, int]:
    """Select context chunks within a token budget, always cutting at chunk boundaries.

    NEVER truncates mid-chunk — legal clause citations must not be split mid-sentence.
    This prevents compliance risk from malformed regulatory text in insurance responses.

    Returns:
        (selected_chunks, tokens_used, total_chunks)
    """
    selected: list[dict] = []
    used_tokens: int = 0

    for chunk in context_chunks:
        chunk_text = chunk.get("text") or chunk.get("snippet", "")
        chunk_tokens = _count_tokens(chunk_text)
        if used_tokens + chunk_tokens > max_tokens:
            # Stop at chunk boundary — never truncate mid-sentence
            break
        selected.append(chunk)
        used_tokens += chunk_tokens

    return selected, used_tokens, len(context_chunks)


async def synthesis_node(
    state: AgenticRAGState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Synthesize legally grounded response using smart LLM and context."""
    cfg = get_configurable(config)
    llm = cfg.get("llm")
    settings: Settings | None = cfg.get("settings")
    llm_model = settings.LLM_MODEL if settings else "openai/gpt-oss-120b"

    system_text = build_system_prompt()

    # Context chunks formatting with explicit priority tiers and dates
    context_chunks = []
    has_attachment_citation = False
    has_official_citation = False

    for c in state.citations:
        source_type = c.get("source_type", "contract")
        if source_type == "user_attachment":
            has_attachment_citation = True
        elif source_type in ("contract", "master_policy", "web_search"):
            has_official_citation = True

        date_str = ""
        if c.get("effective_date"):
            date_str = f" | Effective: {c.get('effective_date')}"

        context_chunks.append(
            {
                "document_title": c.get("document_title", "Policy Document"),
                "source_type": f"{source_type.upper()}{date_str}",
                "uin": c.get("uin") or c.get("source_type", "N/A"),
                "section": c.get("section") or c.get("url", "General"),
                "clause_number": c.get("clause_number", "Clause"),
                "page_number": c.get("page_number", 1),
                "text": c.get("snippet", ""),
            }
        )

    # — Token-Budget Context Assembly —
    # Select chunks that fit within token budget; always cut at chunk boundaries.
    # Never truncates mid-sentence (critical for insurance clause citation integrity).
    budgeted_chunks, tokens_used, total_chunks = _build_context_with_budget(context_chunks)

    if total_chunks > 0 and tokens_used > 0:
        if len(budgeted_chunks) < total_chunks:
            logger.warning(
                "TOKEN_BUDGET_REACHED",
                used=len(budgeted_chunks),
                total=total_chunks,
                tokens=tokens_used,
            )
        else:
            logger.info(
                "CONTEXT_BUDGET_ASSEMBLED",
                chunks=len(budgeted_chunks),
                tokens=tokens_used,
                max_tokens=_MAX_CONTEXT_TOKENS,
            )

    extra_guidance = ""
    if state.query_type == "general_on_topic":
        extra_guidance += (
            "\n\nGENERAL CONCEPT GUIDANCE:\n"
            "The employee is asking for an insurance concept, vocabulary definition, or product architecture principle "
            "(e.g., ULIP, Term Insurance, Grace Period, Sum Assured, Incontestability, Free Look). "
            "Provide an authoritative, clear, and comprehensive explanation grounded in IRDAI regulatory frameworks "
            "and standard Indian life insurance principles. You do not need to require a specific policy document for this general inquiry."
        )

    if state.retrieval_quality == "thin":
        extra_guidance += (
            "\n\nTHIN CONTEXT CAVEAT (IRDAI Accuracy Directive):\n"
            "Only limited context chunks were verified for this query. "
            "Explicitly caveat your answer noting that it is derived from partial context, and advise the user "
            "to review the complete policy schedule or endorsement bond for exhaustive contractual clauses."
        )

    if state.calculation_result:
        extra_guidance += f"\n\nACTUARIAL CALCULATION RESULT (Ground Truth):\n{state.calculation_result}\nState these figures and regulatory terms with precision."

    if has_attachment_citation and has_official_citation:
        extra_guidance += (
            "\n\nCONFLICT CHECK (MANDATORY ACTIVE RESOLUTION):\n"
            "You have retrieved both customer policy documents and standard IRDAI / Master Policy catalog terms.\n"
            "If any numerical values, grace periods, free-look windows, or exclusions conflict, "
            "you MUST format both side-by-side in a comparative Markdown table with columns: "
            "| Provision | Uploaded Document Term | Standard Master / Regulatory Term | Resolution & Binding Hierarchy |\n"
            "Clearly note that signed policy endorsement schedules supersede generic product brochures."
        )

    for c in state.citations:
        if c.get("uin_warning"):
            extra_guidance += f"\n\nPOLICY DOCUMENT NOTICE:\n{c.get('uin_warning')}\n"
            break


    prompt_text = (
        build_query_prompt(
            query=state.sanitized_query,
            context_chunks=budgeted_chunks,
            product_code=state.product_code,
            state=state.state,
        )
        + extra_guidance
    )

    answer_text = ""
    if llm:
        try:
            messages = [
                SystemMessage(content=system_text),
                HumanMessage(content=prompt_text),
            ]
            response = await llm.ainvoke(messages)
            if hasattr(response, "content") and response.content:
                answer_text = str(response.content)
        except Exception as e:
            logger.warning("LLM_SYNTHESIS_ERROR", error=str(e))

    # Structured deterministic fallback if LLM offline
    if not answer_text:
        if state.calculation_result:
            cr = state.calculation_result
            tool_name = cr.get("tool", "")
            if "loan" in tool_name:
                answer_text = (
                    f"### Policy Loan Eligibility & Calculation\n\n"
                    f"- **Eligibility**: {'Eligible' if cr.get('eligible') else 'Not Eligible'}\n"
                    f"- **Maximum Loan Amount**: ₹{cr.get('max_loan_amount', 0):,.2f} ({cr.get('basis', '')})\n"
                    f"- **Interest Terms**: {cr.get('interest_terms', '')}\n"
                    f"- **Statutory Guidance**: {cr.get('statutory_note', '')}\n\n"
                    f"> ℹ️ *This is informational. The policy contract and official communication prevail.*"
                )
            elif "section_45" in tool_name:
                answer_text = (
                    f"### Section 45 Incontestability Assessment\n\n"
                    f"- **Status**: **{cr.get('legal_status', '')}**\n"
                    f"- **Time Elapsed**: {cr.get('years_elapsed', 0)} years since risk commencement/revival\n\n"
                    f"{cr.get('statutory_advisory', '')}\n\n"
                    f"> ℹ️ *This is informational. The policy contract and official communication prevail.*"
                )
            elif "grace" in tool_name:
                answer_text = (
                    f"### Grace Period & Revival Timelines\n\n"
                    f"- **Grace Period**: {cr.get('grace_period_days')} calendar days (ends {cr.get('grace_period_expiry')})\n"
                    f"- **Revival Deadline**: {cr.get('revival_deadline')} (5 consecutive years from FUP)\n\n"
                    f"**Guidelines:**\n"
                    + "\n".join(f"- {r}" for r in cr.get("requirements", []))
                    + "\n\n"
                    "> ℹ️ *This is informational. The policy contract and official communication prevail.*"
                )
            else:
                answer_text = (
                    f"### Policy Assessment\n\n{cr}\n\n"
                    f"> ℹ️ *This is informational. The policy contract and official communication prevail.*"
                )
        elif state.citations:
            top_c = state.citations[0]
            answer_text = (
                f"### Policy Servicing Guidance\n\n"
                f"Regarding your query: **'{state.sanitized_query}'**\n\n"
                f"Under contract terms (**{top_c.get('document_title', 'Master Policy')}**, {top_c.get('section', '')}):\n\n"
                f'> *"{top_c.get("snippet", "")}"*\n\n'
                f"- **Statutory Incontestability**: Under Section 45 of Insurance Act 1938, no policy may be challenged after 3 years.\n"
                f"- **IRDAI Compliance**: Servicing adheres to Protection of Policyholders' Interests Regulations.\n\n"
                f"> ℹ️ *This is informational. The policy contract and official communication prevail.*"
            )
        else:
            answer_text = (
                f"### Policy Servicing Information\n\n"
                f"Regarding: **'{state.sanitized_query}'**\n\n"
                f"Under standard IRDAI guidelines, life insurance policies benefit from statutory non-repudiation under Section 45 after 3 years, "
                f"a 15 to 30 day grace period, and revival within 5 years from FUP.\n\n"
                f"> ℹ️ *This is informational. The policy contract and official communication prevail.*"
            )

    return {
        "raw_answer": answer_text,
        "model_used": llm_model,
    }
