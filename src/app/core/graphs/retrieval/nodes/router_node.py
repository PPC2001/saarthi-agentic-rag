"""Router node implementing LLM-based intent classification for SBI Life corporate knowledge.

3-Tier Defense Architecture:
  Tier 1: Domain Vector Filter (~1ms, zero LLM) — blocks clearly off-topic queries
          using cosine similarity against precomputed insurance domain prototypes
  Tier 2: LLM Router (~300ms) — 4-taxonomy classification for queries that passed Tier 1
  Tier 3: FlashRank CRAG Grader (~50ms) — per-chunk relevance scoring after retrieval
"""

import json
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.core.document_processing.session_docs import SessionDocumentStore
from app.core.graphs.retrieval.state import (
    GRACE_PERIOD_CALC_REGEX,
    LOAN_CALC_REGEX,
    SECTION_45_CALC_REGEX,
    SURRENDER_CALC_REGEX,
    AgenticRAGState,
    get_configurable,
)
from app.core.guardrails.domain_filter import check_domain_relevance
from app.core.guardrails.input_guardrails import (
    OFF_TOPIC_GUARDRAIL_RESPONSE,
    check_prompt_injection,
    mask_pii,
)
from app.core.retrieval.classifier import (
    GREETING_RESPONSE,
    QueryIntent,
    classify_query,
)
from app.observability.logging import get_logger
from app.observability.metrics import record_domain_filter_block

logger = get_logger("router_node")

ROUTER_SYSTEM_PROMPT = """You are the structural intent classification engine for Saarthi AI's life insurance policy servicing knowledge system. Your sole responsibility is to analyze an incoming user query and classify it into exactly one of four predefined categories. You must output your decision strictly as a structured JSON object matching the provided schema. Do not include any introductory remarks, explanations, or conversational filler.

### THE TAXONOMY CATEGORIES

1. "greeting"
   - Criteria: Basic pleasantries, system handshakes, or initial introductory statements.
   - Examples: "hello", "hi", "good morning", "hey there", "who are you?", "help me"

2. "off_topic"
   - Criteria: Requests completely unrelated to life insurance, policy frameworks, underwriting rules, compliance, or IRDAI circulars.
   - Examples: "how do I cook butter chicken?", "write a python script to reverse a linked list", "what is the stock price of Apple?", "who won the cricket match yesterday?"

3. "general_on_topic"
   - Criteria: Insurance-focused queries seeking generic conceptual definitions, industry vocabulary, or basic principles that DO NOT require scanning specific product brochures or policy documents.
   - Examples: "What is the basic difference between a term plan and a ULIP?", "Define what a premium grace period means in insurance.", "What does the term 'Sum Assured' stand for?"

4. "structured_rag"
   - Criteria: High-precision inquiries requiring specific document context, premium matrices, lock-in clauses, surrender charge rules, or active IRDAI regulatory updates. Any query mentioning specific product lines, policy years, or compliance lookups falls here.
   - Examples: "What is the surrender penalty for Smart Platina Plus in the 3rd year?", "Show me the entry age limits for the eShield Next policy.", "What are the latest free-look changes issued in the recent IRDAI master circular?"

### CRITICAL GUARDRAIL CONSTRAINTS
- Strict Scope: If a query contains mixed intent (e.g., "Hello, what is the surrender fee for Smart Platina Plus?"), you MUST prioritize the high-stakes operational intent and route it to "structured_rag".
- Ambiguity Rule: If an insurance term is mentioned alongside an off-topic request (e.g., "I need a policy on how to build a website"), flag it as "off_topic".
- Zero Speculation: Do not attempt to answer the question. Only categorize it.

### SUB-INTENT TAXONOMY (For general_on_topic and structured_rag)
- "claim_enquiry": Death claim, maturity claim, claim repudiation, documents required, settlement ratios.
- "product_comparison": Comparing two or more insurance products, term vs ULIP differences, plan features.
- "renewal_servicing": Premium payment, revival of lapsed policy, grace period, ECS/NACH, change of nomination/address.
- "regulatory_lookup": IRDAI circulars, Ombudsman, Section 45 incontestability, master circular guidelines.
- "underwriting_query": Medical checkups, financial underwriting, BMI limits, occupation loadings, smoker rates.
- "tax_query": Section 80C, 80D, 10(10D), GST on premiums, TDS implications.
- "general": Basic conceptual definitions or general insurance inquiries.

### OUTPUT JSON SCHEMA EXPECTED
{
  "intent": "greeting" | "off_topic" | "general_on_topic" | "structured_rag",
  "sub_intent": "claim_enquiry" | "product_comparison" | "renewal_servicing" | "regulatory_lookup" | "underwriting_query" | "tax_query" | "general",
  "reasoning": "A concise one-sentence extraction of the query attributes that triggered this routing path."
}"""


def detect_sub_intent(query: str) -> str:
    """Heuristic sub-intent classification for life insurance queries."""
    q = query.lower()
    if re.search(r"(?i)\b(compare|versus|\bvs\b|difference\s+between|better\s+plan|which\s+(is\s+better|one))\b", q):
        return "product_comparison"
    if re.search(r"(?i)\b(claim|death\s+claim|maturity\s+claim|repudiat|settlement\s+ratio|nominee\s+payout|claim\s+form)\b", q):
        return "claim_enquiry"
    if re.search(r"(?i)\b(reviv|fup|grace|nach|ecs|mandate|renewal|premium\s+due|alteration|nomination|assignment|lapse)\b", q):
        return "renewal_servicing"
    if re.search(r"(?i)\b(tax|80c|80d|10\(10d\)|tds|gst|rebate|deduction)\b", q):
        return "tax_query"
    if re.search(r"(?i)\b(medical|underwriting|financial\s+underwriting|bmi|smoker|habit|morbid|tele-mer|peme|loading)\b", q):
        return "underwriting_query"
    if re.search(r"(?i)\b(circular|master\s+circular|ombudsman|bima\s+bharosa|section\s+45|regulat)\b", q):
        return "regulatory_lookup"
    return "general"


class IntentClassificationOutput(BaseModel):
    """Structured response from LLM intent classification router."""

    intent: Literal["greeting", "off_topic", "general_on_topic", "structured_rag"]
    sub_intent: Literal[
        "claim_enquiry",
        "product_comparison",
        "renewal_servicing",
        "regulatory_lookup",
        "underwriting_query",
        "tax_query",
        "general",
    ] = "general"
    reasoning: str = Field(
        description="A concise one-sentence extraction of the query attributes that triggered this routing path."
    )



async def classify_query_with_llm(
    query: str,
    llm: Any,
) -> IntentClassificationOutput | None:
    """Invoke LLM to classify query into one of the four corporate categories."""
    if not llm:
        return None

    try:
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"Query to classify:\n{query}"),
        ]
        response = await llm.ainvoke(messages)
        content = getattr(response, "content", "")
        if not content:
            return None

        # Clean JSON markdown fences if present
        text = str(content).strip()
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            text = json_match.group(0)
        elif text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)
        return IntentClassificationOutput.model_validate(data)
    except Exception as e:
        logger.warning("LLM_ROUTER_CLASSIFICATION_FAILED", error=str(e))
        return None


async def router_node(
    state: AgenticRAGState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Execute structural intent classification using 3-tier defense architecture.

    Tier 1: Domain Vector Filter (sub-5ms, zero LLM calls)
    Tier 2: LLM Router Node (4-taxonomy corporate classification)
    """
    clean_query = state.query.strip()

    # 1. Security Check: Prompt Injection / Jailbreaks
    is_injection, injection_reason = check_prompt_injection(clean_query)
    if is_injection:
        logger.warning("SECURITY_PROMPT_INJECTION_BLOCKED", reason=injection_reason)
        return {
            "is_blocked": True,
            "sanitized_query": clean_query,
            "query_type": "security_blocked",
            "fast_path_answer": (
                "⚠️ **Security Notice**: Your query triggered an automated security guardrail "
                "(Instruction override detected). Please submit a valid insurance inquiry."
            ),
            "final_answer": (
                "⚠️ **Security Notice**: Your query triggered an automated security guardrail "
                "(Instruction override detected). Please submit a valid insurance inquiry."
            ),
            "model_used": "guardrail_filter",
        }

    # 2. PII Scrubbing
    sanitized_query, redactions = mask_pii(clean_query)

    # 3. Conversational Handshake Fast-Path (sub-1ms regex match for greeting / help / gratitude)
    fast_intent, fast_resp = classify_query(
        sanitized_query, has_attachment=state.has_attachment
    )
    if fast_intent in (QueryIntent.GREETING, QueryIntent.HELP, QueryIntent.GRATITUDE):
        logger.info("CONVERSATIONAL_HANDSHAKE_FASTPATH", intent=fast_intent.value)
        return {
            "sanitized_query": sanitized_query,
            "query_type": "greeting"
            if fast_intent in (QueryIntent.GREETING, QueryIntent.HELP)
            else "gratitude",
            "fast_path_answer": fast_resp or GREETING_RESPONSE,
            "final_answer": fast_resp or GREETING_RESPONSE,
            "model_used": "intent_router",
        }

    # 4. TIER 1: Domain Vector Anchor Filter (sub-5ms, zero LLM cost)
    # Pre-compute query embedding for domain check if embedder is available
    cached_query_embedding: list[float] | None = None
    cfg = get_configurable(config)
    embedder = cfg.get("embedder")
    if embedder is not None:
        try:
            cached_query_embedding = await embedder.embed_query(sanitized_query)
            is_in_domain, domain_score, domain_label = check_domain_relevance(
                query=sanitized_query,
                query_embedding=cached_query_embedding,
                embedder=embedder,
            )
            if not is_in_domain:
                record_domain_filter_block()
                logger.warning(
                    "DOMAIN_VECTOR_FILTER_BLOCKED",
                    score=round(domain_score, 4),
                    label=domain_label,
                )
                return {
                    "is_blocked": True,
                    "sanitized_query": sanitized_query,
                    "query_type": "off_topic",
                    "fast_path_answer": OFF_TOPIC_GUARDRAIL_RESPONSE,
                    "final_answer": OFF_TOPIC_GUARDRAIL_RESPONSE,
                    "model_used": "domain_vector_filter",
                }
        except Exception as domain_err:
            # Fail open — proceed to LLM router on embedding errors
            logger.warning("DOMAIN_VECTOR_FILTER_SKIPPED", error=str(domain_err))

    # 5. Deterministic Fast-Path: Actuarial Calculation Regex Check (sub-1ms, zero LLM cost)
    clean_lower = sanitized_query.lower()
    if (
        LOAN_CALC_REGEX.search(clean_lower)
        or SURRENDER_CALC_REGEX.search(clean_lower)
        or SECTION_45_CALC_REGEX.search(clean_lower)
        or GRACE_PERIOD_CALC_REGEX.search(clean_lower)
    ):
        logger.info("ACTUARIAL_CALCULATION_FASTPATH_ROUTED", query_prefix=sanitized_query[:60])
        actuarial_sub = (
            "claim_enquiry"
            if SECTION_45_CALC_REGEX.search(clean_lower)
            else "renewal_servicing"
        )
        return {
            "sanitized_query": sanitized_query,
            "query_type": "actuarial_calculation",
            "sub_intent": actuarial_sub,
            "query_embedding": cached_query_embedding,
        }

    # 6. TIER 2: LLM-based Structural Intent Classification
    llm = cfg.get("llm")

    classification = await classify_query_with_llm(sanitized_query, llm)

    # Fallback to heuristic classifier if LLM unavailable or offline
    if not classification:
        has_active_attachment = state.has_attachment or (
            state.session_id is not None
            and SessionDocumentStore.has_documents(state.session_id, user_id=state.user_id)
        )
        fallback_intent, pre_computed = classify_query(
            sanitized_query, has_attachment=has_active_attachment
        )
        intent_val = fallback_intent.value
        if intent_val in (
            QueryIntent.GREETING.value,
            QueryIntent.HELP.value,
            QueryIntent.GRATITUDE.value,
        ):
            cat = "greeting"
        elif intent_val == QueryIntent.OFF_TOPIC.value:
            cat = "off_topic"
        else:
            cat = "structured_rag"
        classification = IntentClassificationOutput(
            intent=cat,  # type: ignore[arg-type]
            sub_intent=detect_sub_intent(sanitized_query),  # type: ignore[arg-type]
            reasoning="Heuristic fallback classification due to LLM router absence.",
        )

    logger.info(
        "INTENT_ROUTED",
        intent=classification.intent,
        sub_intent=classification.sub_intent,
        reasoning=classification.reasoning,
    )

    resolved_sub_intent = (
        classification.sub_intent
        if classification.sub_intent != "general"
        else detect_sub_intent(sanitized_query)
    )

    # 7. Route based on the 4 Taxonomy Categories

    # Category 1: "greeting"
    if classification.intent == "greeting":
        return {
            "sanitized_query": sanitized_query,
            "query_type": "greeting",
            "sub_intent": "general",
            "fast_path_answer": GREETING_RESPONSE,
            "final_answer": GREETING_RESPONSE,
            "model_used": "intent_router",
        }

    # Category 2: "off_topic"
    if classification.intent == "off_topic":
        return {
            "is_blocked": True,
            "sanitized_query": sanitized_query,
            "query_type": "off_topic",
            "sub_intent": "general",
            "fast_path_answer": OFF_TOPIC_GUARDRAIL_RESPONSE,
            "final_answer": OFF_TOPIC_GUARDRAIL_RESPONSE,
            "model_used": "guardrail_filter",
        }

    # Category 3: "general_on_topic" (Generic insurance concepts, definitions, vocabulary)
    if classification.intent == "general_on_topic":
        return {
            "sanitized_query": sanitized_query,
            "query_type": "general_on_topic",
            "sub_intent": resolved_sub_intent,
            "query_embedding": cached_query_embedding,
            "fast_path_answer": None,
            "final_answer": "",
            "is_blocked": False,
        }

    # Category 4: "structured_rag" (High precision, specific matrices, documents)
    # Web search keyword triggers
    if re.search(
        r"(?i)\b(latest\s*circular|irdai(\.gov\.in)?|portal|daily\s*nav|public\s*disclosure|solvency\s*ratio|ombudsman)\b",
        clean_lower,
    ):
        return {
            "sanitized_query": sanitized_query,
            "query_type": "web_search",
            "sub_intent": "regulatory_lookup",
            "query_embedding": cached_query_embedding,
            "fast_path_answer": None,
            "final_answer": "",
            "is_blocked": False,
        }

    # Session attachment active
    has_active_attachment = state.has_attachment or (
        state.session_id is not None
        and SessionDocumentStore.has_documents(state.session_id, user_id=state.user_id)
    )
    if has_active_attachment and (
        re.search(
            r"(?i)\b(attached|attachment|uploaded|this\s*document|my\s*policy)\b", clean_lower
        )
        or len(clean_lower.split()) <= 4
    ):
        return {
            "sanitized_query": sanitized_query,
            "query_type": "attachment_rag",
            "sub_intent": resolved_sub_intent,
            "query_embedding": cached_query_embedding,
            "fast_path_answer": None,
            "final_answer": "",
            "is_blocked": False,
        }

    return {
        "sanitized_query": sanitized_query,
        "query_type": "structured_rag",
        "sub_intent": resolved_sub_intent,
        "query_embedding": cached_query_embedding,
        "fast_path_answer": None,
        "final_answer": "",
        "is_blocked": False,
    }

