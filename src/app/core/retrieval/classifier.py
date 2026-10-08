"""Query intent classifier for Saarthi AI - Life Insurance Policy Assistant.

Sub-millisecond intent detection to route:
- Greetings & conversational salutations (hi, hello, namaste, etc.)
- Gratitude & farewells (thanks, bye, etc.)
- System identity and help inquiries (who are you, what can you do, etc.)
- Session Attachment RAG (uploaded policy bond, schedule, circular)
- Live Web Retrieval (IRDAI circulars, portal disclosures, daily NAVs, claim ratios)
- Standard Policy Contract Inquiries
"""

import re
from enum import StrEnum

from app.core.guardrails.input_guardrails import (
    OFF_TOPIC_GUARDRAIL_RESPONSE,
    check_off_topic,
)
from app.observability.logging import get_logger

logger = get_logger("classifier")

GREETING_REGEX = re.compile(
    r"^(h+[iey]+(l+o+)?|namaste|namaskar|good\s*(morning|afternoon|evening|day)|greetings|halo|hola)"
    r"(\s+(saarthi|assistant|there|team|sir|madam|ai|bot|ji|all|friend))?[!.,? ]*$",
    re.IGNORECASE,
)

HELP_IDENTITY_REGEX = re.compile(
    r"^(who\s*are\s*you|what\s*can\s*you\s*do|how\s*(can|do)\s*you\s*(help|assist)(\s*me)?|what\s*is\s*this|help(\s*me)?|what\s*do\s*you\s*know|about\s*you"
    r"|introduce\s*yourself|what\s*is\s*saarthi(\s*ai)?|how\s*to\s*use\s*this)[!.,? ]*$",
    re.IGNORECASE,
)

COMBINED_GREETING_HELP_REGEX = re.compile(
    r"^(h+[iey]+(l+o+)?|namaste|namaskar|good\s*(morning|afternoon|evening|day)|greetings)"
    r"(\s+(saarthi|assistant|there|team|sir|madam|ai|bot|ji|all|friend))?[!.,? ]+"
    r"(who\s*are\s*you|what\s*can\s*you\s*do|how\s*(can|do)\s*you\s*(help|assist)(\s*me)?|what\s*is\s*this|help(\s*me)?|what\s*do\s*you\s*know|about\s*you"
    r"|introduce\s*yourself|what\s*is\s*saarthi(\s*ai)?)[!.,? ]*$",
    re.IGNORECASE,
)

GRATITUDE_REGEX = re.compile(
    r"^(thank(s|\s+you|\s+u)|thx|ok(ay)?(\s+thanks|\s+thank\s+you)?|got\s+it|bye|goodbye|see\s+you)[!., ]*$",
    re.IGNORECASE,
)

WEB_SEARCH_REGEX = re.compile(
    r"(?i)\b(latest\s*circular|irdai(\.gov\.in)?|official\s*website|portal|claim\s*(settlement\s*)?ratio|daily\s*nav|public\s*disclosure|solvency\s*ratio|bima\s*bharosa|ombudsman\s*rules|new\s*rules|recent\s*guidelines|online\s*portal)\b"
)

ATTACHMENT_REF_REGEX = re.compile(
    r"(?i)\b(attached|attachment|uploaded|this\s*document|my\s*policy|schedule|clause\s*in\s*my|page\s*[0-9]+)\b"
)


class QueryIntent(StrEnum):
    """Categorized intent of the incoming user query."""

    GREETING = "greeting"
    HELP = "help"
    GRATITUDE = "gratitude"
    ATTACHMENT_RAG = "attachment_rag"
    WEB_SEARCH = "web_search"
    OFF_TOPIC = "off_topic"
    POLICY_QUERY = "policy_query"


GREETING_RESPONSE = """Hello! I am **Saarthi AI** — your trusted Life Insurance Policy Assistant.

I can assist you with verified contractual, regulatory, and servicing guidance
How may I assist you with your policy inquiry today?"""

GRATITUDE_RESPONSE = """You are very welcome! If you need any further assistance with policy clauses, attached documents, or IRDAI guidelines, feel free to ask anytime. Have a wonderful day!"""


def classify_query(query: str, has_attachment: bool = False) -> tuple[QueryIntent, str | None]:
    """Classify user query and route to the optimal execution path.

    Returns:
        (QueryIntent, pre_computed_response or None)
    """
    clean_q = query.strip()
    # 1. Greetings & Salutations
    if GREETING_REGEX.match(clean_q) or COMBINED_GREETING_HELP_REGEX.match(clean_q):
        logger.info("QUERY_INTENT_CLASSIFIED", intent="GREETING", query_prefix=clean_q[:80])
        return QueryIntent.GREETING, GREETING_RESPONSE

    # 2. Identity & Capabilities
    if HELP_IDENTITY_REGEX.match(clean_q):
        logger.info("QUERY_INTENT_CLASSIFIED", intent="HELP", query_prefix=clean_q[:80])
        return QueryIntent.HELP, GREETING_RESPONSE

    # 3. Gratitude & Farewells
    if GRATITUDE_REGEX.match(clean_q):
        logger.info("QUERY_INTENT_CLASSIFIED", intent="GRATITUDE", query_prefix=clean_q[:80])
        return QueryIntent.GRATITUDE, GRATITUDE_RESPONSE

    # 4. Attachment-specific query
    if has_attachment and (ATTACHMENT_REF_REGEX.search(clean_q) or len(clean_q.split()) <= 4):
        logger.info("QUERY_INTENT_CLASSIFIED", intent="ATTACHMENT_RAG", query_prefix=clean_q[:80])
        return QueryIntent.ATTACHMENT_RAG, None

    # 5. Live Web & Regulatory Portal Search
    if WEB_SEARCH_REGEX.search(clean_q):
        logger.info("QUERY_INTENT_CLASSIFIED", intent="WEB_SEARCH", query_prefix=clean_q[:80])
        return QueryIntent.WEB_SEARCH, None

    # 6. Off-Topic Restriction Guardrail
    is_off_topic, off_topic_reason = check_off_topic(clean_q, has_attachment=has_attachment)
    if is_off_topic:
        logger.warning(
            "QUERY_INTENT_CLASSIFIED", intent="OFF_TOPIC", reason=off_topic_reason, query_prefix=clean_q[:80]
        )
        return QueryIntent.OFF_TOPIC, OFF_TOPIC_GUARDRAIL_RESPONSE

    # 7. Standard Policy Contract RAG
    logger.info("QUERY_INTENT_CLASSIFIED", intent="POLICY_QUERY", query_prefix=clean_q[:80])
    return QueryIntent.POLICY_QUERY, None
