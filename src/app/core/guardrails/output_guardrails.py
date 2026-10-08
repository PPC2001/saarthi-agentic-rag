"""Output Guardrails for Indian Life Insurance Assistant.

Enforces:
1. No Unauthorized Claim Determination or Legal Guarantee:
   Ensures model outputs do not promise claim payouts or guarantee approval/rejection.
2. Mandatory IRDAI ULIP Risk Disclosures:
   Appends statutory market-risk warnings whenever Unit-Linked products or NAVs are discussed.
3. Output PII Scrubbing:
   Ensures model responses never inadvertently leak sensitive IDs.
"""

import re

from app.core.guardrails.input_guardrails import mask_pii
from app.observability.logging import get_logger

logger = get_logger("output_guardrails")

ULIP_KEYWORD_REGEX = re.compile(
    r"(?i)\b(ulips?|unit\s*linked|fund\s*value|daily\s*nav|navs?\b|equity\s*fund|growth\s*fund|balanced\s*fund|debt\s*fund|discontinued\s*policy\s*fund|fund\s*switch)\b"
)
CLAIM_PROMISE_REGEX = re.compile(
    r"(?i)\b(claim\s*(will|is\s*guaranteed\s*to)\s*(be\s*)?(approved|admitted|paid|settled|rejected))\b"
)
MISSELLING_PROHIBITED_REGEX = re.compile(
    r"(?i)\b((guaranteed|assured|risk-?free|100%\s*guaranteed)\s+(returns?|growth|capital|profit))\b"
)

ULIP_STATUTORY_DISCLOSURE = (
    "\n\n> ⚠️ **Statutory IRDAI Notice on ULIPs**: Unit Linked Insurance Products (ULIPs) are different from "
    "traditional insurance products and are subject to market risks. The investment risk in the investment portfolio "
    "is borne by the policyholder. Unit values (NAVs) fluctuate with market conditions."
)

MISSELLING_STATUTORY_CORRECTION = (
    "\n\n> ⚖️ **IRDAI Statutory Advisory on Market-Linked Products**: Life insurance products linked to capital "
    "markets (ULIPs/pension funds) do not offer guaranteed returns. Returns depend on market performance, and "
    "past performance is not indicative of future results."
)

CLAIM_ADMISSIBILITY_NOTICE = (
    "\n\n> 📋 **Notice on Claim Settlement**: Final claim admissibility is subject to submission of required "
    "statutory proofs, claims committee review, Section 45 verification of the Insurance Act 1938, and "
    "terms of the original policy bond. This AI guidance is for servicing reference only."
)

POLICY_INFORMATIONAL_DISCLAIMER = "\n\n> ℹ️ *This is informational. The policy contract and SBI Life's official communication prevail.*"

EXEMPT_FLOW_MARKERS = (
    "I can only help with life insurance topics.",
    "[SECURITY_BLOCKED]",
    "Security Notice",
    "Hello! I am **Saarthi AI**",
    "Hello! I am **Sarathi AI**",
    "You are welcome!",
)


def is_ulip_product_discussion(text: str) -> bool:
    """Determine whether the response genuinely discusses Unit-Linked products or fund NAVs.

    Strips negative assertions (e.g., 'not a ULIP', 'unlike a ULIP', 'non-linked') so that
    traditional, non-linked policies (e.g., Saral Swadhan, Whole Life) do not trigger false
    statutory disclaimers.
    """
    stripped = re.sub(
        r"(?i)\b(not\s+(an?\s+)?|neither\s+(an?\s+)?|non-?linked\s*,?\s*(and\s+)?not\s+(an?\s+)?|unlike\s+(an?\s+)?)(ulips?|unit\s*linked\s*(insurance|plan|product|policy)?)\b",
        "",
        text,
    )
    return bool(ULIP_KEYWORD_REGEX.search(stripped))


def enforce_output_guardrails(answer: str, query: str = "") -> str:
    """Validate and augment assistant answers with required statutory safeguards."""
    # 1. PII Scrubbing on output
    sanitized_answer, redactions = mask_pii(answer)
    if redactions:
        logger.warning(
            "OUTPUT_PII_SCRUBBED",
            count=len(redactions),
        )

    # Early exit for exempt flows: greetings, off-topic rejections, security notices
    is_exempt_flow = any(marker in sanitized_answer for marker in EXEMPT_FLOW_MARKERS)
    if is_exempt_flow:
        return sanitized_answer

    # 2. Check for genuine ULIP product/NAV mentions and ensure disclosure
    if (
        is_ulip_product_discussion(sanitized_answer)
        and "Statutory IRDAI Notice on ULIPs" not in sanitized_answer
    ):
        logger.info("ULIP_STATUTORY_NOTICE_APPENDED")
        sanitized_answer += ULIP_STATUTORY_DISCLOSURE

    # 3. Check for prohibited guaranteed return mis-selling language in market-linked context
    if (
        MISSELLING_PROHIBITED_REGEX.search(sanitized_answer)
        and "IRDAI Statutory Advisory on Market-Linked Products" not in sanitized_answer
    ):
        logger.warning("MISSELLING_LANGUAGE_DETECTED_CORRECTIVE_NOTICE_APPENDED")
        sanitized_answer += MISSELLING_STATUTORY_CORRECTION

    # 4. Check for definitive claim guarantees and ensure notice
    if (
        CLAIM_PROMISE_REGEX.search(sanitized_answer)
        and "Notice on Claim Settlement" not in sanitized_answer
    ):
        logger.warning("CLAIM_SETTLEMENT_DISCLAIMER_APPENDED")
        sanitized_answer += CLAIM_ADMISSIBILITY_NOTICE

    # 4. Mandatory Statutory Policy Disclaimer for policy-specific responses
    if "The policy contract and" not in sanitized_answer:
        if "sbi" in query.lower() or "sbi" in sanitized_answer.lower():
            sanitized_answer += "\n\n> ℹ️ *This is informational. The policy contract and SBI Life's official communication prevail.*"
        else:
            sanitized_answer += "\n\n> ℹ️ *This is informational. The policy contract and official communication prevail.*"

    return sanitized_answer
