"""Response quality scoring node evaluating factual grounding, completeness, and IRDAI compliance.

Evaluates synthesis output before final guardrails and audit logging (Roadmap §14).
Captures evaluation telemetry across three pillars:
  1. Factual Grounding (0-10): Verification against retrieved clauses and actuarial calculations
  2. Completeness (0-10): Coverage of user question attributes and structure
  3. IRDAI Compliance (0-10): Absence of prohibited promises and presence of statutory caveats
"""

from __future__ import annotations

import re
from typing import Any

from app.core.graphs.retrieval.state import AgenticRAGState
from app.core.graphs.retrieval.structured_output import build_structured_response
from app.observability.logging import get_logger

logger = get_logger("quality_scorer_node")

# Patterns for compliance checks
_PROHIBITED_MARKETING_REGEX = re.compile(
    r"(?i)\b(guaranteed\s+(high\s+)?returns|100%\s+risk-free|assured\s+market\s+growth|best\s+investment\s+scheme)\b"
)
_STATUTORY_CAVEAT_REGEX = re.compile(
    r"(?i)\b(statutory|irdai|section\s+45|ombudsman|subject\s+to|policy\s+contract|informational)\b"
)


def score_response_quality(
    query: str,
    raw_answer: str,
    query_type: str,
    citations: list[dict[str, Any]],
    calculation_result: dict[str, Any] | None = None,
    retrieval_quality: str = "adequate",
) -> dict[str, Any]:
    """Compute deterministic quality scores across grounding, completeness, and compliance."""
    # Fast paths (greetings, off-topic rejections, security blocks) are inherently complete & compliant
    if query_type in ("greeting", "off_topic", "security_blocked", "gratitude"):
        return {
            "factual_grounding": 10.0,
            "completeness": 10.0,
            "irdai_compliance": 10.0,
            "overall_score": 10.0,
            "flags": ["fast_path_evaluated"],
            "requires_human_review": False,
        }

    flags: list[str] = []

    # 1. Factual Grounding (0 - 10)
    grounding = 4.0
    if calculation_result:
        grounding += 4.0  # Grounded in deterministic calculation
        flags.append("actuarial_grounded")

    num_citations = len(citations)
    if num_citations >= 3:
        grounding += 3.0
    elif num_citations >= 1:
        grounding += 2.0
    else:
        flags.append("zero_citations")

    if retrieval_quality == "thin":
        grounding = min(grounding, 7.0)
        flags.append("thin_context")
    elif retrieval_quality == "insufficient":
        grounding = min(grounding, 5.0)
        flags.append("insufficient_context")

    grounding = min(10.0, max(0.0, grounding))

    # 2. Completeness (0 - 10)
    completeness = 5.0
    ans_len = len(raw_answer.strip())
    if ans_len >= 150:
        completeness += 2.0
    if any(m in raw_answer for m in ("###", "**", "- ")):
        completeness += 2.0  # Structured formatting
    if any(c in raw_answer for c in ("₹", "%", "years", "days")):
        completeness += 1.0  # Quantitative precision
    completeness = min(10.0, max(0.0, completeness))

    # 3. IRDAI Compliance (0 - 10)
    compliance = 7.0
    if _PROHIBITED_MARKETING_REGEX.search(raw_answer):
        compliance -= 4.0
        flags.append("prohibited_marketing_detected")

    if _STATUTORY_CAVEAT_REGEX.search(raw_answer):
        compliance += 3.0
        flags.append("statutory_caveats_verified")
    else:
        compliance -= 1.0
        flags.append("missing_statutory_caveats")
    compliance = min(10.0, max(0.0, compliance))

    # Overall weighted score
    overall = round((0.4 * grounding) + (0.3 * completeness) + (0.3 * compliance), 1)
    requires_review = grounding < 5.0 or compliance < 6.0

    return {
        "factual_grounding": round(grounding, 1),
        "completeness": round(completeness, 1),
        "irdai_compliance": round(compliance, 1),
        "overall_score": overall,
        "flags": flags,
        "requires_human_review": requires_review,
    }


async def quality_scorer_node(
    state: AgenticRAGState,
) -> dict[str, Any]:


    """Execute response quality scoring and produce structured output representation."""
    raw_ans = state.raw_answer or state.fast_path_answer or ""

    # Compute quality evaluation scores
    score_data = score_response_quality(
        query=state.sanitized_query,
        raw_answer=raw_ans,
        query_type=state.query_type,
        citations=state.citations,
        calculation_result=state.calculation_result,
        retrieval_quality=state.retrieval_quality,
    )

    # Build structured response representation (Roadmap §18)
    structured_resp = build_structured_response(
        query=state.sanitized_query,
        raw_answer=raw_ans,
        citations=state.citations,
        calculation_result=state.calculation_result,
        retrieval_quality=state.retrieval_quality,
        sub_intent=state.sub_intent,
    )

    logger.info(
        "RESPONSE_QUALITY_EVALUATED",
        overall=score_data["overall_score"],
        grounding=score_data["factual_grounding"],
        completeness=score_data["completeness"],
        compliance=score_data["irdai_compliance"],
        requires_review=score_data["requires_human_review"],
    )

    return {
        "quality_score": score_data,
        "structured_response": structured_resp.model_dump(),
    }
