"""Unit tests for response quality scorer node and structured output generation."""

import pytest

from app.core.graphs.retrieval.nodes.quality_scorer import (
    quality_scorer_node,
    score_response_quality,
)
from app.core.graphs.retrieval.state import AgenticRAGState
from app.core.graphs.retrieval.structured_output import build_structured_response


def test_score_response_quality_fast_path():
    """Verify fast path queries receive full 10.0 score with no human review."""
    score = score_response_quality(
        query="Hello",
        raw_answer="Namaste! How can I assist you?",
        query_type="greeting",
        citations=[],
    )
    assert score["overall_score"] == 10.0
    assert score["requires_human_review"] is False
    assert "fast_path_evaluated" in score["flags"]


def test_score_response_quality_actuarial():
    """Verify grounded actuarial calculation receives high grounding score."""
    calc_res = {
        "tool": "calculate_policy_loan",
        "eligible": True,
        "max_loan_amount": 400000.0,
    }
    score = score_response_quality(
        query="What is my policy loan amount?",
        raw_answer=(
            "### Policy Loan Calculation\n\n"
            "- **Maximum Loan Amount**: ₹4,00,000.00\n"
            "- **Statutory Guidance**: In accordance with IRDAI guidelines, loan interest is 9.5% p.a.\n\n"
            "> ℹ️ *This is informational. The policy contract and official communication prevail.*"
        ),
        query_type="actuarial_calculation",
        citations=[{"document_title": "Master Policy Bond"}],
        calculation_result=calc_res,
    )
    assert score["factual_grounding"] >= 8.0
    assert score["irdai_compliance"] >= 9.0
    assert "actuarial_grounded" in score["flags"]
    assert score["requires_human_review"] is False


def test_score_response_quality_prohibited_marketing():
    """Verify prohibited promotional claims trigger compliance penalty and human review."""
    score = score_response_quality(
        query="Is this the best ULIP?",
        raw_answer="You will get guaranteed high returns and 100% risk-free profits in our fund.",
        query_type="structured_rag",
        citations=[],
    )
    assert score["irdai_compliance"] <= 4.0
    assert "prohibited_marketing_detected" in score["flags"]
    assert score["requires_human_review"] is True


def test_build_structured_response():
    """Verify structured response parser correctly parses markdown tables and bullets."""
    sample_answer = (
        "Under IRDAI Master Circular guidelines, the grace period is 30 days for non-monthly modes.\n\n"
        "Key contractual rules:\n"
        "- Policy remains in full force during grace period.\n"
        "- Death claims during grace period are paid after deducting overdue premium.\n\n"
        "| Provision | Uploaded Term | Master Term | Resolution |\n"
        "| Grace Window | 15 Days | 30 Days | Signed schedule prevails |\n\n"
        "> Statutory Notice: Under Section 45 of Insurance Act 1938, policies are incontestable after 3 years."
    )
    citations = [
        {
            "document_title": "SBI Life Saral Swadhan Bond",
            "section": "Section 4 - Grace Period",
            "clause_number": "4.1",
            "uin": "111N092V03",
        }
    ]

    structured = build_structured_response(
        query="What happens during grace period?",
        raw_answer=sample_answer,
        citations=citations,
        sub_intent="renewal_servicing",
    )
    assert "30 days" in structured.summary
    assert len(structured.key_provisions) >= 2
    assert structured.comparison_table is not None
    assert len(structured.comparison_table) == 1
    assert structured.comparison_table[0]["Provision"] == "Grace Window"
    assert structured.sub_intent == "renewal_servicing"
    assert structured.regulatory_citations[0]["uin"] == "111N092V03"


@pytest.mark.asyncio
async def test_quality_scorer_node_execution():
    """Verify quality_scorer_node returns quality_score and structured_response in state update."""
    state = AgenticRAGState(
        query="What is the grace period for annual premium?",
        sanitized_query="What is the grace period for annual premium?",
        query_type="structured_rag",
        sub_intent="renewal_servicing",
        raw_answer=(
            "### Grace Period Details\n\n"
            "For policies with annual frequency, IRDAI mandates a grace period of 30 calendar days.\n\n"
            "- Cover remains fully in force during grace period.\n\n"
            "> ℹ️ *Informational guidance subject to policy contract provisions.*"
        ),
        citations=[{"document_title": "Master Terms"}],
    )
    res = await quality_scorer_node(state)
    assert "quality_score" in res
    assert "structured_response" in res
    assert res["quality_score"]["overall_score"] >= 7.0
    assert res["structured_response"]["sub_intent"] == "renewal_servicing"
