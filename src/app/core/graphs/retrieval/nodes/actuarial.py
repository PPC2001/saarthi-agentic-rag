"""Deterministic actuarial calculation node."""

import re
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.core.graphs.retrieval.state import (
    GRACE_PERIOD_CALC_REGEX,
    LOAN_CALC_REGEX,
    SECTION_45_CALC_REGEX,
    SURRENDER_CALC_REGEX,
    AgenticRAGState,
)
from app.core.tools.actuarial_calculator import (
    calculate_grace_period_and_revival,
    calculate_policy_loan,
    calculate_surrender_value,
    check_section_45_incontestability,
)
from app.observability.logging import get_logger

logger = get_logger("actuarial_node")


def _extract_amount(text: str) -> float | None:
    """Extract financial amount from query supporting Lakh, Crore, and comma formatting."""
    # Matches: ₹5 Lakh, 2.5 Lacs, 10 Crore, Rs. 50,000, 100000
    lakh_match = re.search(r"(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:lakh|lac)s?", text, re.IGNORECASE)
    if lakh_match:
        return float(lakh_match.group(1)) * 100000.0

    crore_match = re.search(r"(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(?:crore|cr)s?", text, re.IGNORECASE)
    if crore_match:
        return float(crore_match.group(1)) * 10000000.0

    # Match explicit monetary values: ₹1,50,000 or Rs 50000
    currency_match = re.search(r"(?:₹|rs\.?|inr)\s*(\d[\d,]*(?:\.\d+)?)", text, re.IGNORECASE)
    if currency_match:
        return float(currency_match.group(1).replace(",", ""))

    # Generic numbers >= 1000 likely to be monetary amounts
    numbers = [float(n.replace(",", "")) for n in re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", text)]
    large_numbers = [n for n in numbers if n >= 1000]
    return large_numbers[0] if large_numbers else None


async def actuarial_calculator_node(
    state: AgenticRAGState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Execute deterministic actuarial and servicing calculations with strict parameter validation."""
    q = state.sanitized_query.lower()
    calc_result: dict[str, Any] = {}

    # 1. Policy Loan Calculation
    if LOAN_CALC_REGEX.search(q):
        extracted_amount = _extract_amount(q)
        years_match = re.search(r"(\d+)\s*(?:years?|yrs?)", q)
        years = int(years_match.group(1)) if years_match else 3

        if extracted_amount is None:
            calc_result = {
                "tool": "calculate_policy_loan",
                "requires_clarification": True,
                "ask_user": (
                    "Please provide your policy's current Special Surrender Value (SSV) or accumulated cash value "
                    "to calculate the exact maximum loan amount. (Under IRDAI norms, loans are typically capped at 80% of SSV)."
                ),
                "standard_rule": "Maximum 80% of Special Surrender Value (SSV) for traditional endowment/savings plans. Term plans are not eligible for policy loans.",
            }
        else:
            loan_res = calculate_policy_loan(
                surrender_value=extracted_amount,
                premiums_paid_years=years,
                product_line=state.product_code or "savings_endowment",
            )
            calc_result = {
                "tool": "calculate_policy_loan",
                "eligible": loan_res.eligible,
                "surrender_value": extracted_amount,
                "premiums_paid_years": years,
                "max_loan_amount": loan_res.max_loan_amount,
                "basis": loan_res.basis,
                "interest_terms": loan_res.interest_terms,
                "statutory_note": loan_res.statutory_note,
            }

    # 2. Section 45 Incontestability Check (Strict Validation — No Silent 2020 Default)
    elif SECTION_45_CALC_REGEX.search(q):
        date_matches = re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{2}-\d{2}-\d{4}\b", q)
        if not date_matches:
            calc_result = {
                "tool": "check_section_45_incontestability",
                "requires_clarification": True,
                "error": "MISSING_COMMENCEMENT_DATE",
                "ask_user": (
                    "To evaluate Section 45 Incontestability protection, please provide the policy commencement, "
                    "issuance, or latest revival date (in DD/MM/YYYY or YYYY-MM-DD format)."
                ),
                "statutory_advisory": (
                    "Section 45 of the Insurance Act 1938 states that no life insurance policy may be called in question "
                    "by an insurer on any ground (including misstatement/fraud) after the expiry of 3 (three) years from: "
                    "(1) date of issuance, (2) date of risk commencement, (3) date of revival, or (4) date of rider attachment, "
                    "whichever is latest."
                ),
            }
        else:
            start_date = date_matches[0]
            claim_date = date_matches[1] if len(date_matches) > 1 else None
            incontestability = check_section_45_incontestability(start_date, claim_date)
            calc_result = {
                "tool": "check_section_45_incontestability",
                "is_incontestable": incontestability.is_incontestable,
                "years_elapsed": incontestability.years_elapsed,
                "effective_start_date": incontestability.effective_start_date,
                "legal_status": incontestability.legal_status,
                "statutory_advisory": incontestability.statutory_advisory,
            }

    # 3. Grace Period & Revival Timeline (Strict Validation — No Silent 2024 Default)
    elif GRACE_PERIOD_CALC_REGEX.search(q):
        mode = "monthly" if "monthly" in q else "annual"
        date_matches = re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{2}-\d{2}-\d{4}\b", q)
        if not date_matches:
            calc_result = {
                "tool": "calculate_grace_period_and_revival",
                "requires_clarification": True,
                "error": "MISSING_FUP_DATE",
                "ask_user": (
                    "Please provide the First Unpaid Premium (FUP) date or due date (DD/MM/YYYY) to calculate "
                    "the exact grace period expiry date and 5-year revival deadline."
                ),
                "statutory_provisions": {
                    "grace_period": "15 days for monthly mode; 30 days (or 31 days) for quarterly/half-yearly/annual modes.",
                    "revival_window": "Up to 5 consecutive years from the First Unpaid Premium (FUP) date under IRDAI guidelines.",
                },
            }
        else:
            fup_date = date_matches[0]
            grace_res = calculate_grace_period_and_revival(fup_date, payment_frequency=mode)
            calc_result = {
                "tool": "calculate_grace_period_and_revival",
                "grace_period_days": grace_res.grace_period_days,
                "grace_period_expiry": grace_res.grace_period_expiry,
                "revival_deadline": grace_res.revival_deadline,
                "requirements": grace_res.requirements,
            }

    # 4. Surrender Value Calculation (Dynamic Product-Specific Slabs — Roadmap §6)
    elif SURRENDER_CALC_REGEX.search(q):
        years_match = re.search(r"(\d+)\s*(?:years?|yrs?)", q)
        years = int(years_match.group(1)) if years_match else 3
        p_line = state.product_code or "savings_endowment"
        extracted_paid = _extract_amount(q)

        if extracted_paid is not None:
            surrender_res = calculate_surrender_value(
                product_line=p_line,
                total_premiums_paid=extracted_paid,
                policy_year=years,
                product_code=state.product_code,
            )
            calc_result = {
                "tool": "calculate_surrender_value",
                "eligible": surrender_res.eligible,
                "product_line": surrender_res.product_line,
                "lock_in_active": surrender_res.lock_in_active,
                "total_premiums_paid": extracted_paid,
                "policy_year": years,
                "estimated_gsv_factor": surrender_res.estimated_gsv_factor,
                "estimated_surrender_amount": surrender_res.estimated_surrender_amount,
                "is_product_specific": surrender_res.is_product_specific,
                "applied_schedule": surrender_res.applied_schedule,
                "product_name": surrender_res.product_name,
                "surrender_basis": surrender_res.surrender_basis,
                "statutory_provisions": surrender_res.statutory_provisions,
            }
        else:
            surrender_res = calculate_surrender_value(
                product_line=p_line,
                total_premiums_paid=100000.0,
                policy_year=years,
                product_code=state.product_code,
            )
            calc_result = {
                "tool": "calculate_surrender_value",
                "eligible": surrender_res.eligible,
                "product_line": surrender_res.product_line,
                "lock_in_active": surrender_res.lock_in_active,
                "policy_year": years,
                "estimated_gsv_factor": surrender_res.estimated_gsv_factor,
                "estimated_surrender_amount": surrender_res.estimated_surrender_amount,
                "is_product_specific": surrender_res.is_product_specific,
                "applied_schedule": surrender_res.applied_schedule,
                "product_name": surrender_res.product_name,
                "illustrative_basis": "Illustrative based on ₹1,00,000 premium. Please specify your exact cumulative premiums paid for personalized calculation.",
                "statutory_provisions": surrender_res.statutory_provisions,
            }


    logger.info("ACTUARIAL_TOOL_EXECUTED", tool=calc_result.get("tool", "none"))
    return {"calculation_result": calc_result}
