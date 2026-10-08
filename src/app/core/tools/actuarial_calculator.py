"""Deterministic actuarial and policy servicing calculations for Indian Life Insurance.

Eliminates LLM math hallucinations for:
- Policy loan eligibility and maximum loan value (80% of SSV)
- Surrender value eligibility (GSV vs SSV, ULIP 5-year lock-in)
- Section 45 Incontestability 3-year statutory timeline check
- Grace period (15 days monthly, 30 days other) and 5-year revival expiration
"""

from datetime import date, datetime, timedelta

import logfire
from pydantic import BaseModel

from app.observability.logging import get_logger

logger = get_logger("actuarial_calculator")


class PolicyLoanCalculation(BaseModel):
    """Result of policy loan calculation under IRDAI guidelines."""

    eligible: bool
    max_loan_amount: float
    basis: str
    interest_terms: str
    repayment_rules: str
    statutory_note: str


class SurrenderCalculation(BaseModel):
    """Result of surrender value evaluation."""

    eligible: bool
    product_line: str
    lock_in_active: bool
    lock_in_remaining_years: int
    surrender_basis: str
    estimated_gsv_factor: float
    statutory_provisions: str
    is_product_specific: bool = False
    applied_schedule: str = "IRDAI Regulatory Minimum GSV"
    product_name: str | None = None
    estimated_surrender_amount: float = 0.0


class IncontestabilityCheck(BaseModel):
    """Result of Section 45 Insurance Act 1938 non-repudiation assessment."""

    is_incontestable: bool
    years_elapsed: float
    effective_start_date: str
    claim_assessment_date: str
    legal_status: str
    statutory_advisory: str


class GraceAndRevivalTimeline(BaseModel):
    """Timeline calculation for grace period and revival eligibility."""

    grace_period_days: int
    grace_period_expiry: str
    is_in_grace_period: bool
    revival_deadline: str
    revival_eligible: bool
    requirements: list[str]


class MaturityBenefitCalculation(BaseModel):
    """Result of deterministic policy maturity benefit projection."""

    sum_assured: float
    guaranteed_additions: float
    accrued_bonuses: float
    terminal_bonus: float
    total_maturity_payout: float
    tax_status_10_10d: str
    statutory_advisory: str


class FreeLookRefundCalculation(BaseModel):
    """Result of statutory free-look cancellation refund evaluation."""

    is_eligible: bool
    free_look_window_days: int
    days_elapsed: int
    gross_premium_paid: float
    deductions: dict[str, float]
    net_refund_amount: float
    statutory_provisions: str


def parse_flexible_date(date_str: str) -> date:
    """Parse various common date formats (YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY)."""
    clean_str = date_str.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(clean_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(
        f"Unable to parse date string '{date_str}'. Expected YYYY-MM-DD or DD/MM/YYYY."
    )


def calculate_policy_loan(
    surrender_value: float,
    premiums_paid_years: int,
    product_line: str = "savings_endowment",
) -> PolicyLoanCalculation:
    """Calculate maximum policy loan available under IRDAI life insurance norms.

    Args:
        surrender_value: Special Surrender Value (SSV) or cash value in INR.
        premiums_paid_years: Number of full consecutive policy years with paid premiums.
        product_line: Type of insurance plan ('savings_endowment', 'term', 'ulip').
    """
    with logfire.span(
        "actuarial.calculate_policy_loan",
        surrender_value=surrender_value,
        premiums_paid_years=premiums_paid_years,
        product_line=product_line,
    ):
        if product_line.lower() == "term":
            logger.info("POLICY_LOAN_NOT_ELIGIBLE_TERM")
            return PolicyLoanCalculation(
                eligible=False,
                max_loan_amount=0.0,
                basis="Term assurance plans carry no cash value or loan facility.",
                interest_terms="N/A",
                repayment_rules="N/A",
                statutory_note="Pure risk protection plans have zero surrender or loan eligibility under IRDAI guidelines.",
            )

        if product_line.lower() == "ulip":
            logger.info("POLICY_LOAN_NOT_ELIGIBLE_ULIP")
            return PolicyLoanCalculation(
                eligible=False,
                max_loan_amount=0.0,
                basis="ULIP guidelines prohibit policy loans against unit funds.",
                interest_terms="N/A",
                repayment_rules="Partial withdrawals permitted after 5-year lock-in in lieu of loans.",
                statutory_note="As per IRDAI (Unit Linked Insurance Products) Regulations, loans are not permitted on linked plans.",
            )

        if premiums_paid_years < 2:
            logger.info(
                "POLICY_LOAN_NOT_ELIGIBLE_MIN_YEARS",
                years_paid=premiums_paid_years,
            )
            return PolicyLoanCalculation(
                eligible=False,
                max_loan_amount=0.0,
                basis=f"Premiums paid for only {premiums_paid_years} year(s). Minimum 2 full years required.",
                interest_terms="Interest declared half-yearly benchmarked to 10-year G-Sec yield.",
                repayment_rules="Loan available only after policy acquires guaranteed surrender value.",
                statutory_note="Under IRDAI Master Circular on Life Products, loan facility requires acquisition of surrender value.",
            )

        max_loan = round(surrender_value * 0.80, 2)
        logger.info("POLICY_LOAN_CALCULATED_SUCCESS", eligible=True, max_loan_amount=max_loan)
        return PolicyLoanCalculation(
            eligible=True,
            max_loan_amount=max_loan,
            basis="Up to 80% of Special Surrender Value (SSV).",
            interest_terms="Interest compounded half-yearly at prevailing board-approved rate (typically 9.0% - 9.5% p.a.).",
            repayment_rules="Interest payable half-yearly. Principal can be repaid anytime before maturity or deducted from final claim proceeds.",
            statutory_note="If total outstanding loan plus interest exceeds available surrender value on a paid-up policy, contract may be foreclosed after prior statutory notice.",
        )


# Product-Specific Guaranteed Surrender Value (GSV) / Special Surrender Value (SSV) Slabs
# Sourced from SBI Life Master Product Contracts & Technical Notes (Roadmap §6)
SBI_LIFE_PRODUCT_GSV_SCHEDULES: dict[str, dict[int, float]] = {
    "SMART_PLATINA_PLUS": {
        2: 0.30,
        3: 0.35,
        4: 0.50,
        5: 0.54,
        6: 0.58,
        7: 0.65,
        8: 0.70,
        9: 0.75,
        10: 0.80,
    },
    "SARAL_SWADHAN": {
        2: 0.00,  # Requires min 3 years for TROP surrender
        3: 0.30,
        4: 0.50,
        5: 0.50,
        6: 0.50,
        7: 0.50,
        8: 0.55,
        9: 0.60,
        10: 0.65,
    },
    "SHUBH_NIVESH": {
        2: 0.30,
        3: 0.35,
        4: 0.50,
        5: 0.55,
        6: 0.60,
        7: 0.65,
        8: 0.70,
        9: 0.75,
        10: 0.85,
    },
    "SMART_CHAMP_INSURANCE": {
        2: 0.30,
        3: 0.35,
        4: 0.50,
        5: 0.55,
        6: 0.60,
        7: 0.70,
        8: 0.75,
        9: 0.80,
        10: 0.90,
    },
}


def calculate_surrender_value(
    product_line: str,
    total_premiums_paid: float,
    policy_year: int,
    lock_in_years: int = 5,
    product_code: str | None = None,
    custom_gsv_schedule: dict[int, float] | None = None,
) -> SurrenderCalculation:
    """Evaluate surrender value eligibility, product-specific slabs, and statutory lock-in conditions.

    Args:
        product_line: 'term', 'ulip', or 'savings_endowment'.
        total_premiums_paid: Cumulative premiums paid (excluding taxes).
        policy_year: Elapsed policy duration in full completed years.
        lock_in_years: Lock-in threshold (mandatory 5 years for ULIPs).
        product_code: Optional SBI Life product code/identifier (e.g., 'SMART_PLATINA_PLUS').
        custom_gsv_schedule: Optional custom surrender factor schedule mapping {policy_year: factor}.
    """
    with logfire.span(
        "actuarial.calculate_surrender_value",
        product_line=product_line,
        total_premiums_paid=total_premiums_paid,
        policy_year=policy_year,
        product_code=product_code,
    ):
        p_line = product_line.lower()
        if p_line == "term":
            logger.info("SURRENDER_VALUE_NOT_ELIGIBLE_TERM")
            return SurrenderCalculation(
                eligible=False,
                product_line="Term Assurance",
                lock_in_active=False,
                lock_in_remaining_years=0,
                surrender_basis="Zero surrender value on regular-pay term insurance.",
                estimated_gsv_factor=0.0,
                statutory_provisions="Pure risk term plans do not acquire surrender values unless issued under specific Return of Premium (TROP) terms.",
            )

        if "ulip" in p_line:
            is_locked = policy_year < lock_in_years
            remaining = max(0, lock_in_years - policy_year)
            if is_locked:
                logger.info(
                    "SURRENDER_VALUE_ULIP_LOCKIN_ACTIVE", remaining_years=remaining
                )
                return SurrenderCalculation(
                    eligible=False,
                    product_line="Unit Linked Plan (ULIP)",
                    lock_in_active=True,
                    lock_in_remaining_years=remaining,
                    surrender_basis="Statutory 5-year lock-in active. Proceeds routed to Discontinued Policy Fund (DPF).",
                    estimated_gsv_factor=0.0,
                    statutory_provisions=(
                        f"Under IRDAI ULIP Regulations, policy proceeds cannot be disbursed before {lock_in_years} years. "
                        f"Fund value less discontinuation charges is credited to DPF earning minimum 4% p.a. interest, "
                        f"payable upon completion of year 5."
                    ),
                )
            logger.info("SURRENDER_VALUE_ULIP_LOCKIN_COMPLETED")
            return SurrenderCalculation(
                eligible=True,
                product_line="Unit Linked Plan (ULIP)",
                lock_in_active=False,
                lock_in_remaining_years=0,
                surrender_basis="Full Fund Value payable based on prevailing NAV without discontinuation charges.",
                estimated_gsv_factor=1.0,
                statutory_provisions="5-year lock-in successfully completed. Surrender value equals units multiplied by current NAV.",
            )

        # Traditional savings / endowment
        if policy_year < 2:
            logger.info(
                "SURRENDER_VALUE_TRADITIONAL_MIN_YEARS",
                policy_year=policy_year,
            )
            return SurrenderCalculation(
                eligible=False,
                product_line="Savings / Endowment Plan",
                lock_in_active=True,
                lock_in_remaining_years=2 - policy_year,
                surrender_basis="No surrender value payable before completion of 2 full policy years.",
                estimated_gsv_factor=0.0,
                statutory_provisions="Traditional policies acquire Guaranteed Surrender Value (GSV) only after payment of 2 full years' premiums.",
            )

        # Tiered GSV percentage based on policy year (IRDAI Regulatory Minimums)
        if policy_year == 2:
            gsv_factor = 0.30
        elif policy_year == 3:
            gsv_factor = 0.35
        elif 4 <= policy_year <= 7:
            gsv_factor = 0.50
        else:
            gsv_factor = min(0.90, 0.50 + ((policy_year - 7) * 0.05))

        # Check product-specific schedule match (Roadmap §6)
        matched_schedule: dict[int, float] | None = custom_gsv_schedule
        matched_product_name: str | None = None

        if not matched_schedule and product_code:
            norm_code = product_code.upper().replace("-", "_").replace(" ", "_")
            for prod_key, sched in SBI_LIFE_PRODUCT_GSV_SCHEDULES.items():
                if prod_key in norm_code or norm_code in prod_key:
                    matched_schedule = sched
                    matched_product_name = prod_key.replace("_", " ").title()
                    break

        product_specific_factor: float | None = None
        if matched_schedule:
            if policy_year in matched_schedule:
                product_specific_factor = matched_schedule[policy_year]
            elif policy_year > max(matched_schedule.keys()):
                product_specific_factor = matched_schedule[max(matched_schedule.keys())]

        is_product_specific = False
        applied_schedule = "IRDAI Regulatory Minimum GSV"

        # IRDAI mandates that the policyholder receives the HIGHER of Guaranteed Surrender Value (GSV) or Special Surrender Value (SSV)
        if product_specific_factor is not None and product_specific_factor >= gsv_factor:
            gsv_factor = product_specific_factor
            is_product_specific = True
            applied_schedule = f"SBI Life {matched_product_name or product_code} Contract Schedule"

        estimated_amount = round(total_premiums_paid * gsv_factor, 2)

        logger.info(
            "SURRENDER_VALUE_EVALUATED",
            eligible=True,
            gsv_factor=gsv_factor,
            is_product_specific=is_product_specific,
            applied_schedule=applied_schedule,
            policy_year=policy_year,
            estimated_amount=estimated_amount,
        )
        return SurrenderCalculation(
            eligible=True,
            product_line="Savings / Endowment Plan",
            lock_in_active=False,
            lock_in_remaining_years=0,
            surrender_basis=(
                f"Higher of GSV/SSV (~{int(gsv_factor * 100)}% of total premiums paid = ₹{estimated_amount:,.2f}) "
                f"derived from {applied_schedule}."
            ),
            estimated_gsv_factor=gsv_factor,
            statutory_provisions="Surrender value payable is the higher of Guaranteed Surrender Value (GSV) or Special Surrender Value (SSV) per IRDAI Master Circulars.",
            is_product_specific=is_product_specific,
            applied_schedule=applied_schedule,
            product_name=matched_product_name,
            estimated_surrender_amount=estimated_amount,
        )


def check_section_45_incontestability(
    commencement_date_str: str,
    claim_date_str: str | None = None,
) -> IncontestabilityCheck:
    """Assess whether the 3-year statutory incontestability bar under Section 45 applies.

    Section 45 of Insurance Act 1938 mandates that no life insurance policy may be
    called in question on any ground whatsoever after 3 years from commencement, issuance, or revival.
    """
    with logfire.span(
        "actuarial.check_section_45_incontestability",
        commencement_date_str=commencement_date_str,
        claim_date_str=claim_date_str,
    ):
        start_date = parse_flexible_date(commencement_date_str)
        eval_date = parse_flexible_date(claim_date_str) if claim_date_str else date.today()

        days_diff = (eval_date - start_date).days
        years_elapsed = round(days_diff / 365.25, 2)

        if years_elapsed >= 3.0:
            logger.info(
                "SECTION_45_INCONTESTABLE",
                years_elapsed=years_elapsed,
                commencement_date=start_date.isoformat(),
            )
            return IncontestabilityCheck(
                is_incontestable=True,
                years_elapsed=years_elapsed,
                effective_start_date=start_date.isoformat(),
                claim_assessment_date=eval_date.isoformat(),
                legal_status="PROTECTED: Incontestable under Section 45",
                statutory_advisory=(
                    f"Policy has completed {years_elapsed} years (>= 3 years) since inception/revival. "
                    "Under Section 45 of the Insurance Act 1938 as amended, no policy of life insurance "
                    "shall be called in question by an insurer on any ground whatsoever, including misstatement, "
                    "non-disclosure, or fraud. The claim must be settled strictly as per policy terms."
                ),
            )

        remaining_years = round(3.0 - years_elapsed, 2)
        logger.info(
            "SECTION_45_CONTESTABLE_WINDOW",
            years_elapsed=years_elapsed,
            remaining_years=remaining_years,
        )
        return IncontestabilityCheck(
            is_incontestable=False,
            years_elapsed=years_elapsed,
            effective_start_date=start_date.isoformat(),
            claim_assessment_date=eval_date.isoformat(),
            legal_status="CONTESTABLE: Within 3-year statutory contestability window",
            statutory_advisory=(
                f"Policy has completed {years_elapsed} years (< 3 years; {remaining_years} years remaining). "
                "Under Section 45, an insurer may investigate early claims. However, repudiation requires proof "
                "that: (1) material facts were suppressed, (2) the suppression was fraudulent, and (3) the proposer "
                "knew the suppressed facts were material at the time of proposal."
            ),
        )


def calculate_grace_period_and_revival(
    fup_date_str: str,
    payment_frequency: str = "monthly",
    assessment_date_str: str | None = None,
) -> GraceAndRevivalTimeline:
    """Calculate grace period expiry and 5-year revival window from First Unpaid Premium (FUP).

    IRDAI rules:
    - Grace period: 15 calendar days for monthly; 30 calendar days for quarterly/half-yearly/yearly.
    - Revival window: 5 consecutive years from Date of First Unpaid Premium (FUP).
    """
    with logfire.span(
        "actuarial.calculate_grace_period_and_revival",
        fup_date_str=fup_date_str,
        payment_frequency=payment_frequency,
    ):
        fup_date = parse_flexible_date(fup_date_str)
        eval_date = (
            parse_flexible_date(assessment_date_str) if assessment_date_str else date.today()
        )

        freq = payment_frequency.strip().lower()
        grace_days = 15 if freq == "monthly" else 30

        grace_expiry = fup_date + timedelta(days=grace_days)
        revival_deadline = date(fup_date.year + 5, fup_date.month, fup_date.day)

        in_grace = fup_date <= eval_date <= grace_expiry
        revival_valid = eval_date <= revival_deadline

        reqs = [
            f"Grace period duration: {grace_days} calendar days ({freq} premium mode).",
            "During grace period, full life risk cover remains active without penalty.",
            f"Grace period ends on: {grace_expiry.isoformat()}.",
            f"Lapsed policy revival deadline: {revival_deadline.isoformat()} (5 consecutive years from FUP).",
            "Revival requirements: Good Health Declaration (DGH), medical reports (if sum at risk exceeds non-medical limit), and overdue premiums with interest.",
        ]

        logger.info(
            "GRACE_AND_REVIVAL_CALCULATED",
            in_grace=in_grace,
            revival_valid=revival_valid,
            grace_days=grace_days,
            grace_expiry=grace_expiry.isoformat(),
        )

        return GraceAndRevivalTimeline(
            grace_period_days=grace_days,
            grace_period_expiry=grace_expiry.isoformat(),
            is_in_grace_period=in_grace,
            revival_deadline=revival_deadline.isoformat(),
            revival_eligible=revival_valid,
            requirements=reqs,
        )


def calculate_maturity_benefit(
    sum_assured: float,
    guaranteed_additions: float = 0.0,
    accrued_bonuses: float = 0.0,
    terminal_bonus: float = 0.0,
    annual_premium: float = 0.0,
) -> MaturityBenefitCalculation:
    """Project maturity payout under traditional endowment / guaranteed savings plans.

    IRDAI & IT Act guidelines:
    - Total maturity benefit = Sum Assured on Maturity + Guaranteed Additions + Accrued Reversionary Bonuses + Terminal Bonus.
    - Tax exemption under Section 10(10D) applies if annual premium <= 10% of sum assured (for post-April 2012 policies)
      and total aggregate annual premium across policies <= ₹5,00,000 (post-Finance Act 2023).
    """
    total = sum_assured + guaranteed_additions + accrued_bonuses + terminal_bonus

    if annual_premium > 0 and (annual_premium > 0.10 * sum_assured or annual_premium > 500000.0):
        tax_status = (
            "TAXABLE: Annual premium exceeds 10% of sum assured or ₹5,00,000 threshold under Finance Act 2023. "
            "Income is taxable under Section 56(2)(xiii)."
        )
    else:
        tax_status = "EXEMPT: Fully exempt from income tax under Section 10(10D) of the Income Tax Act, 1961."

    return MaturityBenefitCalculation(
        sum_assured=sum_assured,
        guaranteed_additions=guaranteed_additions,
        accrued_bonuses=accrued_bonuses,
        terminal_bonus=terminal_bonus,
        total_maturity_payout=round(total, 2),
        tax_status_10_10d=tax_status,
        statutory_advisory="Maturity benefits are payable provided the policy is fully paid-up and in-force at the end of the policy term.",
    )


def calculate_free_look_refund(
    premium_paid: float,
    mode_of_issuance: str = "electronic",
    days_elapsed: int = 5,
    risk_premium_proportion: float = 0.0,
    medical_expenses: float = 0.0,
    stamp_duty: float = 0.0,
) -> FreeLookRefundCalculation:
    """Calculate free-look cancellation refund pursuant to IRDAI (Protection of Policyholders' Interests) Regulations.

    Statutory rules:
    - 30 days for electronic issuance or distance marketing; 15 days for physical delivery.
    - Insurer deducts: proportionate risk premium for covered period, medical examination expenses, and stamp duty charges.
    """
    is_electronic = mode_of_issuance.lower() in ("electronic", "digital", "distance_marketing", "online")
    window_days = 30 if is_electronic else 15
    is_eligible = days_elapsed <= window_days

    total_deductions = risk_premium_proportion + medical_expenses + stamp_duty
    net_refund = max(0.0, premium_paid - total_deductions) if is_eligible else 0.0

    deductions_breakdown = {
        "proportionate_risk_premium": risk_premium_proportion,
        "medical_examinations": medical_expenses,
        "stamp_duty_charges": stamp_duty,
        "total_deductions": total_deductions,
    }

    provisions = (
        f"Free-look window is {window_days} days for {mode_of_issuance} delivery. "
        f"Days elapsed: {days_elapsed}. "
        + ("Eligible for cancellation refund." if is_eligible else f"EXPIRED: Cancellation request exceeded {window_days} days statutory deadline.")
    )

    return FreeLookRefundCalculation(
        is_eligible=is_eligible,
        free_look_window_days=window_days,
        days_elapsed=days_elapsed,
        gross_premium_paid=premium_paid,
        deductions=deductions_breakdown,
        net_refund_amount=round(net_refund, 2),
        statutory_provisions=provisions,
    )
