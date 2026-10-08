"""Unit tests for deterministic actuarial and policy servicing calculation tools."""

from app.core.tools.actuarial_calculator import (
    calculate_free_look_refund,
    calculate_grace_period_and_revival,
    calculate_maturity_benefit,
    calculate_policy_loan,
    calculate_surrender_value,
    check_section_45_incontestability,
)


def test_calculate_policy_loan_eligible():
    """Verify policy loan calculation grants up to 80% of SSV when >= 2 years paid."""
    res = calculate_policy_loan(surrender_value=500000.0, premiums_paid_years=3)
    assert res.eligible is True
    assert res.max_loan_amount == 400000.0
    assert "80%" in res.basis
    assert "half-yearly" in res.interest_terms


def test_calculate_policy_loan_ineligible_under_two_years():
    """Verify policy loan is rejected if premiums paid for < 2 years."""
    res = calculate_policy_loan(surrender_value=100000.0, premiums_paid_years=1)
    assert res.eligible is False
    assert res.max_loan_amount == 0.0
    assert "Minimum 2 full years required" in res.basis


def test_calculate_policy_loan_term_plan():
    """Verify term assurance plans have zero loan eligibility."""
    res = calculate_policy_loan(surrender_value=0.0, premiums_paid_years=5, product_line="term")
    assert res.eligible is False
    assert "Term assurance" in res.basis


def test_calculate_surrender_value_ulip_lockin():
    """Verify ULIP statutory 5-year lock-in is enforced."""
    res = calculate_surrender_value(
        product_line="ulip", total_premiums_paid=200000.0, policy_year=3
    )
    assert res.eligible is False
    assert res.lock_in_active is True
    assert res.lock_in_remaining_years == 2
    assert "DPF" in res.surrender_basis or "Discontinued Policy Fund" in res.statutory_provisions


def test_calculate_surrender_value_ulip_post_lockin():
    """Verify ULIP surrender value is available after 5 years."""
    res = calculate_surrender_value(
        product_line="ulip", total_premiums_paid=500000.0, policy_year=6
    )
    assert res.eligible is True
    assert res.lock_in_active is False
    assert "NAV" in res.surrender_basis


def test_section_45_incontestability_protected():
    """Verify Section 45 statutory bar after 3 years."""
    res = check_section_45_incontestability(
        commencement_date_str="2020-01-01", claim_date_str="2024-05-01"
    )
    assert res.is_incontestable is True
    assert res.years_elapsed >= 3.0
    assert "PROTECTED" in res.legal_status
    assert "Section 45" in res.statutory_advisory


def test_section_45_incontestability_contestable():
    """Verify Section 45 within 3-year early claim window."""
    res = check_section_45_incontestability(
        commencement_date_str="2023-06-01", claim_date_str="2024-01-01"
    )
    assert res.is_incontestable is False
    assert res.years_elapsed < 3.0
    assert "CONTESTABLE" in res.legal_status


def test_calculate_grace_period_and_revival():
    """Verify grace period (15 days monthly, 30 days annual) and 5-year revival deadline."""
    monthly_res = calculate_grace_period_and_revival(
        fup_date_str="2024-01-01",
        payment_frequency="monthly",
        assessment_date_str="2024-01-10",
    )
    assert monthly_res.grace_period_days == 15
    assert monthly_res.is_in_grace_period is True
    assert monthly_res.revival_eligible is True
    assert "2029-01-01" in monthly_res.revival_deadline

    annual_res = calculate_grace_period_and_revival(
        fup_date_str="2024-01-01",
        payment_frequency="annual",
        assessment_date_str="2024-01-20",
    )
    assert annual_res.grace_period_days == 30
    assert annual_res.is_in_grace_period is True


def test_calculate_maturity_benefit():
    """Verify maturity benefit projection and Section 10(10D) tax status check."""
    res = calculate_maturity_benefit(
        sum_assured=500000.0,
        guaranteed_additions=150000.0,
        accrued_bonuses=50000.0,
        terminal_bonus=25000.0,
        annual_premium=50000.0,
    )
    assert res.total_maturity_payout == 725000.0
    assert "EXEMPT" in res.tax_status_10_10d


def test_calculate_free_look_refund_electronic():
    """Verify electronic issuance grants 30-day free look window and computes net deductions."""
    res = calculate_free_look_refund(
        premium_paid=50000.0,
        mode_of_issuance="electronic",
        days_elapsed=10,
        risk_premium_proportion=500.0,
        stamp_duty=100.0,
    )
    assert res.is_eligible is True
    assert res.free_look_window_days == 30
    assert res.net_refund_amount == 49400.0


def test_calculate_free_look_refund_physical_expired():
    """Verify physical issuance enforces 15-day window and rejects expired requests."""
    res = calculate_free_look_refund(
        premium_paid=50000.0,
        mode_of_issuance="physical",
        days_elapsed=20,
    )
    assert res.is_eligible is False
    assert res.free_look_window_days == 15
    assert res.net_refund_amount == 0.0


def test_calculate_surrender_value_product_specific():
    """Verify SBI Life Smart Platina Plus product-specific GSV schedule applies."""
    res = calculate_surrender_value(
        product_line="savings",
        total_premiums_paid=300000.0,
        policy_year=6,
        product_code="SMART_PLATINA_PLUS",
    )
    assert res.eligible is True
    assert res.is_product_specific is True
    assert res.estimated_gsv_factor == 0.58  # Smart Platina Plus year 6 factor (vs 0.50 reg min)
    assert res.estimated_surrender_amount == 174000.0  # 300,000 * 0.58
    assert "Smart Platina Plus" in res.applied_schedule

