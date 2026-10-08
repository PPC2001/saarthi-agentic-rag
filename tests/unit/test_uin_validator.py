"""Unit tests for IRDAI Life Insurance UIN extraction and validation."""

from app.core.document_processing.uin_validator import (
    extract_uins_from_text,
    validate_uin,
)


def test_validate_sbi_life_uin():
    """Verify SBI Life UIN is correctly identified and classified."""
    # SBI Life eShield Next (111N085V03)
    res = validate_uin("111N085V03")
    assert res is not None
    assert res.is_valid is True
    assert res.is_sbi_life is True
    assert res.insurer_name == "SBI Life Insurance Co. Ltd."
    assert "Non-Linked" in res.product_category
    assert res.version == "03"
    assert res.warning_notice is None


def test_validate_sbi_life_ulip_uin():
    """Verify SBI Life ULIP UIN classification."""
    # SBI Life Smart Wealth Builder (111L095V02)
    res = validate_uin("111L095V02")
    assert res is not None
    assert res.is_sbi_life is True
    assert "Unit Linked" in res.product_category
    assert res.version == "02"


def test_validate_competitor_uin_emits_warning():
    """Verify non-SBI Life policy document triggers appropriate warning notice."""
    # HDFC Life Click 2 Protect 3D Plus (101N101V05)
    res = validate_uin("101N101V05")
    assert res is not None
    assert res.is_valid is True
    assert res.is_sbi_life is False
    assert res.insurer_name == "HDFC Life Insurance Co. Ltd."
    assert res.warning_notice is not None
    assert "External Insurer Document" in res.warning_notice
    assert "HDFC Life Insurance Co. Ltd." in res.warning_notice


def test_validate_invalid_uin():
    """Verify malformed UINs return None."""
    assert validate_uin("INVALID_UIN") is None
    assert validate_uin("111N9999V01") is None  # Too many digits in seq
    assert validate_uin("") is None


def test_extract_uins_from_document_text():
    """Verify regex extraction of multiple UINs embedded in policy bond text."""
    sample_text = (
        "SBI LIFE INSURANCE COMPANY LIMITED\n"
        "Policy Schedule for SBI Life - Smart Platina Plus (UIN: 111N115V01).\n"
        "Optional Rider attached: SBI Life - Accidental Death Benefit Rider (UIN: 111R003V02).\n"
        "Customer previously held HDFC Life policy (UIN: 101N101V05) for reference."
    )
    results = extract_uins_from_text(sample_text)
    assert len(results) == 3

    sbi_product = next(r for r in results if r.uin == "111N115V01")
    assert sbi_product.is_sbi_life is True

    rider = next(r for r in results if r.uin == "111R003V02")
    assert rider.is_sbi_life is True
    assert "Rider" in rider.product_category

    competitor = next(r for r in results if r.uin == "101N101V05")
    assert competitor.is_sbi_life is False
    assert competitor.warning_notice is not None
