"""UIN (Unique Identification Number) extraction and validation for IRDAI life insurance policies.

IRDAI mandates a standardized UIN structure for all life insurance products in India:
Format: [3-digit Insurer Code][1-letter Category][3-digit Seq]V[2-digit Version]
Example: 111N085V03 (SBI Life - eShield Next)

Insurer Code 111 belongs to SBI Life Insurance Company Limited.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from app.observability.logging import get_logger

logger = get_logger("uin_validator")

# Precompiled regex for standard IRDAI Life Insurance UIN format
IRDAI_UIN_PATTERN = re.compile(
    r"\b([0-9]{3})([NLPAQRHGnlpaqrhg])([0-9]{3})[Vv]([0-9]{2})\b"
)

# Known Life Insurance Insurer Codes registered with IRDAI
REGISTERED_INSURERS: dict[str, str] = {
    "111": "SBI Life Insurance Co. Ltd.",
    "101": "HDFC Life Insurance Co. Ltd.",
    "104": "ICICI Prudential Life Insurance Co. Ltd.",
    "105": "Max Life Insurance Co. Ltd.",
    "110": "Tata AIA Life Insurance Co. Ltd.",
    "116": "Kotak Mahindra Life Insurance Co. Ltd.",
    "121": "Aditya Birla Sun Life Insurance Co. Ltd.",
    "128": "Bajaj Allianz Life Insurance Co. Ltd.",
    "133": "PNB MetLife India Insurance Co. Ltd.",
    "512": "Life Insurance Corporation of India (LIC)",
}

# Product Category classifications per IRDAI Product Regulations
PRODUCT_CATEGORIES: dict[str, str] = {
    "N": "Non-Linked (Traditional / Term / Endowment / Savings)",
    "L": "Unit Linked Insurance Plan (ULIP)",
    "P": "Pension (Traditional Non-Linked)",
    "Q": "Pension Linked (ULIP Pension)",
    "H": "Health Insurance",
    "A": "Immediate / Deferred Annuity",
    "R": "Optional Rider Benefit",
    "G": "Group Insurance Scheme",
}


class UINValidationResult(BaseModel):
    """Validation and metadata extraction result for an IRDAI product UIN."""

    uin: str
    is_valid: bool = True
    insurer_code: str
    insurer_name: str
    is_sbi_life: bool
    product_category_code: str
    product_category: str
    sequence_number: str
    version: str
    warning_notice: str | None = None


def validate_uin(uin_str: str) -> UINValidationResult | None:
    """Validate and parse a single IRDAI life insurance UIN string."""
    clean_uin = uin_str.strip().upper()
    match = IRDAI_UIN_PATTERN.fullmatch(clean_uin)
    if not match:
        return None

    insurer_code, cat_code, seq_no, ver = match.groups()
    cat_code = cat_code.upper()
    insurer_name = REGISTERED_INSURERS.get(insurer_code, f"Insurer Code {insurer_code} (Other IRDAI Registered Insurer)")
    is_sbi_life = (insurer_code == "111")
    product_cat = PRODUCT_CATEGORIES.get(cat_code, "Life Insurance Plan")

    warning: str | None = None
    if not is_sbi_life:
        warning = (
            f"⚠️ External Insurer Document: Uploaded policy document belongs to {insurer_name} (UIN: {clean_uin}). "
            f"Saarthi AI knowledge base specializes in SBI Life products and statutory IRDAI guidelines. "
            f"Specific charges, terms, and conditions may differ from SBI Life standard policies."
        )

    return UINValidationResult(
        uin=clean_uin,
        is_valid=True,
        insurer_code=insurer_code,
        insurer_name=insurer_name,
        is_sbi_life=is_sbi_life,
        product_category_code=cat_code,
        product_category=product_cat,
        sequence_number=seq_no,
        version=ver,
        warning_notice=warning,
    )


def extract_uins_from_text(text: str) -> list[UINValidationResult]:
    """Scan document text to extract and validate all IRDAI product UINs."""
    results: list[UINValidationResult] = []
    seen: set[str] = set()

    for match in IRDAI_UIN_PATTERN.finditer(text):
        full_uin = match.group(0).upper()
        if full_uin in seen:
            continue
        seen.add(full_uin)
        val = validate_uin(full_uin)
        if val:
            results.append(val)

    if results:
        logger.info(
            "UINS_EXTRACTED_FROM_DOCUMENT",
            count=len(results),
            uins=[r.uin for r in results],
            has_sbi_life=any(r.is_sbi_life for r in results),
        )
    return results
