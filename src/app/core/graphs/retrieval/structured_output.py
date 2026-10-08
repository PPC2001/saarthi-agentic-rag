"""Structured Pydantic response models and parsers for life insurance synthesis.

Enforces structured outputs (Roadmap §18) enabling frontend UI component
binding, API contract stability, and compliance auditing.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field


class InsuranceStructuredResponse(BaseModel):
    """Structured response schema for Saarthi AI policy servicing answers."""

    summary: str = Field(
        default="",
        description="Executive summary or TL;DR addressing the core user question.",
    )
    key_provisions: list[str] = Field(
        default_factory=list,
        description="Key contractual terms, conditions, or rule bullet points.",
    )
    comparison_table: list[dict[str, str]] | None = Field(
        default=None,
        description="Structured rows for multi-product or conflict comparisons.",
    )
    regulatory_citations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Verified citations with document title, section, clause, and UIN.",
    )
    actuarial_result: dict[str, Any] | None = Field(
        default=None,
        description="Deterministic calculation figures if actuarial tool was executed.",
    )
    conflict_flags: list[str] = Field(
        default_factory=list,
        description="Identified discrepancies between customer uploads and master terms.",
    )
    statutory_notice: str = Field(
        default="",
        description="Statutory disclosure and legal binding hierarchy notice.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="high",
        description="Confidence level based on verified grounding chunk count.",
    )
    sub_intent: str | None = Field(
        default=None,
        description="Classified sub-intent (claim_enquiry, product_comparison, etc.).",
    )


def extract_table_from_markdown(text: str) -> list[dict[str, str]] | None:
    """Parse Markdown table into a list of row dictionaries."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    table_lines = [row for row in lines if row.startswith("|") and row.endswith("|")]
    if len(table_lines) < 2:
        return None

    headers = [h.strip() for h in table_lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []

    start_idx = 1
    if len(table_lines) >= 3 and re.match(r"^\|(\s*:?-+:?\s*\|)+$", table_lines[1]):
        start_idx = 2

    for row_line in table_lines[start_idx:]:
        if re.match(r"^\|(\s*:?-+:?\s*\|)+$", row_line):
            continue
        cols = [c.strip() for c in row_line.strip("|").split("|")]
        if len(cols) == len(headers):
            row_dict = {headers[i]: cols[i] for i in range(len(headers))}
            rows.append(row_dict)

    return rows if rows else None



def build_structured_response(
    query: str,
    raw_answer: str,
    citations: list[dict[str, Any]] | None = None,
    calculation_result: dict[str, Any] | None = None,
    retrieval_quality: str = "adequate",
    sub_intent: str | None = None,
) -> InsuranceStructuredResponse:
    """Transform markdown synthesis text into structured InsuranceStructuredResponse."""
    text = raw_answer.strip()
    citations = citations or []

    # 1. Extract Summary (first non-heading paragraph)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    summary = ""
    for p in paragraphs:
        if not p.startswith("#") and not p.startswith(">") and not p.startswith("|"):
            summary = p
            break
    if not summary and paragraphs:
        summary = paragraphs[0]

    # 2. Extract Key Provisions (bullet points)
    key_provisions: list[str] = []
    for line in text.splitlines():
        trimmed = line.strip()
        if (trimmed.startswith("- ") or trimmed.startswith("* ")) and len(trimmed) > 5:
            clean_bullet = trimmed[2:].strip().lstrip("*").rstrip("*")
            key_provisions.append(clean_bullet)

    # 3. Extract Comparison Table if present
    comparison_table = extract_table_from_markdown(text)

    # 4. Extract Statutory Notice (blockquote or statutory line)
    statutory_notice = ""
    for line in text.splitlines():
        trimmed = line.strip()
        if trimmed.startswith(">") or "Insurance Act 1938" in trimmed or "IRDAI" in trimmed:
            statutory_notice = trimmed.lstrip("> *").rstrip("*").strip()
            if statutory_notice:
                break
    if not statutory_notice:
        statutory_notice = (
            "This information is strictly for servicing guidance. "
            "The official policy bond and IRDAI regulations supersede all summaries."
        )

    # 5. Extract Conflict Flags
    conflict_flags: list[str] = []
    if "CONFLICT" in text.upper() or "DISCREPANCY" in text.upper():
        for line in text.splitlines():
            if any(w in line.lower() for w in ("conflict", "supersede", "discrepancy")):
                conflict_flags.append(line.strip())

    # 6. Determine Confidence
    if retrieval_quality == "insufficient":
        confidence: Literal["high", "medium", "low"] = "low"
    elif retrieval_quality == "thin" or (len(citations) < 2 and not calculation_result):
        confidence = "medium"
    else:
        confidence = "high"

    # 7. Format clean citations
    cleaned_citations = []
    for c in citations[:5]:
        cleaned_citations.append(
            {
                "document_title": c.get("document_title", "Policy Document"),
                "section": c.get("section", "General"),
                "clause_number": c.get("clause_number", ""),
                "uin": c.get("uin"),
                "source_type": c.get("source_type", "contract"),
                "snippet": c.get("snippet", "")[:200],
            }
        )

    return InsuranceStructuredResponse(
        summary=summary,
        key_provisions=key_provisions[:8],
        comparison_table=comparison_table,
        regulatory_citations=cleaned_citations,
        actuarial_result=calculation_result,
        conflict_flags=conflict_flags,
        statutory_notice=statutory_notice,
        confidence=confidence,
        sub_intent=sub_intent,
    )
