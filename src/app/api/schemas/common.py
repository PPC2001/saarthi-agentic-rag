"""Shared schemas and domain enums for the Life Insurance RAG application."""

from enum import StrEnum

from pydantic import BaseModel, Field


class ProductLine(StrEnum):
    """Indian Life insurance product categories."""

    TERM = "term"
    ULIP = "ulip"
    SAVINGS_ENDOWMENT = "savings_endowment"
    ANNUITY_PENSION = "annuity_pension"
    CHILD_SAVINGS = "child_savings"
    HEALTH_RIDER = "health_rider"
    WHOLE_LIFE = "whole_life"
    UNIVERSAL = "universal"
    VARIABLE = "variable"
    ANNUITY = "annuity"


class DocumentType(StrEnum):
    """Types of policy documents per IRDAI standards."""

    MASTER_POLICY = "master_policy"
    POLICY_SCHEDULE = "policy_schedule"
    RIDER = "rider"
    RIDER_DOCUMENT = "rider_document"
    ENDORSEMENT = "endorsement"
    SCHEDULE = "schedule"
    SALES_BROCHURE = "sales_brochure"
    BENEFIT_ILLUSTRATION = "benefit_illustration"
    IRDAI_CIRCULAR = "irdai_circular"
    AMENDMENT = "amendment"


class ContentType(StrEnum):
    """Content classification within a policy."""

    DEFINITION = "definition"
    BENEFIT = "benefit"
    EXCLUSION = "exclusion"
    SURRENDER_VALUE = "surrender_value"
    GRACE_PERIOD = "grace_period"
    REVIVAL_TERMS = "revival_terms"
    FREE_LOOK = "free_look"
    TAX_PROVISION = "tax_provision"
    CLAIM_PROCEDURE = "claim_procedure"
    NOMINATION_ASSIGNMENT = "nomination_assignment"
    PROVISION = "provision"
    PROCEDURE = "procedure"
    RIDER_TERM = "rider_term"


class ErrorDetail(BaseModel):
    """Standard error response detail."""

    code: str
    message: str
    field: str | None = None


class APIErrorResponse(BaseModel):
    """Standardized API error response."""

    error: str
    details: list[ErrorDetail] = Field(default_factory=list)
    request_id: str | None = None
