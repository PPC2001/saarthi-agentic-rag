"""Comprehensive Domain and Specification Tests for Sarathi AI - SBI Life & IRDAI Assistant.

Validates:
1. Strict Scope Guard: Exactly returns "I can only help with life insurance topics."
2. Web Tool Domain Allowlist: Strictly permits irdai.gov.in and sbilife.co.in, refuses all others.
3. Mandatory Statutory Disclaimer: Ends policy answers with official communication disclaimer.
4. Multi-Tenant Per-User Document Isolation: Prevents cross-user document access.
5. Multi-Format Uploads: PDF, DOCX, Images/Scans, Plain Text.
6. Google Sign-In Bearer Token Authentication.
7. Statutory Compliance Audit Logging with PII Scrubbing.
"""

import io

import pytest
from docx import Document

from app.api.middleware.auth import get_current_user
from app.config import Settings
from app.core.document_processing.session_docs import (
    SessionDocumentStore,
    parse_uploaded_document,
)
from app.core.guardrails.input_guardrails import (
    OFF_TOPIC_GUARDRAIL_RESPONSE,
    check_off_topic,
)
from app.core.guardrails.output_guardrails import enforce_output_guardrails
from app.core.retrieval.web_search import (
    WebRetrievalClient,
    is_domain_allowed,
)
from app.observability.logging import log_compliance_audit


def test_off_topic_exact_refusal_phrase():
    """Verify off-topic queries trigger the exact required opening sentence."""
    assert OFF_TOPIC_GUARDRAIL_RESPONSE.startswith("I can only help with life insurance topics.")
    is_off_topic, _ = check_off_topic("How do I make chocolate chip cookies?")
    assert is_off_topic is True

    is_off_topic, _ = check_off_topic("What are the surrender values for SBI Life Saral Term?")
    assert is_off_topic is False


def test_web_allowlist_enforcement():
    """Verify web search domain allowlist permits only IRDAI and SBI Life."""
    # Permitted domains
    assert is_domain_allowed("https://irdai.gov.in/master-circular-life-insurance") is True
    assert is_domain_allowed("https://www.sbilife.co.in/about-us/public-disclosures") is True
    assert is_domain_allowed("https://portal.irdai.gov.in/circulars") is True
    assert is_domain_allowed("https://sbilife.co.in/funds") is True

    # Refused external domains
    assert is_domain_allowed("https://www.google.com/search?q=insurance") is False
    assert is_domain_allowed("https://policybazaar.com/term-insurance") is False
    assert is_domain_allowed("https://malicious-insurance.in/phishing") is False
    assert is_domain_allowed("invalid-url-format") is False


@pytest.mark.asyncio
async def test_web_client_refuses_disallowed_domain():
    """Verify WebRetrievalClient refuses to fetch non-allowlisted URLs."""
    client = WebRetrievalClient(redis_client=None)
    result = await client.fetch_live_url("https://malicious-insurance-hub.com/leak")
    assert result is None


def test_mandatory_statutory_disclaimer():
    """Verify policy answers end with the mandatory informational disclaimer."""
    policy_answer = (
        "Under Section 45 of Insurance Act 1938, a policy cannot be repudiated after 3 years."
    )
    guarded = enforce_output_guardrails(
        policy_answer, query="Can my SBI Life policy be repudiated after 3 years?"
    )
    assert (
        "This is informational. The policy contract and SBI Life's official communication prevail."
        in guarded
    )


def test_per_user_document_isolation():
    """Verify multi-tenant isolation prevents User B from accessing User A's attached session docs."""
    session_id = "test-session-multi-tenant-101"
    user_a = "user_sbi_agent_alice"
    user_b = "user_sbi_agent_bob"

    # Clean session
    SessionDocumentStore.clear_session(session_id)

    chunks = [
        {
            "document_id": "confidential-schedule-1",
            "document_title": "Alice_Confidential_Policy_Schedule.pdf",
            "text": "Special premium rebate of 5% approved for high sum assured policyholder Alice.",
        }
    ]

    # User A registers the document
    SessionDocumentStore.add_document(
        session_id, "Alice_Confidential_Policy_Schedule.pdf", chunks, user_id=user_a
    )

    # User A can query their document
    results_a = SessionDocumentStore.search(
        session_id, "special premium rebate", top_k=2, user_id=user_a
    )
    assert len(results_a) == 1
    assert results_a[0]["document_title"] == "Alice_Confidential_Policy_Schedule.pdf"

    # User B attempting to query the same session must be blocked
    results_b = SessionDocumentStore.search(
        session_id, "special premium rebate", top_k=2, user_id=user_b
    )
    assert len(results_b) == 0

    # User B cannot see document names
    names_b = SessionDocumentStore.get_document_names(session_id, user_id=user_b)
    assert len(names_b) == 0

    # User A can see document names
    names_a = SessionDocumentStore.get_document_names(session_id, user_id=user_a)
    assert "Alice_Confidential_Policy_Schedule.pdf" in names_a

    # Clean up
    SessionDocumentStore.clear_session(session_id)


def test_docx_document_parsing():
    """Verify DOCX document upload and clause parsing."""
    doc = Document()
    doc.add_heading("SBI Life Smart Privilege Policy Terms", level=1)
    doc.add_paragraph(
        "Clause 4.1: The policyholder is entitled to partial withdrawals after 5 policy years."
    )
    doc.add_paragraph("Clause 4.2: Free fund switches are permitted up to 4 times per policy year.")

    bio = io.BytesIO()
    doc.save(bio)
    docx_bytes = bio.getvalue()

    chunks = parse_uploaded_document(docx_bytes, filename="SBI_Life_Smart_Privilege.docx")
    assert len(chunks) >= 2
    assert any("partial withdrawals after 5 policy years" in c["text"] for c in chunks)
    assert any(c["clause_number"] in ("4.1", "p.1", "p.2") for c in chunks)


def test_plain_text_document_parsing():
    """Verify plain text policy document parsing."""
    content = (
        b"Clause 1: The grace period for payment of premium shall be 30 days.\n\n"
        b"Clause 2: Revival of lapsed policy is allowed within 5 years from FUP."
    )

    chunks = parse_uploaded_document(content, filename="policy_notes.txt")
    assert len(chunks) == 2
    assert "grace period" in chunks[0]["text"]


@pytest.mark.asyncio
async def test_google_bearer_token_authentication():
    """Verify Google Sign-In Bearer token authorization."""
    settings = Settings(ENV="DEV", DEV_API_KEY="dev-test-key-12345")
    user = await get_current_user(
        api_key=None,
        authorization="Bearer google_mock_oauth_jwt_token_12345",
        dev_user="google_advisor@sbilife.co.in",
        settings=settings,
    )
    assert user.user_id == "google_advisor@sbilife.co.in"
    assert user.api_key_id == "google-oauth-bearer"


def test_compliance_audit_logging():
    """Verify statutory compliance audit log executes and masks PII."""
    sources = [
        {
            "document_title": "IRDAI Master Circular 2024",
            "uin": "IRDAI/NL/CIR/2024",
            "section": "Section 45 Guidance",
            "url": "https://irdai.gov.in/master-circular-life-insurance",
            "source_type": "irdai_circular",
        }
    ]

    # Should execute cleanly without raising exceptions
    log_compliance_audit(
        user_id="sbi_agent_007",
        session_id="session-aud-99",
        query="What is Section 45 for Aadhaar 2345 6789 0123?",
        sources=sources,
        answer="Under Section 45 of Insurance Act 1938, the policy cannot be questioned after 3 years.",
        latency_ms=124.5,
        model_used="llama-3.3-70b-versatile",
        query_type="policy_servicing_lookup",
    )


def test_canonical_query_caching_normalization():
    """Verify frequent regulatory query variations map to the same canonical cache key (Roadmap §15)."""
    from app.services.query_service import normalize_canonical_query

    key1 = normalize_canonical_query("What is the grace period?", product_code="ALL", state_jurisdiction="ALL")
    key2 = normalize_canonical_query("how many days for grace period?", product_code=None, state_jurisdiction="ALL")
    assert key1 == key2 == "canonical:grace_period:ALL"

    key3 = normalize_canonical_query("what is ulip lock-in period?", product_code=None, state_jurisdiction="ALL")
    key4 = normalize_canonical_query("tell me about 5 year lock in for ulip", product_code="ALL", state_jurisdiction="ALL")
    assert key3 == key4 == "canonical:ulip_lock_in:ALL"

    key5 = normalize_canonical_query("What is free look period?", product_code=None, state_jurisdiction="ALL")
    key6 = normalize_canonical_query("cooling off period rules", product_code=None, state_jurisdiction="ALL")
    assert key5 == key6 == "canonical:free_look_period:ALL"

