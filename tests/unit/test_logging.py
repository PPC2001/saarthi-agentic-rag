"""Unit tests for PHI masking and structured logging."""

from app.observability.logging import mask_phi


def test_mask_ssn():
    """Verify Social Security Numbers are masked in logs."""
    raw = "Policyholder SSN is 123-45-6789 for claim verification."
    masked = mask_phi(raw)
    assert "123-45-6789" not in masked
    assert "***-**-****" in masked


def test_mask_email():
    """Verify email addresses are masked in logs."""
    raw = "Send statement to insured.person@example.com immediately."
    masked = mask_phi(raw)
    assert "insured.person@example.com" not in masked
    assert "***@***.***" in masked


def test_mask_phone():
    """Verify phone numbers are masked in logs."""
    raw = "Agent called beneficiary at 800-555-0199."
    masked = mask_phi(raw)
    assert "800-555-0199" not in masked
    assert "***-***-****" in masked


def test_mask_credit_card():
    """Verify credit card numbers are masked in logs."""
    raw = "Premium billed to card 4111-2222-3333-4444."
    masked = mask_phi(raw)
    assert "4111-2222-3333-4444" not in masked
    assert "****-****-****-****" in masked


def test_pytz_timestamp_processor_kolkata():
    """Verify pytz timestamp processor adds ISO timestamp with IST (+05:30) offset."""
    from app.observability.logging import (
        get_localized_now,
        get_localized_timestamp,
        pytz_timestamp_processor,
    )

    processor = pytz_timestamp_processor("Asia/Kolkata")
    event = processor(None, "info", {"event": "user_logged_in"})

    assert "timestamp" in event
    # Asia/Kolkata is UTC+05:30
    assert "+05:30" in event["timestamp"]

    # Verify helper functions
    now = get_localized_now("Asia/Kolkata")
    assert now.tzinfo is not None

    iso_ts = get_localized_timestamp("Asia/Kolkata")
    assert "+05:30" in iso_ts
