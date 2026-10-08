"""Integration tests for Indian Life Insurance domain and IRDAI compliance."""

from fastapi.testclient import TestClient


def test_indian_life_policy_query_saral_term(client: TestClient, auth_headers: dict[str, str]):
    """Test querying Saral Term suicide exclusion under IRDAI regulations."""
    response = client.post(
        "/v1/query",
        headers=auth_headers,
        json={
            "query": "What is the claim settlement payout if death occurs by suicide within 12 months under Saral Term?",
            "product_code": "SARAL-TERM",
            "state": "ALL",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert len(data["citations"]) > 0

    first_cit = data["citations"][0]
    assert first_cit["product_code"] == "SARAL-TERM"
    assert first_cit["uin"] == "111N090V03"
    assert "Section 45" in data["disclaimer"] or "IRDAI" in data["disclaimer"]


def test_indian_life_policy_query_ulip_lockin(client: TestClient, auth_headers: dict[str, str]):
    """Test querying ULIP 5-year statutory lock-in period and partial withdrawal rules."""
    response = client.post(
        "/v1/query",
        headers=auth_headers,
        json={
            "query": "What are the lock-in rules and partial withdrawal limits for ULIP Wealth?",
            "product_code": "ULIP-WEALTH",
            "state": "MH",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["citations"]) > 0
    assert data["citations"][0]["uin"] == "111L120V02"
    assert "lock-in" in data["citations"][0]["snippet"].lower()


def test_indian_life_policy_query_sanchay_loan(client: TestClient, auth_headers: dict[str, str]):
    """Test querying loan against policy for Sanchay Guaranteed Endowment."""
    response = client.post(
        "/v1/query",
        headers=auth_headers,
        json={
            "query": "What is the policy loan eligibility and maximum limit for Sanchay Guaranteed?",
            "product_code": "SANCHAY-GUARANTEED",
            "state": "DL",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["citations"]) > 0
    assert data["citations"][0]["uin"] == "111N115V04"
    assert "loan" in data["citations"][0]["snippet"].lower()


def test_no_forbidden_company_names_in_response(client: TestClient, auth_headers: dict[str, str]):
    """Verify that forbidden corporate entity names do not appear anywhere in responses."""
    response = client.post(
        "/v1/query",
        headers=auth_headers,
        json={
            "query": "Tell me about life insurance policy terms and conditions",
            "product_code": "SARAL-TERM",
        },
    )
    assert response.status_code == 200
    raw_text = response.text.lower()
    assert "sbi" not in raw_text
    assert "state bank of india" not in raw_text


def test_greeting_does_not_return_policy_citations(
    client: TestClient, auth_headers: dict[str, str]
):
    """Verify that greetings like 'hii' or 'hello' return welcoming responses without contract citations."""
    for greet in ["hii", "Hello", "good morning", "Namaste"]:
        response = client.post(
            "/v1/query",
            headers=auth_headers,
            json={"query": greet, "product_code": "SARAL-TERM"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query_type"] == "greeting"
        assert len(data["citations"]) == 0
        assert "Hello!" in data["answer"]
        assert "Upon death of the Life Assured" not in data["answer"]


def test_gratitude_does_not_return_policy_citations(
    client: TestClient, auth_headers: dict[str, str]
):
    """Verify that gratitude expressions like 'thank you' return polite responses without contract citations."""
    response = client.post(
        "/v1/query",
        headers=auth_headers,
        json={"query": "thank you", "product_code": "SARAL-TERM"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query_type"] == "gratitude"
    assert len(data["citations"]) == 0
    assert "welcome" in data["answer"].lower()
