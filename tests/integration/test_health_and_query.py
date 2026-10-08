"""Integration tests for health, query, and ingestion endpoints."""

import io

from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """Test GET / returns application information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "saarthi-ai"
    assert data["environment"] == "DEV"


def test_health_endpoint(client: TestClient):
    """Test GET /health returns diagnostic component status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "components" in data
    assert data["environment"] == "DEV"
    assert data["llm_provider"] == "groq"
    assert data["embedding_provider"] in ("qwen", "fastembed")


def test_readiness_and_liveness(client: TestClient):
    """Test Kubernetes readiness and liveness probes."""
    assert client.get("/ready").status_code == 200
    assert client.get("/live").status_code == 200


def test_query_endpoint_unauthorized(client: TestClient):
    """Test POST /v1/query without API key returns 401."""
    response = client.post(
        "/v1/query",
        json={"query": "What is the suicide exclusion period?"},
    )
    assert response.status_code == 401


def test_query_endpoint_authorized(client: TestClient, auth_headers: dict[str, str]):
    """Test POST /v1/query with valid dev authentication."""
    response = client.post(
        "/v1/query",
        headers=auth_headers,
        json={
            "query": "What is the suicide exclusion period for Whole Life Premier in Texas?",
            "product_code": "WLP-2024",
            "state": "TX",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert len(data["citations"]) > 0
    assert data["citations"][0]["product_code"] == "WLP-2024"
    assert "disclaimer" in data

    # Verify security headers and request correlation
    assert "x-request-id" in response.headers
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_ingest_endpoint_valid_pdf(client: TestClient, auth_headers: dict[str, str]):
    """Test POST /v1/ingest accepts a valid PDF file."""
    fake_pdf = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
    files = {"file": ("master_policy.pdf", io.BytesIO(fake_pdf), "application/pdf")}
    data = {
        "product_name": "Whole Life Premier",
        "product_code": "WLP-2024",
        "product_line": "whole_life",
        "document_type": "master_policy",
        "state": "TX",
    }
    response = client.post(
        "/v1/ingest",
        headers=auth_headers,
        data=data,
        files=files,
    )
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "accepted"
    assert res["chunks_created"] >= 1


def test_ingest_endpoint_rejects_disallowed_extension(
    client: TestClient, auth_headers: dict[str, str]
):
    """Test POST /v1/ingest rejects unauthorized file types like .exe or .sh."""
    files = {"file": ("malicious.sh", io.BytesIO(b"echo 'bad'"), "text/x-shellscript")}
    data = {
        "product_name": "Term Life",
        "product_code": "TERM-20",
        "product_line": "term",
        "document_type": "master_policy",
        "state": "ALL",
    }
    response = client.post(
        "/v1/ingest",
        headers=auth_headers,
        data=data,
        files=files,
    )
    assert response.status_code == 400
