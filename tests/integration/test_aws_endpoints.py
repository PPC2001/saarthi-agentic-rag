"""Integration tests for AWS-backed endpoints (S3 presigned URLs, DynamoDB document catalog and status)."""

import io

from fastapi.testclient import TestClient


def test_ingest_and_aws_tracking(client: TestClient, auth_headers: dict[str, str]):
    """Test full document ingest pipeline with S3 archive and DynamoDB tracking."""
    fake_pdf = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
    files = {"file": ("hdfc_life_term.pdf", io.BytesIO(fake_pdf), "application/pdf")}
    data = {
        "product_name": "HDFC Click 2 Protect",
        "product_code": "HDFC-C2P",
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
    assert response.status_code == 200
    res = response.json()
    assert res["status"] == "accepted"
    doc_id = res["document_id"]
    assert doc_id is not None
    assert "s3_uri" in res
    assert "download_url" in res

    # 1. Query status from DynamoDB endpoint
    status_res = client.get(f"/v1/ingest/{doc_id}/status", headers=auth_headers)
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["doc_id"] == doc_id
    assert status_data["filename"] == "hdfc_life_term.pdf"
    assert status_data["product_code"] == "HDFC-C2P"
    assert status_data["status"] in ("INDEXED", "PROCESSING", "INDEX_ERROR")

    # 2. Query S3 presigned download URL
    url_res = client.get(f"/v1/ingest/{doc_id}/download-url", headers=auth_headers)
    assert url_res.status_code == 200
    url_data = url_res.json()
    assert "download_url" in url_data
    assert "hdfc_life_term.pdf" in url_data["download_url"]

    # 3. List registered documents from DynamoDB
    catalog_res = client.get("/v1/ingest/catalog", headers=auth_headers)
    assert catalog_res.status_code == 200
    catalog = catalog_res.json()
    assert isinstance(catalog, list)
    assert any(d.get("doc_id") == doc_id for d in catalog)
