"""Unit tests for the 6 enterprise-rag inspired components in Saarthi AI:

1. Hybrid Dense + BM25 Sparse Search (fastembed BM25 vectors)
2. FlashRank Cross-Encoder Reranker
3. Domain Vector Anchor Filter (pre-warming & cosine gating)
4. Token Budget Context Assembly (chunk-boundary safe truncation)
5. Prometheus Metrics & Cache Stats
6. Evaluation Dataset Integrity
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_current_user
from app.api.middleware.auth import AuthenticatedUser
from app.core.graphs.retrieval.nodes.synthesizer import (
    _build_context_with_budget,
    _count_tokens,
)
from app.core.guardrails.domain_filter import (
    INSURANCE_DOMAIN_PROTOTYPES,
    check_domain_relevance,
    initialize_domain_anchors,
    reset_anchors,
)
from app.core.retrieval.reranker import rerank_clauses
from app.core.retrieval.sparse_embeddings import (
    SPARSE_VECTOR_NAME,
    embed_sparse_query,
    embed_sparse_texts,
)
from app.main import app
from app.observability.metrics import (
    record_domain_filter_block,
    record_http_request,
    record_rag_query,
    record_reranker_result,
)

# ── 1. Domain Vector Anchor Filter Tests ──────────────────────────────────────


@pytest.mark.asyncio
async def test_domain_anchor_initialization_and_prewarming():
    """Verify initialize_domain_anchors pre-warms all insurance prototypes in one batch."""
    reset_anchors()

    # Mock embedder returning normalized vectors
    mock_embedder = MagicMock()
    dims = 64
    fake_vectors = [
        (np.random.randn(dims) / np.linalg.norm(np.random.randn(dims))).tolist()
        for _ in INSURANCE_DOMAIN_PROTOTYPES
    ]
    mock_embedder.embed_documents = AsyncMock(return_value=fake_vectors)

    anchors = await initialize_domain_anchors(mock_embedder)
    assert len(anchors) == len(INSURANCE_DOMAIN_PROTOTYPES)
    assert mock_embedder.embed_documents.call_count == 1

    # Verify unit-norm property
    for _label, vec in anchors:
        assert isinstance(vec, np.ndarray)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


@pytest.mark.asyncio
async def test_domain_vector_filter_relevance_check():
    """Verify cosine similarity correctly discriminates in-domain vs out-of-domain."""
    reset_anchors()

    # 5-dimensional orthogonal prototype vectors (dim 4 is 0.0 for all prototypes)
    mock_embedder = MagicMock()
    mock_embedder.embed_documents = AsyncMock(
        return_value=[
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.7071, 0.7071, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    await initialize_domain_anchors(mock_embedder)

    # In-domain query close to Prototype 0
    in_domain_vec = [0.95, 0.1, 0.0, 0.0, 0.0]
    is_in, score, label = check_domain_relevance(
        "What is the surrender value?", in_domain_vec, threshold=0.30
    )
    assert is_in is True
    assert score > 0.80

    # Out-of-domain query purely in dimension 4 (dot product = 0.0 with all prototypes)
    out_domain_vec = [0.0, 0.0, 0.0, 0.0, 1.0]
    is_in_out, score_out, label_out = check_domain_relevance(
        "How to bake butter chicken?", out_domain_vec, threshold=0.30
    )
    assert is_in_out is False
    assert score_out < 0.30

    # Empty query check
    is_empty, score_empty, _ = check_domain_relevance("", [1.0, 0.0, 0.0, 0.0, 0.0])
    assert is_empty is False
    assert score_empty == 0.0


# ── 2. Cross-Encoder Reranker Tests ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_flashrank_reranker_relevance_and_threshold():
    """Verify FlashRank scores query-passage pairs and drops low-scoring chunks."""
    query = "What is Section 45 incontestability period under Insurance Act?"
    clauses = [
        {
            "document_id": "doc-sec45",
            "document_title": "Insurance Act 1938",
            "snippet": "Section 45: No policy of life insurance shall be called in question on any ground after the expiry of three years from the date of issuance.",
            "score": 0.50,
        },
        {
            "document_id": "doc-cake",
            "document_title": "Baking Guide",
            "snippet": "Preheat oven to 350 degrees F and mix sugar with flour thoroughly.",
            "score": 0.45,
        },
    ]

    reranked = await rerank_clauses(query=query, clauses=clauses, top_n=5, min_score=0.10)
    assert len(reranked) >= 1
    # Relevant insurance clause should be ranked #1
    assert reranked[0]["document_id"] == "doc-sec45"
    assert "rerank_score" in reranked[0]

    # Test empty input handling
    assert await rerank_clauses(query="test", clauses=[]) == []


# ── 3. BM25 Sparse Vector Tests ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sparse_bm25_embeddings():
    """Verify BM25 sparse vector generation returns valid Qdrant SparseVector instances."""
    query = "IRDAI Section 45 ULIP"
    sparse_vec = await embed_sparse_query(query)

    assert hasattr(sparse_vec, "indices")
    assert hasattr(sparse_vec, "values")
    assert len(sparse_vec.indices) > 0
    assert len(sparse_vec.indices) == len(sparse_vec.values)
    assert SPARSE_VECTOR_NAME == "sparse"

    # Batch test
    texts = ["Grace period 30 days", "Surrender value computation", "Nomination Section 39"]
    batch_vecs = await embed_sparse_texts(texts)
    assert len(batch_vecs) == len(texts)
    for v in batch_vecs:
        assert len(v.indices) > 0


# ── 4. Token Budget Context Assembly Tests ────────────────────────────────────


def test_token_budget_context_assembly():
    """Verify _build_context_with_budget enforces token limit without mid-chunk truncation."""
    chunks = [
        {"document_id": "1", "snippet": "First policy clause with information about grace period."},
        {
            "document_id": "2",
            "snippet": "Second policy clause regarding surrender value computation and formula.",
        },
        {
            "document_id": "3",
            "snippet": "Third policy clause discussing Section 45 incontestability timelines.",
        },
    ]

    # Large budget: all chunks kept
    full, used, total = _build_context_with_budget(chunks, max_tokens=1000)
    assert len(full) == 3
    assert total == 3

    # Small budget: stops at chunk boundary
    token_1 = _count_tokens(chunks[0]["snippet"])
    budget = token_1 + 1  # Not enough for chunk 2

    budgeted, used_b, total_b = _build_context_with_budget(chunks, max_tokens=budget)
    assert len(budgeted) == 1
    assert budgeted[0]["document_id"] == "1"
    # Never truncated mid-chunk
    assert budgeted[0]["snippet"] == chunks[0]["snippet"]


# ── 5. Prometheus Metrics & Telemetry Tests ───────────────────────────────────


def test_observability_metric_recorders():
    """Verify metric recording helper functions execute safely without error."""
    # Calling metric functions must never raise an exception
    record_rag_query("structured_rag", 0.45)
    record_reranker_result(kept=3, total=5)
    record_domain_filter_block()
    record_http_request("POST", "/api/v1/query", 200, 0.35)


def test_metrics_http_endpoint():
    """Verify /metrics and /cache/stats HTTP endpoints."""
    # Override auth for test client
    mock_user = AuthenticatedUser(
        user_id="test-agent",
        api_key_id="test-api-key-id",
        email="agent@sbilife.co.in",
        role="insurance_agent",
        scopes=["query:read"],
    )
    app.dependency_overrides[get_current_user] = lambda: mock_user

    client = TestClient(app)

    # 1. Prometheus text scrape
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]

    # Also at versioned endpoint
    resp_v1 = client.get("/api/v1/metrics")
    assert resp_v1.status_code == 200

    # 2. Cache stats
    cache_resp = client.get("/api/v1/cache/stats")
    assert cache_resp.status_code == 200
    cache_data = cache_resp.json()
    assert cache_data["backend"] == "redis"
    assert "cache_ttl_seconds" in cache_data
    assert "hit_ratio" in cache_data
    assert "total_lookups" in cache_data

    app.dependency_overrides.clear()


# ── 6. Golden Evaluation Dataset Integrity Tests ─────────────────────────────


def test_ragas_golden_dataset_structure():
    """Verify golden evaluation dataset exists and has valid schema for RAGAS evaluation."""
    golden_path = Path(__file__).resolve().parent.parent / "evaluation" / "golden_dataset.json"
    assert golden_path.exists(), f"Golden dataset missing at: {golden_path}"

    with open(golden_path, encoding="utf-8") as f:
        data = json.load(f)

    # Dataset can be a list or a dict wrapping "rag_samples"
    samples = data.get("rag_samples", data) if isinstance(data, dict) else data
    assert isinstance(samples, list)
    assert len(samples) >= 10, "Golden dataset should have at least 10 insurance samples"

    required_keys = {"id", "domain", "question", "reference", "relevant_contexts", "expected_tools"}
    domains_covered = set()

    for item in samples:
        assert required_keys.issubset(item.keys()), f"Missing keys in sample: {item.get('id')}"
        assert len(item["question"].strip()) > 5
        assert len(item["reference"].strip()) > 5
        assert isinstance(item["relevant_contexts"], list)
        if item.get("domain") not in ("off_topic", "greeting"):
            assert len(item["relevant_contexts"]) > 0
        domains_covered.add(item["domain"])

    # Ensure key insurance domains are covered
    assert any("ulip" in d.lower() for d in domains_covered)
    assert any("45" in d.lower() for d in domains_covered)
    assert any("nomination" in d.lower() for d in domains_covered)
