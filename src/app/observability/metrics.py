"""Prometheus metrics registry and telemetry helpers for Saarthi AI.

Centralizes metrics definitions away from API controllers to eliminate
circular dependencies between endpoints, middleware, and domain services.

Metrics exposed:
  - saarthi_http_requests_total{method, endpoint, status_code} — request counter
  - saarthi_http_request_duration_seconds{method, endpoint} — latency histogram
  - saarthi_rag_queries_total{query_type} — RAG query counter by type
  - saarthi_rag_query_duration_seconds — end-to-end RAG pipeline latency
  - saarthi_reranker_chunks_kept_ratio — FlashRank relevance gate ratio
  - saarthi_domain_filter_blocked_total — domain vector filter rejection counter
  - saarthi_cache_hit_ratio — semantic Redis cache hit ratio
  - saarthi_cache_size — number of cached query-response pairs
"""

# ── Prometheus Counters, Gauges & Histograms ─────────────────────────────────
try:
    from prometheus_client import (  # type: ignore[import-untyped]
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True

    # Request metrics
    HTTP_REQUESTS_TOTAL = Counter(
        "saarthi_http_requests_total",
        "Total HTTP requests by method, endpoint, and status code",
        ["method", "endpoint", "status_code"],
    )
    HTTP_REQUEST_DURATION = Histogram(
        "saarthi_http_request_duration_seconds",
        "HTTP request duration in seconds",
        ["method", "endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )

    # RAG pipeline metrics
    RAG_QUERIES_TOTAL = Counter(
        "saarthi_rag_queries_total",
        "Total RAG queries processed by intent type",
        ["query_type"],
    )
    RAG_QUERY_DURATION = Histogram(
        "saarthi_rag_query_duration_seconds",
        "End-to-end RAG pipeline latency in seconds",
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    )

    # Quality metrics
    RERANKER_CHUNKS_KEPT = Histogram(
        "saarthi_reranker_chunks_kept_ratio",
        "Fraction of retrieved chunks kept after FlashRank relevance gating (0.0–1.0)",
        buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    DOMAIN_FILTER_BLOCKED = Counter(
        "saarthi_domain_filter_blocked_total",
        "Total queries blocked by domain vector anchor filter",
    )

    # Cache metrics
    CACHE_HIT_RATIO = Gauge(
        "saarthi_cache_hit_ratio",
        "Current semantic cache hit ratio (0.0–1.0)",
    )
    CACHE_SIZE = Gauge(
        "saarthi_cache_size",
        "Number of entries in the semantic Redis cache",
    )

except ImportError:
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

    # Stub no-ops so caller code doesn't need try/except guards
    class _NoOpMetric:
        def labels(self, **kwargs):
            return self

        def inc(self, *a, **kw):
            pass

        def observe(self, *a, **kw):
            pass

        def set(self, *a, **kw):
            pass

    HTTP_REQUESTS_TOTAL = _NoOpMetric()  # type: ignore[assignment]
    HTTP_REQUEST_DURATION = _NoOpMetric()  # type: ignore[assignment]
    RAG_QUERIES_TOTAL = _NoOpMetric()  # type: ignore[assignment]
    RAG_QUERY_DURATION = _NoOpMetric()  # type: ignore[assignment]
    RERANKER_CHUNKS_KEPT = _NoOpMetric()  # type: ignore[assignment]
    DOMAIN_FILTER_BLOCKED = _NoOpMetric()  # type: ignore[assignment]
    CACHE_HIT_RATIO = _NoOpMetric()  # type: ignore[assignment]
    CACHE_SIZE = _NoOpMetric()  # type: ignore[assignment]

    def generate_latest() -> bytes:  # type: ignore[misc]
        return b"# prometheus_client not installed\n"


# ── Helper functions (called from pipeline nodes & services) ─────────────────


def record_rag_query(query_type: str, duration_seconds: float) -> None:
    """Record a completed RAG query with its category and latency."""
    RAG_QUERIES_TOTAL.labels(query_type=query_type).inc()
    RAG_QUERY_DURATION.observe(duration_seconds)


def record_reranker_result(kept: int, total: int) -> None:
    """Record FlashRank cross-encoder chunks kept ratio."""
    if total > 0:
        RERANKER_CHUNKS_KEPT.observe(kept / total)


def record_domain_filter_block() -> None:
    """Increment domain vector anchor filter block counter."""
    DOMAIN_FILTER_BLOCKED.inc()


def record_http_request(method: str, endpoint: str, status_code: int, duration: float) -> None:
    """Record an HTTP request metric."""
    HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
    HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
