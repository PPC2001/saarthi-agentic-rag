"""Prometheus metrics and cache endpoints for Saarthi AI.

Exposes:
  GET /metrics — Prometheus text exposition format for Grafana / CloudWatch scraping
  GET /api/v1/cache/stats — Operational KPI monitoring (hit ratio, size, TTL)
  DELETE /api/v1/cache/clear — Flush Redis query cache (admin operation)
"""

import time

from fastapi import APIRouter, Response
from pydantic import BaseModel

from app.api.deps import RedisDep, SettingsDep
from app.observability.logging import get_logger
from app.observability.metrics import (
    CACHE_HIT_RATIO,
    CACHE_SIZE,
    CONTENT_TYPE_LATEST,
    PROMETHEUS_AVAILABLE,
    generate_latest,
    record_domain_filter_block,
    record_http_request,
    record_rag_query,
    record_reranker_result,
)

logger = get_logger("metrics_endpoint")
router = APIRouter(tags=["Observability"])

# Re-export for backwards compatibility
__all__ = [
    "CacheStatsResponse",
    "clear_cache",
    "cache_stats",
    "prometheus_metrics",
    "record_domain_filter_block",
    "record_http_request",
    "record_rag_query",
    "record_reranker_result",
    "router",
]


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    """Prometheus-compatible metrics endpoint for Grafana / CloudWatch Agent scraping."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


class CacheStatsResponse(BaseModel):
    """Semantic cache runtime statistics."""

    backend: str = "redis"
    redis_url: str
    cache_ttl_seconds: int
    similarity_threshold: float
    total_lookups: int = 0
    total_hits: int = 0
    hit_ratio: float = 0.0
    cache_size: int = 0
    prometheus_available: bool = False
    timestamp: float


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(
    settings: SettingsDep,
    redis_client: RedisDep,
) -> CacheStatsResponse:
    """Return semantic cache operational statistics.

    These are critical production KPIs:
    - hit_ratio < 0.20 → most queries are hitting the expensive LLM pipeline
    - cache_size → number of cached Q&A pairs (grows over time)
    """
    cache_size = 0
    total_lookups = 0
    total_hits = 0
    hit_ratio = 0.0
    try:
        if redis_client is not None:
            # Count cached query keys
            cursor = 0
            total = 0
            while True:
                cursor, keys = await redis_client.scan(
                    cursor, match="saarthi:cache:query:*", count=100
                )
                total += len(keys)
                if cursor == 0:
                    break
            cache_size = total
            CACHE_SIZE.set(cache_size)

            # Read performance counters
            raw_lookups = await redis_client.get("saarthi:cache:lookups")
            raw_hits = await redis_client.get("saarthi:cache:hits")
            total_lookups = int(raw_lookups) if raw_lookups else 0
            total_hits = int(raw_hits) if raw_hits else 0
            hit_ratio = round(total_hits / total_lookups, 4) if total_lookups > 0 else 0.0
            CACHE_HIT_RATIO.set(hit_ratio)
    except Exception as e:
        logger.warning("CACHE_STATS_REDIS_ERROR", error=str(e))

    return CacheStatsResponse(
        backend="redis",
        redis_url=settings.REDIS_URL.split("@")[-1],  # Mask credentials
        cache_ttl_seconds=settings.REDIS_CACHE_TTL_SECONDS,
        similarity_threshold=settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
        total_lookups=total_lookups,
        total_hits=total_hits,
        hit_ratio=hit_ratio,
        cache_size=cache_size,
        prometheus_available=PROMETHEUS_AVAILABLE,
        timestamp=time.time(),
    )


@router.delete("/cache/clear", summary="Flush semantic cache")
async def clear_cache(
    redis_client: RedisDep,
) -> dict:
    """Flush all entries from the semantic Redis cache.

    Admin operation — useful when index has been re-ingested with corrected documents.
    """
    cleared = 0
    try:
        if redis_client is not None:
            cursor = 0
            keys_to_delete = []
            while True:
                cursor, keys = await redis_client.scan(
                    cursor, match="saarthi:cache:*", count=100
                )
                keys_to_delete.extend(keys)
                if cursor == 0:
                    break
            if keys_to_delete:
                await redis_client.delete(*keys_to_delete)
                cleared = len(keys_to_delete)
            logger.info("SEMANTIC_CACHE_CLEARED", count=cleared)
            CACHE_SIZE.set(0)
    except Exception as e:
        logger.warning("CACHE_CLEAR_ERROR", error=str(e))

    return {"status": "ok", "keys_cleared": cleared, "timestamp": time.time()}
