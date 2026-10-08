"""Cross-Encoder Reranker using FlashRank for post-retrieval relevance gating.

Inspired by enterprise-rag's ranking_service.py — replaces the LLM-based
document_grader_node's expensive LLM call with a local ONNX cross-encoder
that runs in <50ms, costs nothing, and is more accurate at per-chunk scoring.

Architecture:
  1. Qdrant returns top-K candidate chunks (dense or hybrid)
  2. FlashRank (ms-marco-MiniLM-L-6-v2 ONNX) cross-encodes each (query, chunk) pair
  3. Chunks scoring below MIN_RERANK_SCORE are dropped
  4. Remaining ranked chunks are returned in relevance order

Relevance Gating:
  If ALL retrieved chunks score below the threshold, an empty list is returned.
  The CRAG grader then routes to web_search instead of allowing the synthesizer
  to hallucinate from parametric memory — critical for insurance compliance.

Why FlashRank over LLM grading:
  - Local ONNX inference: ~8MB model, no API call, ~10-50ms per batch
  - Cross-encoder: evaluates (query, chunk) pairs jointly → far more precise than
vector similarity alone (which is "fuzzy")
  - Zero cost vs ~300ms + token cost for an LLM grader call
"""

from __future__ import annotations

import asyncio
import threading
import time

import logfire

from app.observability.logging import get_logger

logger = get_logger("reranker")

# Minimum relevance score to keep a chunk (0.0–1.0 scale from FlashRank)
MIN_RERANK_SCORE: float = 0.20

# FlashRank cache directory for ONNX weights
_FLASHRANK_CACHE_DIR = "/tmp/flashrank"

_ranker_lock = threading.Lock()
_ranker = None  # type: ignore[assignment]


def _get_ranker():  # type: ignore[return]
    """Initialize the FlashRank cross-encoder lazily (thread-safe singleton).

    Uses ms-marco-MiniLM-L-6-v2 ONNX model — a distilled, quantized bi-encoder
    that provides cross-encoder quality at near-bi-encoder speed.
    """
    global _ranker
    if _ranker is not None:
        return _ranker

    with _ranker_lock:
        if _ranker is not None:
            return _ranker
        try:
            from flashrank import Ranker  # type: ignore[import-untyped]

            _ranker = Ranker(cache_dir=_FLASHRANK_CACHE_DIR)
            logger.info("FLASHRANK_RANKER_INITIALIZED", cache_dir=_FLASHRANK_CACHE_DIR)
        except ImportError:
            logger.warning(
                "FLASHRANK_NOT_INSTALLED",
                message="FlashRank not installed — reranker disabled. Install with: pip install flashrank",
            )
            _ranker = None
        except Exception as e:
            logger.error("FLASHRANK_INIT_FAILED", error=str(e))
            _ranker = None
        return _ranker


async def rerank_clauses(
    query: str,
    clauses: list[dict],
    top_n: int = 5,
    min_score: float = MIN_RERANK_SCORE,
) -> list[dict]:
    """Rerank retrieved policy clauses using FlashRank cross-encoder.

    Args:
        query:     The sanitized user query
        clauses:   List of clause dicts with 'snippet' or 'text' keys
        top_n:     Max number of clauses to return after reranking
        min_score: Minimum FlashRank score to keep a clause (default: 0.20)

    Returns:
        Filtered and re-ordered clause list. Empty list if all below threshold
        (triggers CRAG web-search fallback in document_grader_node).
    """
    if not clauses:
        return []

    with logfire.span(
        "reranker.rerank_clauses",
        query=query,
        input_count=len(clauses),
        min_score=min_score,
    ):
        ranker = _get_ranker()

        # If FlashRank unavailable, fall back to original Qdrant score order
        if ranker is None:
            logger.warning("FLASHRANK_UNAVAILABLE_FALLBACK")
            return clauses[:top_n]

        start = time.perf_counter()

        def _rerank_sync() -> list[dict]:
            from flashrank import RerankRequest  # type: ignore[import-untyped]

            # Build passages list (FlashRank expects [{id, text}, ...])
            passages = [
                {
                    "id": i,
                    "text": c.get("snippet") or c.get("text", ""),
                    "_original": c,  # Carry original clause dict through
                }
                for i, c in enumerate(clauses)
            ]

            request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(request)

            kept: list[dict] = []
            for res in results[:top_n]:
                score = float(res.get("score", 0.0))
                if score >= min_score:
                    # Return original clause dict, enriched with reranker score
                    original_clause = res.get("_original", {}) or clauses[res.get("id", 0)]
                    enriched = {**original_clause, "rerank_score": round(score, 4)}
                    kept.append(enriched)
                else:
                    logger.debug(
                        "CHUNK_DROPPED_RERANK_THRESHOLD",
                        score=round(score, 4),
                        min_score=min_score,
                    )
            return kept

        loop = asyncio.get_running_loop()
        reranked = await loop.run_in_executor(None, _rerank_sync)

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "RERANKER_COMPLETE",
            kept=len(reranked),
            total=len(clauses),
            duration_ms=round(duration_ms, 1),
        )
        return reranked
