"""Sparse (BM25) embedding service for Qdrant hybrid retrieval.

Uses fastembed's Qdrant/bm25 model — same approach as enterprise-rag's
sparse_embeddings.py — for vocabulary-consistent BM25 scoring at both
ingestion and query time.

Why fastembed/Qdrant BM25 over rank_bm25:
  - Native Qdrant SparseVector format (indices + values)
  - Stateless per-document tokenization — no corpus-level IDF fitting needed
  - Same model at ingestion AND query time → vocabulary consistency guaranteed
  - Zero extra dependency: fastembed is already in pyproject.toml
"""

from __future__ import annotations

import asyncio
import threading

from fastembed import SparseTextEmbedding
from qdrant_client.http.models import SparseVector

from app.observability.logging import get_logger

logger = get_logger("sparse_embeddings")

# Qdrant's official BM25 implementation via fastembed
_BM25_MODEL = "Qdrant/bm25"

# Named vector field stored in Qdrant — must match what is used at ingestion
SPARSE_VECTOR_NAME = "sparse"

_model_lock = threading.Lock()
_bm25_model: SparseTextEmbedding | None = None


def _get_bm25_model() -> SparseTextEmbedding:
    """Lazy singleton BM25 model (thread-safe double-checked locking).

    SparseTextEmbedding downloads ONNX weights on first call.
    lru_cache cannot be used here because we need thread-safety.
    """
    global _bm25_model
    if _bm25_model is not None:
        return _bm25_model

    with _model_lock:
        if _bm25_model is not None:
            return _bm25_model
        _bm25_model = SparseTextEmbedding(model_name=_BM25_MODEL)
        logger.info("BM25_SPARSE_MODEL_LOADED", model=_BM25_MODEL)
        return _bm25_model


async def embed_sparse_query(query: str) -> SparseVector:
    """Encode a single query string into a Qdrant SparseVector for BM25 retrieval.

    Runs the synchronous fastembed call in a thread executor to avoid blocking
    the async event loop.

    Returns:
        SparseVector(indices=[...], values=[...]) ready for qdrant_client.query_points()
    """
    loop = asyncio.get_running_loop()

    def _encode() -> SparseVector:
        model = _get_bm25_model()
        result = next(model.query_embed(query))
        return SparseVector(
            indices=result.indices.tolist(),
            values=result.values.tolist(),
        )

    vec = await loop.run_in_executor(None, _encode)
    logger.info("BM25_QUERY_EMBEDDING_GENERATED", nnz=len(vec.indices))
    return vec


async def embed_sparse_texts(texts: list[str]) -> list[SparseVector]:
    """Encode a batch of document texts into SparseVectors for Qdrant indexing.

    Used during ingestion to create the 'sparse' named vector on each Qdrant point.
    """
    loop = asyncio.get_running_loop()

    def _encode_batch() -> list[SparseVector]:
        model = _get_bm25_model()
        return [
            SparseVector(
                indices=res.indices.tolist(),
                values=res.values.tolist(),
            )
            for res in model.passage_embed(texts)
        ]

    vecs = await loop.run_in_executor(None, _encode_batch)
    logger.info("BM25_TEXTS_EMBEDDED", count=len(vecs))
    return vecs
