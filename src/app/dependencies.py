"""FastAPI Dependency Injection container.

Wires configuration, abstract LLM interfaces, vector database, cache,
and audit systems into endpoints and LangGraph workflows without hardcoding providers.
"""

from typing import TYPE_CHECKING

from fastapi import Depends
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

if TYPE_CHECKING:
    from app.core.aws.dynamodb_service import DynamoDBService
    from app.core.aws.s3_service import S3Service
    from app.core.aws.sqs_service import SQSService

from app.config import Settings, get_settings
from app.core.http_clients import close_http_client
from app.core.llm_gateway.client import create_llm
from app.core.llm_gateway.embeddings import Embedder, create_embedder
from app.observability.audit import AuditLogger
from app.observability.logging import get_logger

logger = get_logger("dependencies")

# Singletons for connection pools & models
_qdrant_client: AsyncQdrantClient | None = None
_redis_client: Redis | None = None
_audit_logger: AuditLogger | None = None
_embedder: Embedder | None = None
_llm: ChatOpenAI | None = None
_checkpointer: BaseCheckpointSaver | None = None


def get_llm(settings: Settings = Depends(get_settings)) -> ChatOpenAI:
    """Inject the LLM (one model for all tasks — routing, synthesis, grading)."""
    global _llm
    if _llm is None:
        _llm = create_llm(settings)
    return _llm


def get_embedder(settings: Settings = Depends(get_settings)) -> Embedder:
    """Inject the embedding provider (Titan V2 in PROD, FastEmbed in DEV)."""
    global _embedder
    if _embedder is None:
        _embedder = create_embedder(settings)
    return _embedder


def get_audit_logger(settings: Settings = Depends(get_settings)) -> AuditLogger:
    """Inject the HIPAA audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(settings)
    return _audit_logger


def get_qdrant_client(settings: Settings = Depends(get_settings)) -> AsyncQdrantClient:
    """Inject asynchronous Qdrant vector database client."""
    global _qdrant_client
    if _qdrant_client is None:
        logger.info(
            "Connecting to Qdrant Cloud",
            url=settings.QDRANT_URL,
        )
        _qdrant_client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=10.0,
        )
    return _qdrant_client


def get_redis_client(settings: Settings = Depends(get_settings)) -> Redis:
    """Inject Redis client for semantic caching."""
    global _redis_client
    if _redis_client is None:
        logger.info("Connecting to Redis", url=settings.REDIS_URL)
        _redis_client = Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_timeout=5.0,
        )
    return _redis_client


def get_s3_service() -> "S3Service":
    """Inject Amazon S3 service."""
    from app.core.aws.s3_service import get_s3_service as _get_s3
    return _get_s3()


def get_dynamodb_service() -> "DynamoDBService":
    """Inject Amazon DynamoDB service."""
    from app.core.aws.dynamodb_service import get_dynamodb_service as _get_dynamo
    return _get_dynamo()


def get_sqs_service() -> "SQSService":
    """Inject Amazon SQS service."""
    from app.core.aws.sqs_service import get_sqs_service as _get_sqs
    return _get_sqs()


def get_checkpointer(
    settings: Settings = Depends(get_settings),
    redis_client: Redis | None = Depends(get_redis_client),
) -> BaseCheckpointSaver:
    """Inject LangGraph state checkpointer (DynamoDB in AWS, Redis, or Memory)."""
    global _checkpointer
    if _checkpointer is None:
        from app.core.graphs.checkpointer import get_checkpointer as _create_cp
        _checkpointer = _create_cp(settings, redis_client=redis_client)
    return _checkpointer


async def close_connections() -> None:
    """Clean up open connection pools on application shutdown."""
    global _qdrant_client, _redis_client, _embedder, _audit_logger, _llm, _checkpointer
    if _qdrant_client is not None:
        try:
            await _qdrant_client.close()
        except Exception as e:
            logger.debug("QDRANT_CLOSE_EXCEPTION", error=str(e))
        _qdrant_client = None
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except Exception as e:
            logger.debug("REDIS_CLOSE_EXCEPTION", error=str(e))
        _redis_client = None
    try:
        await close_http_client()
    except Exception as e:
        logger.debug("HTTP_CLIENT_CLOSE_EXCEPTION", error=str(e))
    _embedder = None
    _audit_logger = None
    _llm = None
    _checkpointer = None
    try:
        from app.core.aws.session import reset_sessions
        reset_sessions()
    except Exception:
        pass
    logger.info("External connection pools closed")
