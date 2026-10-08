"""Common API dependencies for authentication, database clients, and configuration.

Modeled after enterprise FastAPI standards to provide reusable, type-annotated
dependency injections across all API endpoints.
"""

from typing import Annotated

from fastapi import Depends
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from app.api.middleware.auth import (
    AuthenticatedUser,
    get_current_user,
    get_optional_current_user,
    require_roles,
)
from app.config import Settings, get_settings
from app.core.aws.dynamodb_service import DynamoDBService
from app.core.aws.s3_service import S3Service
from app.core.aws.sqs_service import SQSService
from app.core.llm_gateway.embeddings import Embedder
from app.dependencies import (
    get_checkpointer,
    get_dynamodb_service,
    get_embedder,
    get_llm,
    get_qdrant_client,
    get_redis_client,
    get_s3_service,
    get_sqs_service,
)
from app.services.query_service import QueryService

# Re-usable Type Aliases for Route Handlers
CurrentUser = Annotated[AuthenticatedUser, Depends(get_current_user)]
OptionalCurrentUser = Annotated[AuthenticatedUser, Depends(get_optional_current_user)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
QdrantDep = Annotated[AsyncQdrantClient, Depends(get_qdrant_client)]
RedisDep = Annotated[Redis | None, Depends(get_redis_client)]
LLMDep = Annotated[ChatOpenAI | None, Depends(get_llm)]
SmartLLMDep = LLMDep
FastLLMDep = LLMDep
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
CheckpointerDep = Annotated[BaseCheckpointSaver, Depends(get_checkpointer)]

# AWS Service Dependencies
S3Dep = Annotated[S3Service, Depends(get_s3_service)]
DynamoDBDep = Annotated[DynamoDBService, Depends(get_dynamodb_service)]
SQSDep = Annotated[SQSService, Depends(get_sqs_service)]


def get_query_service(
    llm: LLMDep,
    embedder: EmbedderDep,
    qdrant: QdrantDep,
    redis_client: RedisDep,
    settings: SettingsDep,
    checkpointer: CheckpointerDep,
) -> QueryService:
    """Dependency provider for QueryService adhering to Dependency Inversion."""
    return QueryService(
        llm=llm,
        embedder=embedder,
        qdrant=qdrant,
        redis_client=redis_client,
        settings=settings,
        checkpointer=checkpointer,
    )


QueryServiceDep = Annotated[QueryService, Depends(get_query_service)]

__all__ = [
    "AuthenticatedUser",
    "CheckpointerDep",
    "CurrentUser",
    "OptionalCurrentUser",
    "DynamoDBDep",
    "EmbedderDep",
    "FastLLMDep",
    "LLMDep",
    "QdrantDep",
    "QueryService",
    "QueryServiceDep",
    "RedisDep",
    "S3Dep",
    "SQSDep",
    "SettingsDep",
    "SmartLLMDep",
    "get_checkpointer",
    "get_current_user",
    "get_dynamodb_service",
    "get_embedder",
    "get_llm",
    "get_qdrant_client",
    "get_query_service",
    "get_redis_client",
    "get_s3_service",
    "get_settings",
    "get_sqs_service",
    "require_roles",
]
