import time

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.deps import QdrantDep, RedisDep, SettingsDep
from app.observability.logging import get_logger

router = APIRouter(tags=["Health"])
logger = get_logger("health")


class ComponentHealth(BaseModel):
    """Health status of an external dependency."""

    status: str  # "healthy", "unhealthy", "disabled", "not_configured"
    latency_ms: float | None = None
    message: str | None = None


class HealthResponse(BaseModel):
    """Aggregate health check response."""

    status: str  # "healthy", "degraded", "unhealthy"
    environment: str
    version: str
    llm_provider: str
    embedding_provider: str
    components: dict[str, ComponentHealth] = Field(default_factory=dict)
    timestamp: float


@router.get("/health", response_model=HealthResponse)
async def check_health(
    settings: SettingsDep,
    qdrant: QdrantDep,
    redis_client: RedisDep,
) -> HealthResponse:
    """Check connectivity to all core external dependencies."""
    components: dict[str, ComponentHealth] = {}
    is_degraded = False

    # 1. Check Qdrant
    start = time.perf_counter()
    try:
        await qdrant.get_collections()
        components["qdrant"] = ComponentHealth(
            status="healthy",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )
    except Exception as e:
        logger.warning("Health check: Qdrant unreachable", error=str(e))
        components["qdrant"] = ComponentHealth(
            status="unhealthy",
            message=str(e),
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
        )
        is_degraded = True

    # 2. Check Amazon ElastiCache (Redis)
    if redis_client is not None:
        start = time.perf_counter()
        try:
            await redis_client.ping()
            components["elasticache_redis"] = ComponentHealth(
                status="healthy",
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
                message="Amazon ElastiCache (Redis) connected and responsive",
            )
        except Exception as e:
            logger.warning("Health check: Redis unreachable", error=str(e))
            components["elasticache_redis"] = ComponentHealth(
                status="unhealthy",
                message=str(e),
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )
            is_degraded = True
    else:
        components["elasticache_redis"] = ComponentHealth(
            status="disabled",
            message="ElastiCache Redis cache is not configured",
        )

    # 3. Check Amazon S3
    from app.core.aws.s3_service import get_s3_service
    s3 = get_s3_service()
    if s3.is_available():
        start = time.perf_counter()
        s3_status = await s3.check_health()
        components["amazon_s3"] = ComponentHealth(
            status=s3_status.get("status", "unknown"),
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            message=f"Bucket: {s3.bucket_name} ({s3_status.get('detail', 'connected')})",
        )
    else:
        components["amazon_s3"] = ComponentHealth(
            status="inactive",
            message="AWS credentials or S3 bucket not configured (mock mode active)",
        )

    # 4. Check Amazon DynamoDB
    from app.core.aws.dynamodb_service import get_dynamodb_service
    dynamo = get_dynamodb_service()
    if dynamo.is_available():
        start = time.perf_counter()
        dyn_status = await dynamo.check_health()
        components["amazon_dynamodb"] = ComponentHealth(
            status=dyn_status.get("status", "unknown"),
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            message=f"Table: {dynamo.table_documents} ({dyn_status.get('detail', 'connected')})",
        )
    else:
        components["amazon_dynamodb"] = ComponentHealth(
            status="inactive",
            message="AWS credentials not configured (mock mode active)",
        )

    # 5. Check Amazon SQS
    from app.core.aws.sqs_service import get_sqs_service
    sqs = get_sqs_service()
    if sqs.is_available():
        start = time.perf_counter()
        sqs_status = await sqs.check_health()
        components["amazon_sqs"] = ComponentHealth(
            status=sqs_status.get("status", "unknown"),
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            message=f"Queue: {sqs.queue_name} ({sqs_status.get('detail', 'connected')})",
        )
    else:
        components["amazon_sqs"] = ComponentHealth(
            status="inactive",
            message="AWS credentials not configured (mock mode active)",
        )

    # 6. Check AWS Secrets Manager
    from app.core.aws.session import has_aws_credentials
    if has_aws_credentials():
        components["aws_secrets_manager"] = ComponentHealth(
            status="healthy",
            message=f"AWS credentials active; Secret: {settings.SECRET_NAME}",
        )
    else:
        components["aws_secrets_manager"] = ComponentHealth(
            status="inactive",
            message="AWS credentials not configured; using local .env variables",
        )

    # 7. LLM Gateway / Provider Configuration
    components["llm_gateway"] = ComponentHealth(
        status="healthy",
        message=f"Configured for {settings.LLM_PROVIDER.value} (model: {settings.LLM_MODEL})",
    )

    # 8. Embedding Provider
    components["embedding_provider"] = ComponentHealth(
        status="healthy",
        message=f"Configured for {settings.EMBEDDING_PROVIDER.value} ({settings.EMBEDDING_DIMENSIONS}d)",
    )

    overall_status = "degraded" if is_degraded else "healthy"

    logger.info(
        "HEALTH_CHECK_DIAGNOSTICS",
        status=overall_status,
        qdrant_status=components["qdrant"].status,
        redis_status=components["elasticache_redis"].status,
    )

    return HealthResponse(
        status=overall_status,
        environment=settings.ENV.value,
        version=settings.APP_VERSION,
        llm_provider=settings.LLM_PROVIDER.value,
        embedding_provider=settings.EMBEDDING_PROVIDER.value,
        components=components,
        timestamp=time.time(),
    )


@router.get("/ready")
async def readiness_probe() -> dict[str, str]:
    """Kubernetes / ECS readiness probe."""
    return {"status": "ready"}


@router.get("/live")
async def liveness_probe() -> dict[str, str]:
    """Kubernetes / ECS liveness probe."""
    return {"status": "live"}
