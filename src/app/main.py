"""FastAPI application entrypoint for Life Insurance Policy Servicing RAG Platform.

Configured with:
- Lifespan management (startup & graceful shutdown)
- Multi-tier observability (Logfire, structlog, request correlation)
- HIPAA compliance audit trail middleware
- Sliding window rate limiting
- Environment-driven provider switching (Groq/fastembed in DEV, Bedrock/Titan in PROD)
- Secure exception handling and header protection
"""

import os
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import logfire
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.api_v1.endpoints import health, info, metrics
from app.api.api_v1.router import api_router
from app.api.middleware.audit import HIPAAAuditMiddleware
from app.api.middleware.logging import RequestCorrelationMiddleware
from app.api.middleware.rate_limit import RateLimitMiddleware
from app.api.schemas.common import APIErrorResponse, ErrorDetail
from app.api.schemas.exceptions import BaseDomainException
from app.config import get_settings
from app.core.guardrails.domain_filter import initialize_domain_anchors
from app.dependencies import close_connections, get_embedder
from app.observability.logfire_setup import setup_logfire
from app.observability.logging import get_logger, setup_logging

warnings.filterwarnings("ignore", message=".*Could not obtain an event loop.*", category=UserWarning)

settings = get_settings()

# Configure LangSmith distributed tracing for LangChain and LangGraph
if settings.LANGSMITH_API_KEY and settings.LANGSMITH_TRACING:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
else:
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ── Logging & Observability init order matters:
# 1. setup_logging() first  → structlog chain is configured
# 2. setup_logfire() second → OTEL exporter configured; StructlogProcessor injected into chain
setup_logging(
    is_dev=(settings.ENV.value == "DEV"),
    log_level=settings.LOG_LEVEL,
    timezone=settings.TIMEZONE,
)

# Initialize Logfire telemetry AFTER logging so all loggers and spans use the project credentials
setup_logfire(None, settings)

logger = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler for application startup and shutdown."""
    with logfire.span("app.lifespan", app_name=settings.APP_NAME, environment=settings.ENV.value):
        logfire.info(f"🚀 Starting Saarthi AI application ({settings.ENV.value})...")
        logger.info(
            "APPLICATION_STARTUP",
            app_name=settings.APP_NAME,
            environment=settings.ENV.value,
            version=settings.APP_VERSION,
            llm_provider=settings.LLM_PROVIDER.value,
            embedding_provider=settings.EMBEDDING_PROVIDER.value,
            embedding_dimensions=settings.EMBEDDING_DIMENSIONS,
            hipaa_mode=settings.HIPAA_MODE,
        )

        # Log LangSmith tracing status
        if settings.LANGSMITH_API_KEY and settings.LANGSMITH_TRACING:
            logfire.info(f"📊 LangSmith tracing active (Project: '{settings.LANGSMITH_PROJECT}')")
        else:
            logfire.info("📊 LangSmith tracing inactive.")

        # Pre-warm domain vector anchors for sub-5ms guardrail filtering
        try:
            embedder = get_embedder(settings)
            await initialize_domain_anchors(embedder)
        except Exception as e:
            logger.warning("DOMAIN_ANCHORS_STARTUP_FAILED", error=str(e))

        yield

        logfire.info("🛑 Shutting down Saarthi AI application...")
        logger.info("APPLICATION_SHUTDOWN_INITIATED")
        try:
            logfire.force_flush()
        except Exception:
            pass
        await close_connections()
        logger.info("APPLICATION_SHUTDOWN_COMPLETED")


app = FastAPI(
    title="Life Insurance Policy Servicing RAG API",
    description=(
        "Enterprise RAG platform for life insurance policy servicing. "
        "Enables agents and CSRs to query master contracts, riders, endorsements, and exclusions "
        "with legal citation verification and HIPAA audit compliance."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Logfire Observability Instrumentation ──
setup_logfire(app, settings)

# ── Middlewares (Order: outermost executed first) ──
# 1. Request correlation & security headers
app.add_middleware(RequestCorrelationMiddleware)

# 2. Rate limiting
app.add_middleware(RateLimitMiddleware)

# 3. HIPAA Audit trail
app.add_middleware(HIPAAAuditMiddleware)

# 4. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)


# ── Exception Handlers ──
@app.exception_handler(BaseDomainException)
async def domain_exception_handler(request: Request, exc: BaseDomainException) -> JSONResponse:
    """Handle RFC 7807 typed domain exceptions with trace links."""
    problem = exc.to_problem_details()
    logger.warning(
        "DOMAIN_EXCEPTION_RAISED",
        code=problem.code,
        title=problem.title,
        status_code=problem.status,
        instance=problem.instance,
        path=request.url.path,
    )
    return JSONResponse(
        status_code=problem.status,
        content=problem.model_dump(),
        media_type="application/problem+json",
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation failures with structured, non-leaking error responses."""
    details = [
        ErrorDetail(
            code="VALIDATION_ERROR",
            message=err.get("msg", "Invalid value"),
            field=".".join(str(loc) for loc in err.get("loc", [])),
        )
        for err in exc.errors()
    ]
    logger.warning(
        "REQUEST_VALIDATION_ERROR",
        path=request.url.path,
        error_count=len(details),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=APIErrorResponse(
            error="Validation failed for input data",
            details=details,
            request_id=request.headers.get("X-Request-ID"),
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle explicit HTTP exceptions."""
    logger.info(
        "HTTP_EXCEPTION_RAISED",
        status_code=exc.status_code,
        detail=str(exc.detail),
        path=request.url.path,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=APIErrorResponse(
            error=str(exc.detail),
            request_id=request.headers.get("X-Request-ID"),
        ).model_dump(),
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global catch-all preventing internal stack trace disclosure."""
    logger.error("UNHANDLED_EXCEPTION", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=APIErrorResponse(
            error="An internal error occurred while processing the request.",
            request_id=request.headers.get("X-Request-ID"),
        ).model_dump(),
    )


# ── Route Registration ──
# 1. Top-level diagnostic & info probes (/health, /live, /ready, /, /metrics)
app.include_router(health.router)
app.include_router(info.router)
app.include_router(metrics.router)

# 2. Versioned API routes (/v1 and /api/v1 for standard microservice routing)
app.include_router(api_router, prefix="/v1")
app.include_router(api_router, prefix="/api/v1")
