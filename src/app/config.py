"""Configuration module for Life Insurance RAG Platform.

Handles environment-driven settings (DEV vs PROD) without code duplication.
Adheres to strict typing and environment variable overrides via Pydantic Settings.
"""

from enum import StrEnum
from functools import lru_cache
from typing import Any, Literal

from pydantic import ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.security.mock_auth import DEFAULT_MOCK_ACCOUNTS, MockAccount  # noqa: E402


class Environment(StrEnum):
    """Runtime environment enum."""

    DEV = "DEV"
    PROD = "PROD"


class LLMProvider(StrEnum):
    """LLM provider selection."""

    GROQ = "groq"
    BEDROCK = "bedrock"


class EmbeddingProvider(StrEnum):
    """Vector embedding provider selection."""

    FASTEMBED = "fastembed"
    QWEN = "qwen"
    BEDROCK = "bedrock"


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env files."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # ── Application ──
    ENV: Environment = None
    APP_NAME: str = "saarthi-ai"
    APP_VERSION: str = "0.1.0"
    LOG_LEVEL: str = "INFO"
    TIMEZONE: str = "Asia/Kolkata"
    HIPAA_MODE: bool = False
    API_HOST: str = "0.0.0.0"
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # ── Provider Switching ──
    LLM_PROVIDER: LLMProvider | None = None
    EMBEDDING_PROVIDER: EmbeddingProvider | None = None

    # ── Portkey AI Gateway ──
    PORTKEY_API_KEY: str = ""
    PORTKEY_VIRTUAL_KEY: str = ""  # Virtual key for the active provider (Groq in DEV, Bedrock in PROD)
    PORTKEY_BASE_URL: str = "https://api.portkey.ai/v1"

    # ── LLM Model (single model for all tasks: routing, synthesis, grading) ──
    LLM_MODEL: str | None = None

    # ── Embedding Models (Qwen3-Embedding-8B in DEV, Titan V2 in PROD) ──
    EMBEDDING_MODEL: str | None = None
    EMBEDDING_DIMENSIONS: int | None = None

    # ── AWS Core: S3, DynamoDB, SQS, Secrets Manager & ElastiCache ──
    AWS_REGION: str = "ap-southeast-2"  # Sydney (authorized region under AWS SCP)
    S3_BUCKET_NAME: str | None = None  # saarthi-dev-uploads in DEV, saarthi-prod-uploads in PROD
    SECRET_NAME: str | None = None  # saarthi-ai/dev/secrets in DEV, saarthi-ai/prod/secrets in PROD
    DYNAMODB_TABLE_PREFIX: str = "saarthi-ai"  # same for both DEV and PROD
    DYNAMODB_TABLE_DOCUMENTS: str = "saarthi-ai-documents"  # same for both DEV and PROD
    DYNAMODB_TABLE_AUDIT: str = "saarthi-ai-audit-log"  # same for both DEV and PROD
    DYNAMODB_TABLE_SESSIONS: str = "saarthi-ai-chat-sessions"  # same for both DEV and PROD
    SQS_INGEST_QUEUE_URL: str | None = None
    SQS_INGEST_QUEUE_NAME: str = "saarthi-ai-ingest-queue"  # same for both DEV and PROD
    AWS_SECRETS_ENABLED: bool = True

    # ── LangGraph Checkpointer (State Persistence) ──
    CHECKPOINTER_BACKEND: str = "auto"  # "auto", "dynamodb", "redis", "memory"
    CHECKPOINT_TTL_DAYS: int = 30  # Auto-purge expired session checkpoints via DynamoDB TTL

    # ── Qdrant Vector Store ──
    QDRANT_URL: str | None = None  # Full URL for Qdrant Cloud (e.g. https://xyz.cloud.qdrant.io)
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "saarthi-ai"  # hardcoded: same for DEV and PROD; only URL and API key vary per ENV
    # Qdrant aliases (§3 — knowledge-base routing)
    QDRANT_ORG_KB_ALIAS: str = "saarthi_org_kb"
    QDRANT_USER_DOCS_ALIAS: str = "saarthi_user_docs"

    # ── Redis Semantic Cache ──
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL_SECONDS: int = 86400  # 24 hours
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = 0.95  # min cosine similarity for cache hit

    # ── Observability: Logfire & LangSmith ──
    LANGSMITH_TRACING: bool = True
    LANGSMITH_PROJECT: str = "saarthi-ai-dev"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY: str = ""

    LOGFIRE_TOKEN: str = ""
    LOGFIRE_ENVIRONMENT: str | None = None
    LOGFIRE_SERVICE_NAME: str = "saarthi-ai"
    LOGFIRE_SEND_TO_LOGFIRE: bool = True

    # ── Authentication & Rate Limiting ──
    API_KEY_HEADER_NAME: str = "X-API-Key"
    API_KEY_HASH_SALT: str = "dev-salt-change-me"
    DEV_API_KEY: str = "dev-test-key-12345"
    RATE_LIMIT_PER_MINUTE: int = 60

    # ── JWT Authentication ──
    JWT_SECRET_KEY: str = "saarthi-ai-enterprise-jwt-signing-secret-key-32chars"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24

    # ── Mock Auth (§8.5) ──
    # AUTH_MODE: "google" uses real OIDC; "mock" shows the fixed account chooser.
    AUTH_MODE: Literal["google", "mock"] = "mock"
    ALLOW_MOCK_IN_PROD: bool = False  # set True only for a deliberate, time-boxed demo
    # Overridable list of mock accounts (never persisted to DB; only active in mock mode).
    MOCK_ACCOUNTS: list[MockAccount] = DEFAULT_MOCK_ACCOUNTS
    SESSION_ABSOLUTE_HOURS: int = 8  # mock sessions use a shorter TTL (e.g. 4h can be set via env)

    # ── Insurance Domain & Guardrails ──
    GUARDRAILS_ENABLED: bool = True
    GUARDRAILS_STRICT_MODE: bool = False
    MIN_FAITHFULNESS_SCORE: float = 0.7

    @field_validator("ENV", mode="before")
    @classmethod
    def validate_environment(cls, v: Any) -> Environment:
        """Validate environment name. Must be DEV or PROD, else raises ValueError."""
        if isinstance(v, str):
            v_upper = v.upper()
            if v_upper in ("DEV", "PROD"):
                return Environment(v_upper)
            raise ValueError(f"Invalid environment name: '{v}'. Must be 'DEV' or 'PROD'.")
        if isinstance(v, Environment):
            return v
        raise ValueError(f"Environment name is required and must be 'DEV' or 'PROD', got {v}")

    @field_validator("S3_BUCKET_NAME", mode="before")
    @classmethod
    def set_s3_bucket_name(cls, v: Any, info: ValidationInfo) -> str:
        """Sets S3_BUCKET_NAME based on ENV variable."""
        if v:
            return v
        env = info.data.get("ENV")

        bucket_map = {
            Environment.DEV: "saarthi-dev-uploads",
            Environment.PROD: "saarthi-prod-uploads",
        }

        if not env or env not in bucket_map:
            raise ValueError(f"Invalid environment for S3 bucket name: {env}")
        return bucket_map[env]

    @field_validator("SECRET_NAME", mode="before")
    @classmethod
    def set_secret_name(cls, v: Any, info: ValidationInfo) -> str:
        """Sets SECRET_NAME for AWS Secrets Manager based on ENV variable."""
        if v:
            return v
        env = info.data.get("ENV")

        secret_map = {
            Environment.DEV: "saarthi-ai/dev/secrets",
            Environment.PROD: "saarthi-ai/prod/secrets",
        }

        if not env or env not in secret_map:
            raise ValueError(f"Invalid environment for Secrets Manager: {env}")
        return secret_map[env]

    @field_validator("SQS_INGEST_QUEUE_NAME", mode="before")
    @classmethod
    def set_sqs_queue_name(cls, v: Any) -> str:
        """Sets SQS queue name (same for both DEV and PROD unless explicitly overridden)."""
        return v or "saarthi-ai-ingest-queue"

    @field_validator("DYNAMODB_TABLE_PREFIX", mode="before")
    @classmethod
    def set_dynamodb_prefix(cls, v: Any) -> str:
        """Sets DynamoDB table prefix (same for both DEV and PROD unless explicitly overridden)."""
        return v or "saarthi-ai"

    @field_validator("DYNAMODB_TABLE_DOCUMENTS", mode="before")
    @classmethod
    def set_dynamodb_table_documents(cls, v: Any, info: ValidationInfo) -> str:
        """Sets DynamoDB documents table (same for both DEV and PROD unless explicitly overridden)."""
        if v:
            return v
        prefix = info.data.get("DYNAMODB_TABLE_PREFIX") or "saarthi-ai"
        return f"{prefix}-documents"

    @field_validator("DYNAMODB_TABLE_AUDIT", mode="before")
    @classmethod
    def set_dynamodb_table_audit(cls, v: Any, info: ValidationInfo) -> str:
        """Sets DynamoDB audit log table (same for both DEV and PROD unless explicitly overridden)."""
        if v:
            return v
        prefix = info.data.get("DYNAMODB_TABLE_PREFIX") or "saarthi-ai"
        return f"{prefix}-audit-log"

    @field_validator("DYNAMODB_TABLE_SESSIONS", mode="before")
    @classmethod
    def set_dynamodb_table_sessions(cls, v: Any, info: ValidationInfo) -> str:
        """Sets DynamoDB chat sessions table (same for both DEV and PROD unless explicitly overridden)."""
        if v:
            return v
        prefix = info.data.get("DYNAMODB_TABLE_PREFIX") or "saarthi-ai"
        return f"{prefix}-chat-sessions"

    @field_validator("LLM_PROVIDER", mode="before")
    @classmethod
    def set_llm_provider(cls, v: Any, info: ValidationInfo) -> LLMProvider:
        """Sets LLM_PROVIDER based on ENV variable (groq in DEV, bedrock in PROD)."""
        if v:
            return v if isinstance(v, LLMProvider) else LLMProvider(v)
        env = info.data.get("ENV")
        return LLMProvider.GROQ if env == Environment.DEV else LLMProvider.BEDROCK

    @field_validator("EMBEDDING_PROVIDER", mode="before")
    @classmethod
    def set_embedding_provider(cls, v: Any, info: ValidationInfo) -> EmbeddingProvider:
        """Sets EMBEDDING_PROVIDER based on ENV variable (qwen in DEV, bedrock in PROD)."""
        if v:
            return v if isinstance(v, EmbeddingProvider) else EmbeddingProvider(v)
        env = info.data.get("ENV")
        return EmbeddingProvider.QWEN if env == Environment.DEV else EmbeddingProvider.BEDROCK

    @field_validator("LLM_MODEL", mode="before")
    @classmethod
    def set_llm_model(cls, v: Any, info: ValidationInfo) -> str:
        """Sets LLM_MODEL based on ENV variable (or allows Portkey remote config)."""
        if v:
            return v
        env = info.data.get("ENV")
        return "openai/gpt-oss-120b" if env == Environment.DEV else "anthropic.claude-sonnet-4-20250514-v1:0"

    @field_validator("EMBEDDING_MODEL", mode="before")
    @classmethod
    def set_embedding_model(cls, v: Any, info: ValidationInfo) -> str:
        """Sets EMBEDDING_MODEL based on ENV variable (Qwen in DEV, Titan V2 in PROD)."""
        if v:
            return v
        env = info.data.get("ENV")
        return "Qwen/Qwen3-Embedding-8B" if env == Environment.DEV else "amazon.titan-embed-text-v2:0"

    @field_validator("EMBEDDING_DIMENSIONS", mode="before")
    @classmethod
    def set_embedding_dimensions(cls, v: Any, info: ValidationInfo) -> int:
        """Sets EMBEDDING_DIMENSIONS based on ENV (4096 in DEV, 1024 in PROD)."""
        if v:
            return int(v)
        env = info.data.get("ENV")
        return 4096 if env == Environment.DEV else 1024

    @field_validator("LANGSMITH_PROJECT", mode="before")
    @classmethod
    def set_langsmith_project(cls, v: Any, info: ValidationInfo) -> str:
        """Sets LANGSMITH_PROJECT based on ENV (saarthi-ai-dev in DEV, saarthi-ai-prod in PROD)."""
        if v:
            return v
        env = info.data.get("ENV")
        return "saarthi-ai-dev" if env == Environment.DEV else "saarthi-ai-prod"

    @field_validator("LOGFIRE_ENVIRONMENT", mode="before")
    @classmethod
    def set_logfire_environment(cls, v: Any, info: ValidationInfo) -> str | None:
        """Allow LOGFIRE_ENVIRONMENT to be explicitly set or None (matching enterprise-rag)."""
        return v if v else None

    @model_validator(mode="after")
    def _mock_guard(self) -> "Settings":
        """Prevent AUTH_MODE=mock from accidentally reaching production."""
        if (
            self.AUTH_MODE == "mock"
            and self.ENV == Environment.PROD
            and not self.ALLOW_MOCK_IN_PROD
        ):
            raise ValueError(
                "AUTH_MODE=mock is not permitted in PROD unless ALLOW_MOCK_IN_PROD=true. "
                "Set AUTH_MODE=google for production deployments."
            )
        return self


_secrets_hydrated: bool = False


def _hydrate_aws_secrets() -> None:
    """Hydrate environment variables from AWS Secrets Manager if credentials are present."""
    global _secrets_hydrated
    if _secrets_hydrated:
        return
    _secrets_hydrated = True

    import os
    try:
        from app.core.aws.session import has_aws_credentials
        if has_aws_credentials():
            from app.core.aws.secrets_manager import fetch_and_apply_secrets
            env_name = os.getenv("ENV", "DEV").upper()
            secret_name = os.getenv("SECRET_NAME") or (
                "saarthi-ai/dev/secrets" if env_name == "DEV" else "saarthi-ai/prod/secrets"
            )
            region = os.getenv("AWS_REGION", "ap-southeast-2")
            fetch_and_apply_secrets(secret_name=secret_name, region_name=region)
    except Exception:
        pass


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance with AWS Secrets Manager support."""
    _hydrate_aws_secrets()
    return Settings()


# Export default global settings
settings = get_settings()
