"""Unit tests for configuration and provider switching settings."""

from app.config import (
    EmbeddingProvider,
    Environment,
    LLMProvider,
    Settings,
)


def test_default_config_is_dev():
    """Verify default environment is DEV with fast models and local embeddings."""
    settings = Settings(
        ENV=Environment.DEV,
        LLM_PROVIDER=LLMProvider.GROQ,
        EMBEDDING_PROVIDER=EmbeddingProvider.FASTEMBED,
        EMBEDDING_DIMENSIONS=384,
    )
    assert settings.ENV == Environment.DEV
    assert settings.ENV.value == "DEV"
    assert settings.LLM_PROVIDER == LLMProvider.GROQ
    assert settings.EMBEDDING_PROVIDER == EmbeddingProvider.FASTEMBED
    assert settings.EMBEDDING_DIMENSIONS == 384
    assert settings.HIPAA_MODE is False


def test_qwen_config_settings():
    """Verify Qwen3-Embedding-8B configuration values."""
    settings = Settings(
        EMBEDDING_PROVIDER=EmbeddingProvider.QWEN,
        EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B",
        EMBEDDING_DIMENSIONS=4096,
    )
    assert settings.EMBEDDING_PROVIDER == EmbeddingProvider.QWEN
    assert settings.EMBEDDING_MODEL == "Qwen/Qwen3-Embedding-8B"
    assert settings.EMBEDDING_DIMENSIONS == 4096


def test_prod_config_settings():
    """Verify PROD configuration values."""
    settings = Settings(
        ENV=Environment.PROD,
        AUTH_MODE="google",
        LLM_PROVIDER=LLMProvider.BEDROCK,
        EMBEDDING_PROVIDER=EmbeddingProvider.BEDROCK,
        EMBEDDING_DIMENSIONS=1024,
        HIPAA_MODE=True,
    )
    assert settings.ENV == Environment.PROD
    assert settings.ENV.value == "PROD"
    assert settings.LLM_PROVIDER == LLMProvider.BEDROCK
    assert settings.EMBEDDING_PROVIDER == EmbeddingProvider.BEDROCK
    assert settings.EMBEDDING_DIMENSIONS == 1024
    assert settings.HIPAA_MODE is True


def test_langsmith_config_settings():
    """Verify LangSmith configuration values."""
    settings = Settings(
        ENV=Environment.DEV,
        LANGSMITH_TRACING=True,
        LANGSMITH_API_KEY="test-langsmith-key",
    )
    assert settings.LANGSMITH_TRACING is True
    assert settings.LANGSMITH_PROJECT == "saarthi-ai-dev"
    assert settings.LANGSMITH_ENDPOINT == "https://api.smith.langchain.com"
    assert settings.LANGSMITH_API_KEY == "test-langsmith-key"

