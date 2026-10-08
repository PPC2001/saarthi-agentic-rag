"""LLM client factory via Portkey AI Gateway.

Provides a single ChatOpenAI interface regardless of environment:
- DEV:  Groq Llama 3.3 70B (configurable via LLM_MODEL)
- PROD: AWS Bedrock Claude Sonnet (configurable via LLM_MODEL)

One model handles all tasks: intent routing, answer synthesis, grading.
"""

from langchain_openai import ChatOpenAI
from portkey_ai import createHeaders

from app.config import Settings
from app.observability.logging import get_logger

logger = get_logger("llm_gateway")


def create_llm(settings: Settings) -> ChatOpenAI:
    """Create the single LLM client used for all graph tasks.

    DEV:  Groq Llama 3.3 70B  (fast, cheap, great for internal tooling)
    PROD: Bedrock Claude Sonnet (deep reasoning, compliance-grade)

    The same model handles intent routing, answer synthesis, and grading.
    This removes the fast/smart split — unnecessary complexity for an internal app.
    """
    if not settings.PORTKEY_API_KEY:
        raise ValueError("PORTKEY_API_KEY is not set")
    if not settings.PORTKEY_VIRTUAL_KEY:
        raise ValueError("PORTKEY_VIRTUAL_KEY is not set.")

    model = settings.LLM_MODEL
    env_str = settings.ENV.value if hasattr(settings.ENV, "value") else (str(settings.ENV) if settings.ENV else "DEV")
    logger.info("Creating LLM client", model=settings.LLM_MODEL, env=env_str)

    headers = createHeaders(
        api_key=settings.PORTKEY_API_KEY,
        virtual_key=settings.PORTKEY_VIRTUAL_KEY,
        metadata={
            "environment": env_str,
            "app_version": str(settings.APP_VERSION),
        },
    )
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=settings.PORTKEY_BASE_URL,
        default_headers=headers,
        model=model,
        temperature=0.0,
        timeout=60,
        max_retries=0,
    )
