"""Pydantic Logfire setup and auto-instrumentation.

Provides Tier-1 application observability:
- Request lifecycle tracing (FastAPI endpoints)
- Pydantic schema validation telemetry
- External HTTP and LLM Gateway tracing (httpx & OpenAI / Portkey)

Common pitfalls:
- Token in .env must match the active Logfire project token (Settings → API Tokens in UI).
- setup_logfire(None, settings) should be called AFTER setup_logging() so that
  logfire.StructlogProcessor is registered into an already-configured structlog chain.
- Call logfire.force_flush() on shutdown to drain the async OTEL exporter buffer.
"""

import json
import os
from pathlib import Path
from typing import Any

from app.config import Settings
from app.observability.logging import get_logger

logger = get_logger(__name__)

_logfire_configured = False


def _load_credentials_token() -> str | None:
    """Read the write token from .logfire/logfire_credentials.json if present.

    This file is created by `logfire auth` / `logfire projects use` and contains
    the canonical token for the local project. We use it as a fallback when
    LOGFIRE_TOKEN is not explicitly set in the environment.
    """
    candidates = [
        Path(os.getcwd()) / ".logfire" / "logfire_credentials.json",
        Path(__file__).resolve().parents[5] / ".logfire" / "logfire_credentials.json",
    ]
    for creds_path in candidates:
        if creds_path.exists():
            try:
                data = json.loads(creds_path.read_text())
                return data.get("token")
            except Exception:
                pass
    return None


def setup_logfire(app: Any, settings: Settings) -> None:
    """Initialize Logfire and instrument FastAPI and dependencies.

    IMPORTANT — call order in main.py:
        1. setup_logging()          ← must come first so structlog chain exists
        2. setup_logfire(None, …)   ← configures OTEL exporter + adds StructlogProcessor
        3. app = FastAPI(…)
        4. setup_logfire(app, …)    ← instruments FastAPI (second call is a no-op for configure)
    """
    global _logfire_configured
    try:
        import logfire

        if not _logfire_configured:
            env_token: str = (settings.LOGFIRE_TOKEN or "").strip()
            creds_token: str | None = _load_credentials_token()

            # Prefer the local credentials file (.logfire/logfire_credentials.json)
            # because that is the explicit project the user selected with `logfire projects use`.
            if env_token and creds_token and env_token != creds_token:
                logger.warning(
                    "Logfire token mismatch: LOGFIRE_TOKEN in environment differs from "
                    ".logfire/logfire_credentials.json. Preferring local project credentials token. "
                    "Run `logfire projects use` or update LOGFIRE_TOKEN to sync.",
                    env_token_prefix=env_token[:16] + "...",
                    creds_token_prefix=creds_token[:16] + "...",
                )

            active_token = creds_token or env_token or ""

            # Configure Logfire: send traces when token is present and sending is enabled
            send_to_logfire = settings.LOGFIRE_SEND_TO_LOGFIRE and bool(active_token)

            config_kwargs: dict[str, Any] = {
                "token": active_token,
                "send_to_logfire": send_to_logfire,
                "inspect_arguments": False,
            }
            if settings.LOGFIRE_SERVICE_NAME:
                config_kwargs["service_name"] = settings.LOGFIRE_SERVICE_NAME
            if settings.LOGFIRE_ENVIRONMENT:
                config_kwargs["environment"] = settings.LOGFIRE_ENVIRONMENT

            logfire.configure(**config_kwargs)

            # Auto-instrument HTTPX
            logfire.instrument_httpx()

            # Auto-instrument Redis
            try:
                logfire.instrument_redis()
            except Exception as re_err:
                logger.debug("Logfire Redis instrumentation notice", error=str(re_err))

            # Auto-instrument Pydantic model validations
            try:
                logfire.instrument_pydantic()
            except Exception as pe:
                logger.debug("Logfire Pydantic instrumentation notice", error=str(pe))

            # Auto-instrument OpenAI / Portkey LLM calls
            try:
                logfire.instrument_openai()
            except Exception as oe:
                logger.debug("Logfire OpenAI instrumentation notice", error=str(oe))

            # Flush any buffered spans from the configure phase so they appear
            # immediately in the Logfire UI (important for short-lived dev processes).
            logfire.force_flush()

            logger.info(
                "Logfire initialized successfully",
                service=settings.LOGFIRE_SERVICE_NAME,
                environment=settings.LOGFIRE_ENVIRONMENT,
                send_to_logfire=send_to_logfire,
                token_source="env" if env_token else "credentials_file",
            )
            _logfire_configured = True

        # Auto-instrument FastAPI when app is provided
        if app is not None:
            logfire.instrument_fastapi(app, capture_headers=True)

    except Exception as e:
        # Re-raise as a warning-level log but don't crash the app.
        # IMPORTANT: Print the full exception so it's visible in startup logs.
        logger.warning(
            "Logfire instrumentation skipped or failed",
            error=str(e),
            exc_info=True,
        )
