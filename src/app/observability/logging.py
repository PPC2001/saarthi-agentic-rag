"""Structured JSON logging with request correlation and PHI masking.

Ensures compliance with HIPAA and insurance data security policies:
- Masks sensitive PHI (SSN, emails, phone numbers, credit cards)
- Formats logs as JSON in PROD and clean console in DEV
- Binds request correlation IDs to every log line
"""

import logging
import re
import sys
from contextvars import ContextVar
from datetime import datetime
from typing import Any

import pytz
import structlog

# Context variable for request correlation ID
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="")
user_id_ctx: ContextVar[str] = ContextVar("user_id", default="anonymous")

# PHI/PII Masking patterns (US & Indian identifiers)
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
US_PHONE_PATTERN = re.compile(
    r"\b(?:\+?1[-. ]?)?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b"
)
IN_PHONE_PATTERN = re.compile(r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b")
AADHAAR_PATTERN = re.compile(r"\b[2-9]{1}[0-9]{3}[ -]?[0-9]{4}[ -]?[0-9]{4}(?![ -]?[0-9])\b")
PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b")


def mask_phi(text: str) -> str:
    """Mask common PHI/PII data in log messages, including Indian Aadhaar and PAN."""
    text = CREDIT_CARD_PATTERN.sub("****-****-****-****", text)
    text = SSN_PATTERN.sub("***-**-****", text)
    text = EMAIL_PATTERN.sub("***@***.***", text)
    text = US_PHONE_PATTERN.sub("***-***-****", text)
    text = IN_PHONE_PATTERN.sub("+91-XXXXX-****", text)
    text = AADHAAR_PATTERN.sub("XXXX-XXXX-****", text)
    text = PAN_PATTERN.sub("*********", text)
    return text


def phi_masking_processor(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor that masks PHI in the log event message and attributes."""
    if "event" in event_dict and isinstance(event_dict["event"], str):
        event_dict["event"] = mask_phi(event_dict["event"])

    for key, value in list(event_dict.items()):
        if isinstance(value, str):
            event_dict[key] = mask_phi(value)

    return event_dict


def correlation_id_processor(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Inject correlation ID and user ID from contextvars."""
    corr_id = correlation_id_ctx.get()
    if corr_id:
        event_dict["request_id"] = corr_id

    uid = user_id_ctx.get()
    if uid:
        event_dict["user_id"] = uid

    return event_dict


def pytz_timestamp_processor(
    tz_name: str = "Asia/Kolkata",
) -> structlog.types.Processor:
    """Structlog processor that injects an ISO timestamp localized using pytz."""
    try:
        tz = pytz.timezone(tz_name)
    except Exception:
        tz = pytz.timezone("Asia/Kolkata")

    def processor(
        logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
    ) -> dict[str, Any]:
        event_dict["timestamp"] = datetime.now(tz).isoformat()
        return event_dict

    return processor


def get_localized_now(tz_name: str = "Asia/Kolkata") -> datetime:
    """Return current datetime localized using pytz."""
    try:
        tz = pytz.timezone(tz_name)
    except Exception:
        tz = pytz.timezone("Asia/Kolkata")
    return datetime.now(tz)


def get_localized_timestamp(tz_name: str = "Asia/Kolkata") -> str:
    """Return ISO 8601 formatted timestamp localized using pytz."""
    return get_localized_now(tz_name).isoformat()


def setup_logging(
    is_dev: bool = True,
    log_level: str = "INFO",
    timezone: str = "Asia/Kolkata",
) -> None:
    """Configure structlog and standard logging based on environment and timezone."""
    logfire_processors: list[structlog.types.Processor] = []
    try:
        import logfire

        logfire_processors.append(logfire.StructlogProcessor())
    except Exception:
        pass

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        pytz_timestamp_processor(timezone),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        correlation_id_processor,
        phi_masking_processor,
        *logfire_processors,
    ]

    level = getattr(logging, log_level.upper(), logging.INFO)

    if is_dev:
        # Development: pretty console printing
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # Production: JSON output for CloudWatch / OpenSearch
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    try:
        import logfire

        root_logger.addHandler(logfire.LogfireLoggingHandler())
    except Exception:
        pass
    root_logger.setLevel(level)

    # Quiet down chatty external loggers
    for logger_name in ("httpx", "httpcore", "botocore", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger."""
    return structlog.get_logger(name)


def log_compliance_audit(
    user_id: str,
    session_id: str | None,
    query: str,
    sources: list[dict[str, Any]],
    answer: str,
    latency_ms: float,
    model_used: str,
    query_type: str,
) -> None:
    """Log structured immutable compliance audit record for statutory and IRDAI regulatory scrutiny."""
    audit_logger = structlog.get_logger("compliance_audit")
    sanitized_query = mask_phi(query)
    preview_answer = mask_phi(answer[:300] + ("..." if len(answer) > 300 else ""))

    clean_sources = []
    for s in sources:
        clean_sources.append(
            {
                "title": s.get("document_title") or s.get("title", "Policy Doc"),
                "uin": s.get("uin"),
                "section": s.get("section"),
                "url": s.get("url"),
                "source_type": s.get("source_type", "contract"),
            }
        )

    audit_logger.info(
        "STATUTORY_COMPLIANCE_AUDIT_LOG",
        user_id=user_id,
        session_id=session_id or "default",
        sanitized_query=sanitized_query,
        sources_count=len(sources),
        sources=clean_sources,
        answer_preview=preview_answer,
        latency_ms=round(latency_ms, 2),
        model_used=model_used,
        query_type=query_type,
    )
