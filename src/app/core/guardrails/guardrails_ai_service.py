"""All-in-One Guardrails AI Integration for Saarthi AI - Life Insurance Assistant.

Implements unified Guardrails AI architecture for:
1. Stopping Jailbreaks & Prompt Injections (PromptInjectionGuardrail)
2. Preventing PII Leaks (PresidioPIIGuardrail via Microsoft Presidio)
3. Keeping the Bot On-Topic (InsuranceTopicGuardrail)
4. Enforcing JSON Schemas & Output Compliance (Guard.for_pydantic & StatutoryComplianceGuardrail)
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Any

from guardrails import Guard
from guardrails.validator_base import FailResult, PassResult, Validator, register_validator
from pydantic import BaseModel

from app.core.guardrails.input_guardrails import (
    OFF_TOPIC_GUARDRAIL_RESPONSE,
    check_off_topic,
    check_prompt_injection,
    mask_pii,
)
from app.core.guardrails.output_guardrails import enforce_output_guardrails
from app.core.retrieval.classifier import (
    COMBINED_GREETING_HELP_REGEX,
    GRATITUDE_REGEX,
    GRATITUDE_RESPONSE,
    GREETING_REGEX,
    GREETING_RESPONSE,
    HELP_IDENTITY_REGEX,
    QueryIntent,
)
from app.observability.logging import get_logger

# Silence external OTLP telemetry to ensure offline, deterministic execution
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["GUARDRAILS_TELEMETRY"] = "false"
os.environ["GUARDRAILS_ANONYMOUS_TELEMETRY"] = "false"

# Filter benign validator event loop fallback warnings when running under sync/thread-pool executors
warnings.filterwarnings("ignore", message=".*Could not obtain an event loop.*", category=UserWarning)

logger = get_logger("guardrails_ai_service")


# ── 1. Custom Guardrails AI Validators ──


@register_validator(name="saarthi/prompt_injection", data_type="string")
class PromptInjectionGuardrail(Validator):
    """Guardrails AI validator to intercept jailbreaks and instruction override attacks."""

    rail_alias = "saarthi/prompt_injection"

    def __init__(self, on_fail: str = "fix", **kwargs):
        super().__init__(on_fail=on_fail, **kwargs)

    def validate(self, value: Any, metadata: dict | None = None) -> PassResult | FailResult:
        query_str = str(value)
        is_injection, reason = check_prompt_injection(query_str)
        if is_injection:
            logger.warning("GUARDRAILS_PROMPT_INJECTION_DETECTED", reason=reason)
            return FailResult(
                error_message=reason or "Prompt injection or jailbreak attempt detected.",
                fix_value="[SECURITY_BLOCKED]",
            )
        return PassResult()


@register_validator(name="saarthi/presidio_pii", data_type="string")
class PresidioPIIGuardrail(Validator):
    """Guardrails AI validator wrapping Microsoft Presidio for Indian PII detection & masking."""

    rail_alias = "saarthi/presidio_pii"

    def __init__(self, on_fail: str = "fix", **kwargs):
        super().__init__(on_fail=on_fail, **kwargs)

    def validate(self, value: Any, metadata: dict | None = None) -> PassResult | FailResult:
        query_str = str(value)
        if query_str == "[SECURITY_BLOCKED]":
            return PassResult()
        sanitized_text, redactions = mask_pii(query_str)
        if redactions:
            logger.info(
                "GUARDRAILS_PRESIDIO_PII_MASKED", count=len(redactions)
            )
            return FailResult(
                error_message="PII detected and anonymized",
                fix_value=sanitized_text,
            )
        return PassResult()


@register_validator(name="saarthi/insurance_topic", data_type="string")
class InsuranceTopicGuardrail(Validator):
    """Guardrails AI validator to keep conversations strictly on Indian Life Insurance topics."""

    rail_alias = "saarthi/insurance_topic"

    def __init__(self, on_fail: str = "fix", **kwargs):
        super().__init__(on_fail=on_fail, **kwargs)

    def validate(self, value: Any, metadata: dict | None = None) -> PassResult | FailResult:
        query_str = str(value)
        if query_str == "[SECURITY_BLOCKED]":
            return PassResult()

        meta = metadata or {}
        has_attachment = meta.get("has_attachment", False)

        # Allow conversational greetings, identity, and gratitude without off-topic block
        if (
            GREETING_REGEX.match(query_str)
            or COMBINED_GREETING_HELP_REGEX.match(query_str)
            or HELP_IDENTITY_REGEX.match(query_str)
            or GRATITUDE_REGEX.match(query_str)
        ):
            return PassResult()

        is_off_topic, reason = check_off_topic(query_str, has_attachment=has_attachment)
        if is_off_topic:
            logger.warning("GUARDRAILS_OFF_TOPIC_BLOCKED", reason=reason)
            return FailResult(
                error_message=reason or "Off-topic query outside Indian Life Insurance domain.",
                fix_value=OFF_TOPIC_GUARDRAIL_RESPONSE,
            )
        return PassResult()


@register_validator(name="saarthi/statutory_compliance", data_type="string")
class StatutoryComplianceGuardrail(Validator):
    """Guardrails AI output validator for IRDAI statutory notices and disclaimers."""

    rail_alias = "saarthi/statutory_compliance"

    def __init__(self, on_fail: str = "fix", **kwargs):
        super().__init__(on_fail=on_fail, **kwargs)

    def validate(self, value: Any, metadata: dict | None = None) -> PassResult | FailResult:
        raw_output = str(value)
        meta = metadata or {}
        query = meta.get("query", "")
        guarded_output = enforce_output_guardrails(raw_output, query=query)
        if guarded_output != raw_output:
            return FailResult(
                error_message="Output compliance notices appended",
                fix_value=guarded_output,
            )
        return PassResult()


# ── 2. Guardrails Manager & Structured Pipeline ──


@dataclass
class GuardrailValidationResult:
    """Structured result from Guardrails AI input evaluation."""

    is_valid: bool
    sanitized_query: str
    query_type: str
    blocked_reason: str | None = None
    precomputed_answer: str | None = None
    pii_redactions: list[dict] = field(default_factory=list)


class SaarthiGuardrailsManager:
    """Unified Guardrails AI Manager for Saarthi AI."""

    def __init__(self):
        # 1. Initialize Input Guard pipeline
        self._input_guard = Guard().use(
            PromptInjectionGuardrail(on_fail="fix"),
            PresidioPIIGuardrail(on_fail="fix"),
            InsuranceTopicGuardrail(on_fail="fix"),
        )

        # 2. Initialize Output Guard pipeline
        self._output_guard = Guard().use(StatutoryComplianceGuardrail(on_fail="fix"))

    def validate_input(
        self,
        query: str,
        has_attachment: bool = False,
    ) -> GuardrailValidationResult:
        """Run all input guardrails (Jailbreak, Presidio PII, On-Topic) via Guardrails AI.

        Returns:
            GuardrailValidationResult with sanitized query and routing instructions.
        """
        clean_q = query.strip()

        # 1. Check Fast Conversational Intents
        if GREETING_REGEX.match(clean_q) or COMBINED_GREETING_HELP_REGEX.match(clean_q):
            return GuardrailValidationResult(
                is_valid=True,
                sanitized_query=clean_q,
                query_type=QueryIntent.GREETING.value,
                precomputed_answer=GREETING_RESPONSE,
            )
        if HELP_IDENTITY_REGEX.match(clean_q):
            return GuardrailValidationResult(
                is_valid=True,
                sanitized_query=clean_q,
                query_type=QueryIntent.HELP.value,
                precomputed_answer=GREETING_RESPONSE,
            )
        if GRATITUDE_REGEX.match(clean_q):
            return GuardrailValidationResult(
                is_valid=True,
                sanitized_query=clean_q,
                query_type=QueryIntent.GRATITUDE.value,
                precomputed_answer=GRATITUDE_RESPONSE,
            )

        # 2. Execute Chained Guardrails AI Input Guard
        try:
            validation_res = self._input_guard.validate(
                clean_q,
                metadata={"has_attachment": has_attachment},
            )
            validated_text = str(validation_res.validated_output or clean_q)

            # Check if Security Blocked (Jailbreak)
            if validated_text == "[SECURITY_BLOCKED]":
                logger.warning("GUARDRAILS_SECURITY_INJECTION_BLOCKED")
                return GuardrailValidationResult(
                    is_valid=False,
                    sanitized_query=clean_q,
                    query_type="security_blocked",
                    blocked_reason="Potential instruction override or prompt injection attempt detected.",
                    precomputed_answer=(
                        "⚠️ **Security Notice**: Your query triggered an automated security guardrail "
                        "(Instruction override detected). Please submit a valid insurance inquiry."
                    ),
                )

            # Check if Off-Topic Blocked
            if validated_text == OFF_TOPIC_GUARDRAIL_RESPONSE:
                logger.info("GUARDRAILS_OFF_TOPIC_QUERY_BLOCKED")
                return GuardrailValidationResult(
                    is_valid=False,
                    sanitized_query=clean_q,
                    query_type=QueryIntent.OFF_TOPIC.value,
                    blocked_reason="Query is outside the scope of Indian Life Insurance.",
                    precomputed_answer=OFF_TOPIC_GUARDRAIL_RESPONSE,
                )

            # Determine query type for in-scope queries
            query_type = (
                QueryIntent.ATTACHMENT_RAG.value
                if has_attachment
                else QueryIntent.POLICY_QUERY.value
            )

            return GuardrailValidationResult(
                is_valid=True,
                sanitized_query=validated_text,
                query_type=query_type,
            )

        except Exception as e:
            logger.warning("GUARDRAILS_AI_INPUT_FALLBACK", error=str(e))
            sanitized, _ = mask_pii(clean_q)
            return GuardrailValidationResult(
                is_valid=True,
                sanitized_query=sanitized,
                query_type=QueryIntent.POLICY_QUERY.value,
            )

    def validate_output(self, raw_answer: str, query: str) -> str:
        """Run output compliance guardrails via Guardrails AI."""
        try:
            res = self._output_guard.validate(
                raw_answer,
                metadata={"query": query},
            )
            return str(res.validated_output or raw_answer)
        except Exception as e:
            logger.warning("GUARDRAILS_AI_OUTPUT_FALLBACK", error=str(e))
            return enforce_output_guardrails(raw_answer, query=query)

    def create_json_schema_guard(self, pydantic_model: type[BaseModel]) -> Guard:
        """Create a Guardrails AI instance strictly enforcing the given Pydantic JSON schema."""
        return Guard().for_pydantic(output_class=pydantic_model)


# Singleton instance for application lifecycle
guardrails_manager = SaarthiGuardrailsManager()
