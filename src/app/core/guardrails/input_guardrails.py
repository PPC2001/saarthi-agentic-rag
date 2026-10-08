"""Input Guardrails for Indian Life Insurance Assistant with Microsoft Presidio.

Enforces:
1. Microsoft Presidio PII/PHI Detection & Anonymization:
   - Built-in Recognizers: EMAIL_ADDRESS, PHONE_NUMBER, PERSON, US_BANK_NUMBER, IBAN_CODE, CREDIT_CARD
   - Custom Recognizers: IN_AADHAAR (12-digit Indian National ID), IN_PAN (Permanent Account Number), IN_BANK_ACCOUNT
2. Prompt Injection & Jailbreak Defense
"""

import re

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

from app.observability.logging import get_logger

logger = get_logger("input_guardrails")

# Singleton engines for Microsoft Presidio
_ANALYZER: AnalyzerEngine | None = None
_ANONYMIZER: AnonymizerEngine | None = None

# Regex patterns for fallback
AADHAAR_REGEX = re.compile(r"\b[2-9]{1}[0-9]{3}[ -]?[0-9]{4}[ -]?[0-9]{4}\b")
PAN_REGEX = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b")
PHONE_REGEX = re.compile(r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b")
BANK_ACC_REGEX = re.compile(
    r"(?i)\b(?:a/c|acc|account|bank)\s*(?:no\.?|num\.?|number)?\s*[:#-]?\s*(\d{9,18})\b"
)
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Prompt Injection heuristics
INJECTION_PATTERNS = [
    re.compile(r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|directives|prompts)\b"),
    re.compile(r"(?i)\byou\s+are\s+now\s+(unrestricted|in\s+dan\s+mode|jailbroken)\b"),
    re.compile(
        r"(?i)\b(reveal|print|show|output)\s+(out\s+)?(your\s+|the\s+)?(internal\s+|hidden\s+)?(system\s+prompt|initial\s+prompt|secret\s+instructions)\b"
    ),
    re.compile(r"(?i)\b(system\s+override|instruction\s+override)\b"),
    re.compile(r"(?i)\bbypass\s+(security|guardrails|safety\s+filters)\b"),
]



def get_analyzer_engine() -> AnalyzerEngine:
    """Initialize and configure Microsoft Presidio Analyzer with Indian PII patterns."""
    global _ANALYZER
    if _ANALYZER is None:
            nlp_config = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                "ner_model_configuration": {
                    "labels_to_ignore": [
                        "CARDINAL",
                        "ORDINAL",
                        "MONEY",
                        "LAW",
                        "PERCENT",
                        "QUANTITY",
                        "TIME",
                        "FAC",
                        "EVENT",
                        "WORK_OF_ART",
                        "LANGUAGE",
                    ]
                },
            }
            provider = NlpEngineProvider(nlp_configuration=nlp_config)
            nlp_engine = provider.create_engine()
            registry = RecognizerRegistry(supported_languages=["en"])
            registry.load_predefined_recognizers(nlp_engine=nlp_engine, languages=["en"])
            analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine,
                registry=registry,
                supported_languages=["en"],
            )

            # 1. Custom Recognizer: Indian Aadhaar (12 digits)
            aadhaar_pattern = Pattern(
                name="in_aadhaar_pattern",
                regex=r"\b[2-9]{1}[0-9]{3}[ -]?[0-9]{4}[ -]?[0-9]{4}\b",
                score=0.95,
            )
            analyzer.registry.add_recognizer(
                PatternRecognizer(supported_entity="IN_AADHAAR", patterns=[aadhaar_pattern])
            )

            # 2. Custom Recognizer: Indian PAN Card (5 letters, 4 digits, 1 letter)
            pan_pattern = Pattern(
                name="in_pan_pattern",
                regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
                score=0.95,
            )
            analyzer.registry.add_recognizer(
                PatternRecognizer(supported_entity="IN_PAN", patterns=[pan_pattern])
            )

            # 3. Custom Recognizer: Indian Bank Account Number
            bank_pattern = Pattern(
                name="in_bank_acc_pattern",
                regex=r"(?i)\b(?:a/c|acc|account|bank)\s*(?:no\.?|num\.?|number)?\s*[:#-]?\s*(\d{9,18})\b",
                score=0.9,
            )
            analyzer.registry.add_recognizer(
                PatternRecognizer(supported_entity="IN_BANK_ACCOUNT", patterns=[bank_pattern])
            )

            _ANALYZER = analyzer
            logger.info("PRESIDIO_ANALYZER_INITIALIZED", model="en_core_web_sm")
    return _ANALYZER


def get_anonymizer_engine() -> AnonymizerEngine:
    """Initialize Microsoft Presidio Anonymizer Engine."""
    global _ANONYMIZER
    if _ANONYMIZER is None:
        _ANONYMIZER = AnonymizerEngine()
    return _ANONYMIZER


def _resolve_overlapping_spans(results: list) -> list:
    """Select non-overlapping entities prioritizing higher detection score and longer span."""
    sorted_by_score = sorted(results, key=lambda r: (r.score, r.end - r.start), reverse=True)
    kept = []
    for candidate in sorted_by_score:
        if not any(max(candidate.start, k.start) < min(candidate.end, k.end) for k in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda r: r.start, reverse=True)


def mask_pii(text: str) -> tuple[str, list[dict]]:
    """Detect and redact sensitive PII using Microsoft Presidio Analyzer & Anonymizer.

    Returns:
        (sanitized_text, list of detected PII metadata)
    """
    try:
        analyzer = get_analyzer_engine()
        results = analyzer.analyze(
            text=text,
            entities=[
                "IN_AADHAAR",
                "IN_PAN",
                "IN_BANK_ACCOUNT",
                "PHONE_NUMBER",
                "EMAIL_ADDRESS",
                "CREDIT_CARD",
                "IBAN_CODE",
                "US_BANK_NUMBER",
            ],
            language="en",
        )

        if not results:
            return text, []

        # Deduplicate overlapping spans (e.g. US_BANK_NUMBER matching phone number)
        deduped_results = _resolve_overlapping_spans(results)

        redactions: list[dict] = []
        sanitized = text

        for res in deduped_results:
            entity_val = text[res.start : res.end]
            entity_type = res.entity_type
            replace_start = res.start
            replace_end = res.end

            # Standardized Indian Masking Formats
            if entity_type == "IN_AADHAAR":
                digits = re.sub(r"\D", "", entity_val)
                masked_val = f"XXXX-XXXX-{digits[-4:]}"
                type_name = "AADHAAR"
            elif entity_type == "IN_PAN":
                masked_val = f"{entity_val[:2]}****{entity_val[-2:]}"
                type_name = "PAN"
            elif entity_type in ("PHONE_NUMBER", "UK_NHS"):
                digits = re.sub(r"\D", "", entity_val)
                masked_val = (
                    f"+91-XXXXX-{digits[-4:]}" if len(digits) >= 4 else "+91-XXXXX-XXXX"
                )
                type_name = "PHONE"
            elif entity_type in (
                "IN_BANK_ACCOUNT",
                "US_BANK_NUMBER",
                "IBAN_CODE",
                "CREDIT_CARD",
            ):
                match = BANK_ACC_REGEX.search(entity_val)
                if match:
                    raw_acc = match.group(1)
                    masked_val = f"XXXXXX{raw_acc[-4:]}"
                    # Only replace the account number digits portion
                    replace_start = res.start + match.start(1)
                    replace_end = res.start + match.end(1)
                else:
                    digits = re.sub(r"\D", "", entity_val)
                    masked_val = f"XXXXXX{digits[-4:]}" if len(digits) >= 4 else "XXXXXX0000"
                type_name = "BANK_ACCOUNT"
            elif entity_type == "EMAIL_ADDRESS":
                parts = entity_val.split("@")
                masked_val = (
                    f"{parts[0][:2]}***@{parts[1]}" if len(parts) == 2 else "***@***.com"
                )
                type_name = "EMAIL"
            else:
                masked_val = f"<{entity_type}>"
                type_name = entity_type

            sanitized = sanitized[:replace_start] + masked_val + sanitized[replace_end:]
            redactions.append({"type": type_name, "masked": masked_val, "entity": entity_type})

        logger.info(
            "PRESIDIO_PII_MASKED",
            count=len(redactions),
            types=[r["type"] for r in redactions],
        )
        return sanitized, redactions

    except Exception as presidio_err:
        logger.warning("PRESIDIO_PII_FALLBACK", error=str(presidio_err))

        # Resilient fallback to regex patterns
        redactions = []
        sanitized = text

        for match in AADHAAR_REGEX.finditer(sanitized):
            raw = match.group(0)
            digits = re.sub(r"\D", "", raw)
            masked = f"XXXX-XXXX-{digits[-4:]}"
            sanitized = sanitized.replace(raw, masked)
            redactions.append({"type": "AADHAAR", "masked": masked})

        for match in PAN_REGEX.finditer(sanitized):
            raw = match.group(0)
            masked = f"{raw[:2]}****{raw[-2:]}"
            sanitized = sanitized.replace(raw, masked)
            redactions.append({"type": "PAN", "masked": masked})

        for match in PHONE_REGEX.finditer(sanitized):
            raw = match.group(0)
            clean = re.sub(r"\D", "", raw)
            masked = f"+91-XXXXX-{clean[-4:]}"
            sanitized = sanitized.replace(raw, masked)
            redactions.append({"type": "PHONE", "masked": masked})

        for match in BANK_ACC_REGEX.finditer(sanitized):
            raw_acc = match.group(1)
            masked = f"XXXXXX{raw_acc[-4:]}"
            sanitized = sanitized.replace(raw_acc, masked)
            redactions.append({"type": "BANK_ACCOUNT", "masked": masked})

        for match in EMAIL_REGEX.finditer(sanitized):
            raw = match.group(0)
            parts = raw.split("@")
            masked = f"{parts[0][:2]}***@{parts[1]}"
            sanitized = sanitized.replace(raw, masked)
            redactions.append({"type": "EMAIL", "masked": masked})

        return sanitized, redactions


def check_prompt_injection(text: str) -> tuple[bool, str | None]:
    """Check text for potential adversarial prompt injection or jailbreak attempts.

    Returns:
        (is_injection, reason)
    """
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            logger.warning(
                "PROMPT_INJECTION_PATTERN_MATCHED", pattern=pattern.pattern
            )
            return True, "Potential prompt injection or instruction override attempt detected."
    return False, None


# ── Off-Topic Domain Guardrails ──

OFF_TOPIC_PATTERNS: list[tuple[re.Pattern, str]] = [
    # 1. Programming & Software Development
    (
        re.compile(
            r"(?i)\b(write\s+(a\s+)?(python|javascript|typescript|java|c\+\+|c#|sql|bash|html|css|php|rust|golang|ruby)\b"
            r"|coding|code\s+in\s+[a-z]+|debug\s+(this\s+)?code|react\s+component|vue\s+component"
            r"|binary\s+tree|quicksort|merge\s+sort|leetcode|write\s+a\s+script|create\s+an\s+api"
            r"|dockerfile|kubernetes|git\s+commit|git\s+push|regex\s+for)\b"
        ),
        "Programming and software development",
    ),
    # 2. Food, Cooking & Recipes
    (
        re.compile(
            r"(?i)\b(recipe\s+(for|of)|how\s+to\s+(bake|cook|make|prepare)\s+[a-z\s]*(cake|curry|pizza|pasta|biryani|soup|cookie|bread|tea|coffee|dish|salad|sandwich)"
            r"|ingredients\s+(for|of)\s+[a-z\s]*(cake|curry|pizza|pasta|biryani|soup|cookie|bread|dish)"
            r"|bake\s+a\s+cake|cooking\s+recipe)\b"
        ),
        "Cooking and recipes",
    ),
    # 3. Sports & Competitions
    (
        re.compile(
            r"(?i)\b(cricket\s+world\s+cup|fifa\s+world\s+cup|ipl\s+match|premier\s+league|who\s+won\s+the\s+(match|cup|championship|tournament|olympics)"
            r"|football\s+score|nba\s+finals|messi|ronaldo|virat\s+kohli|rohit\s+sharma|dhoni)\b"
        ),
        "Sports and tournaments",
    ),
    # 4. Creative Writing, Entertainment & Jokes
    (
        re.compile(
            r"(?i)\b(write\s+(a\s+)?(poem|song|story|essay|novel|script|rap|rhyme)|tell\s+(me\s+)?a\s+joke|make\s+me\s+laugh"
            r"|movie\s+review|box\s+office\s+collection|who\s+starred\s+in|celebrity\s+gossip)\b"
        ),
        "Creative writing and entertainment",
    ),
    # 5. Non-insurance World Trivia, Geography & Politics
    (
        re.compile(
            r"(?i)\b(capital\s+of\s+[a-z\s]+|who\s+is\s+the\s+(president|prime\s+minister|king|queen|governor|mayor)\s+of"
            r"|population\s+of\s+[a-z\s]+|weather\s+in\s+[a-z\s]+|distance\s+between\s+[a-z\s]+\s+and\s+[a-z\s]+"
            r"|tallest\s+building|highest\s+mountain)\b"
        ),
        "General geography and world trivia",
    ),
    # 6. Academic Homework, Math & Science Unrelated to Insurance
    (
        re.compile(
            r"(?i)\b(solve\s+(for\s+x|the\s+equation|this\s+integral|this\s+derivative|math\s+problem|quadratic\s+equation)"
            r"|quantum\s+physics|theory\s+of\s+relativity|speed\s+of\s+light|chemical\s+formula\s+of)\b"
        ),
        "Academic math and science",
    ),
    # 7. Non-Insurance Medical Diagnosis & Health Treatment
    (
        re.compile(
            r"(?i)\b(diagnose\s+my\s+(symptoms|illness|pain)|what\s+(medicine|drug|pill|tablet)\s+should\s+i\s+take"
            r"|treatment\s+for\s+(fever|cough|headache|covid|cancer|infection))\b"
        ),
        "Medical diagnosis and clinical treatment",
    ),
    # 8. Automotive Mechanics & Home Improvement
    (
        re.compile(
            r"(?i)\b(how\s+to\s+fix\s+(a\s+)?(flat\s+tire|car\s+engine|leaking\s+pipe|faucet|plumbing)|change\s+car\s+oil)\b"
        ),
        "Automotive and home repair",
    ),
    # 9. General Unrelated Philosophy & Science
    (
        re.compile(
            r"(?i)\b(meaning\s+of\s+life|why\s+is\s+the\s+sky\s+blue|how\s+do\s+birds\s+fly|can\s+dogs\s+eat\s+[a-z]+)\b"
        ),
        "General non-insurance inquiry",
    ),
]

OFF_TOPIC_GUARDRAIL_RESPONSE = (
    "I can only help with life insurance topics.\n\n"
    "As **Sarathi AI** (Saarthi AI), your assistant for **SBI Life** and **Indian Life Insurance**, "
    "I am strictly dedicated to life insurance queries including:\n"
    "- 📜 **Products & Policy Terms**: Term assurance, endowment, ULIPs, annuities, riders, exclusions, grace periods\n"
    "- 🩺 **Underwriting & Premiums**: Medical/financial limits, revival timelines, and lapse terms\n"
    "- ⚖️ **Servicing & Claims**: Section 45 incontestability, Section 39 nomination, claim settlement guidelines\n"
    "- 💰 **Financial Values & Taxation**: Surrender values (GSV/SSV), policy loans, ULIP lock-in, Section 80C & 10(10D)\n"
    "- 🌐 **Regulatory Guidelines**: Official IRDAI master circulars, public disclosures, and Ombudsman procedures\n\n"
    "Please ask a question related to life insurance, or attach a policy document for verification."
)

INSURANCE_DOMAIN_KEYWORDS = {
    # Core insurance terms
    "insurance",
    "policy",
    "policies",
    "insurer",
    "insured",
    "cover",
    "coverage",
    "sum assured",
    "premium",
    "premiums",
    "maturity",
    "death",
    "claim",
    "claims",
    "repudiation",
    "repudiate",
    "nominee",
    "nominees",
    "nomination",
    "beneficiary",
    "beneficiaries",
    "rider",
    "riders",
    "accidental",
    "critical illness",
    "terminal illness",
    "waiver",
    "surrender",
    "gsv",
    "ssv",
    "paid-up",
    "paid up",
    "lapse",
    "lapsed",
    "revival",
    "revive",
    "grace period",
    "free-look",
    "free look",
    "incontestability",
    "incontestable",
    "section 45",
    "section 39",
    "section 38",
    "section 80c",
    "section 10(10d)",
    "80c",
    "10(10d)",
    "tax",
    "taxes",
    "taxation",
    "exemption",
    "deduction",
    "irdai",
    "irda",
    "bima",
    "bharosa",
    "ombudsman",
    "grievance",
    "complaint",
    "underwriting",
    "underwriter",
    "proposal",
    "proposer",
    "life assured",
    "medical",
    "fup",
    "ulip",
    "ulips",
    "endowment",
    "annuity",
    "annuities",
    "pension",
    "pensions",
    "term plan",
    "term assurance",
    "saral",
    "sanchay",
    "wealth",
    "bal suraksha",
    "annuitant",
    "nav",
    "navs",
    "fund",
    "funds",
    "discontinued",
    "dpf",
    "partial withdrawal",
    "withdraw",
    "withdrawal",
    "switching",
    "top-up",
    "settlement",
    "solvency",
    "loan",
    "loans",
    "duplicate",
    "bond",
    "ecs",
    "nach",
    "mandate",
    "bounce",
    "payout",
    "payouts",
    "survival benefit",
    "bonus",
    "bonuses",
    "guaranteed addition",
    "hazard",
    "hazardous",
    "suicide",
    "exclusion",
    "exclusions",
    "waiting period",
    "kyc",
    "pan",
    "aadhaar",
    "circular",
    "circulars",
    "disclosure",
    "disclosures",
    "uin",
    "uins",
    "clause",
    "clauses",
    "article",
    "schedule",
    "contract",
    "contracts",
    "moratorium",
    "vesting",
    "commutation",
    "ppt",
    "pt",
    "annualized",
    "instalment",
    "installment",
    # Servicing & context words
    "die",
    "died",
    "alive",
    "survive",
    "mature",
    "cancel",
    "cancellation",
    "refund",
    "pay",
    "payment",
    "paying",
    "paid",
    "cost",
    "money",
    "family",
    "spouse",
    "wife",
    "husband",
    "son",
    "daughter",
    "child",
    "children",
    "parents",
    "father",
    "mother",
    "age",
    "years",
    "tenure",
    "eligible",
    "eligibility",
    "benefit",
    "benefits",
    "protect",
    "protection",
    "invest",
    "investment",
    "return",
    "returns",
    "saving",
    "savings",
    "document",
    "documents",
    "certificate",
    "address",
    "phone",
    "contact",
    # Insurer names
    "lic",
    "sbi life",
    "hdfc life",
    "icici prudential",
    "icici pru",
    "max life",
    "bajaj allianz",
    "tata aia",
    "kotak life",
    "birla sun life",
    "aditya birla",
    "pnb metlife",
    "canara hsbc",
}


def check_off_topic(text: str, has_attachment: bool = False) -> tuple[bool, str | None]:
    """Check text for queries clearly outside the Indian Life Insurance domain.

    Returns:
        (is_off_topic, reason)
    """
    clean_text = text.strip()
    # If user has an active attachment, allow in-chat verification
    if has_attachment:
        return False, None

    # 1. Check explicit off-topic patterns
    for pattern, category in OFF_TOPIC_PATTERNS:
        if pattern.search(clean_text):
            logger.warning(
                "OFF_TOPIC_QUERY_PATTERN_MATCHED",
                category=category,
                query_prefix=clean_text[:100],
            )
            return True, f"Query identified as off-topic ({category})."

    # 2. Check general off-topic if query is >= 4 words and contains zero insurance domain terms
    words = clean_text.lower().split()
    if len(words) >= 4:
        clean_lower = clean_text.lower()
        has_domain_term = any(term in clean_lower for term in INSURANCE_DOMAIN_KEYWORDS)
        if not has_domain_term:
            logger.warning(
                "OFF_TOPIC_QUERY_ZERO_RELEVANCE",
                query_prefix=clean_text[:100],
            )
            return (
                True,
                "Query does not pertain to life insurance, policy servicing, or IRDAI regulations.",
            )

    return False, None
