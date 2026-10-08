"""Domain Vector Anchor Filter for Saarthi AI.

Provides ultra-fast (sub-5ms) mathematical domain boundary enforcement using
precomputed vector cosine similarity against Indian Life Insurance domain prototypes.

Inspired by enterprise-rag's domain_filter.py — adapted for the SBI Life /
IRDAI / Indian insurance regulatory domain.

Architecture:
  - Allowlist approach (NOT a blocklist of forbidden topics)
  - Precompute prototype embeddings at startup once
  - On each query: cosine_sim(query, prototypes) → if max_sim < threshold → block
  - Runs BEFORE the LLM router node (saves ~300-800ms on off-topic queries)

3-Tier Defense (after this filter):
  Tier 1: Domain Vector Filter  (this module, <5ms, zero LLM calls)
  Tier 2: LLM Router Node       (~300ms, 4-taxonomy classification)
  Tier 3: CRAG FlashRank Gating (~50ms, per-chunk relevance scoring)

Fail-Open Design:
  Any exception during embedding or similarity computation causes the filter to
  return (True, 1.0, "Fallback On Error") — ensuring a transient network issue
  never blocks a legitimate user query. Downstream LLM router provides secondary defense.
"""

from __future__ import annotations

import threading

import logfire
import numpy as np

from app.observability.logging import get_logger

logger = get_logger("domain_filter")

# ── Domain Prototype Definitions (Allowlist Scope) ────────────────────────────
# Each tuple: (label, description_text_used_to_compute_prototype_embedding)
# The description should be rich enough to cover the full concept space.

INSURANCE_DOMAIN_PROTOTYPES: list[tuple[str, str]] = [
    (
        "Life Insurance Core Concepts",
        "Life insurance policy, term assurance, endowment, sum assured, premium, "
        "death benefit, maturity benefit, policy term, premium paying term, "
        "grace period, lapse, revival, surrender, paid-up, policy loan, "
        "rider, critical illness, accidental death, waiver of premium, "
        "free-look period, policy document, UIN, insured, policyholder, proposer.",
    ),
    (
        "IRDAI Regulatory & Compliance",
        "IRDAI Insurance Regulatory and Development Authority India, "
        "IRDAI circular, master circular, IRDAI regulations, insurance act 1938, "
        "Section 45 incontestability, Section 39 nomination, Section 38 assignment, "
        "Protection of Policyholders Interests, IRDAI guidelines, compliance, "
        "ombudsman, grievance redressal, surrender regulation, GSV guaranteed surrender value, "
        "SSV special surrender value, discontinuance policy fund DPF, lock-in period.",
    ),
    (
        "SBI Life Products & Underwriting",
        "SBI Life Smart Platina Plus eShield Next Saral Term annuity ULIP Smart Wealth, "
        "underwriting medical financial limits, entry age, maximum age, "
        "Bajaj Allianz HDFC Life ICICI Prudential Max Life LIC life insurance company, "
        "product brochure benefit illustration Key Feature Document KFD, "
        "suicide clause waiting period exclusion hazardous occupation.",
    ),
    (
        "Financial Values & Actuarial Calculations",
        "Calculate policy loan surrender value GSV SSV cash value, "
        "actuarial calculation incontestability section 45 timeline, "
        "grace period expiry date revival deadline FUP first unpaid premium, "
        "NAV net asset value fund switch partial withdrawal ULIP fund value, "
        "bonus reversionary terminal guaranteed addition, section 80C 10(10D) tax exemption deduction.",
    ),
    (
        "Greeting & Conversational",
        "hello hi good morning good afternoon good evening hey there namaste, "
        "who are you what can you do help me assist, "
        "what is Saarthi what is Sarathi AI overview capabilities, "
        "thank you thanks well done good job, "
        "tell me about yourself what do you know.",
    ),
]

# Cosine similarity threshold — queries below this are blocked without LLM call
# Set conservatively at 0.30 to allow borderline insurance queries through to LLM router
DOMAIN_SIMILARITY_THRESHOLD: float = 0.30

_anchor_lock = threading.RLock()
_anchor_embeddings: list[tuple[str, np.ndarray]] | None = None


async def initialize_domain_anchors(embedder) -> list[tuple[str, np.ndarray]]:
    """Precompute and cache domain prototype embeddings (once per process).

    Uses a single batch embed_documents call for maximum efficiency.
    Call this from the FastAPI lifespan startup handler to eliminate cold-start latency.
    """
    global _anchor_embeddings
    with _anchor_lock:
        if _anchor_embeddings is not None:
            return _anchor_embeddings

    logger.info(
        "PRECOMPUTING_DOMAIN_VECTOR_ANCHORS",
        count=len(INSURANCE_DOMAIN_PROTOTYPES),
    )

    try:
        texts = [text for _, text in INSURANCE_DOMAIN_PROTOTYPES]
        vec_lists = await embedder.embed_documents(texts)

        if getattr(embedder, "is_fallback", False) is True:
            logger.warning("DOMAIN_ANCHORS_FALLBACK_EMBEDDER_DETECTED")
            with _anchor_lock:
                _anchor_embeddings = []
            return []

        anchors: list[tuple[str, np.ndarray]] = []
        for (label, _), vec_list in zip(INSURANCE_DOMAIN_PROTOTYPES, vec_lists, strict=False):
            vec = np.array(vec_list, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            anchors.append((label, vec))

        with _anchor_lock:
            _anchor_embeddings = anchors

        logger.info("DOMAIN_ANCHORS_INITIALIZED", count=len(anchors))
        return _anchor_embeddings
    except Exception as e:
        logger.warning("DOMAIN_ANCHORS_INIT_FAILED", error=str(e))
        return []


def check_domain_relevance(
    query: str,
    query_embedding: list[float],
    threshold: float = DOMAIN_SIMILARITY_THRESHOLD,
    embedder=None,  # type: ignore[assignment]
) -> tuple[bool, float, str]:
    """Check if a user query belongs to the Indian Life Insurance domain.

    This is the fast-path check (pure numpy math, <1ms once anchors cached).

    Args:
        query:           Raw query string (for logging)
        query_embedding: Pre-computed query embedding vector
        threshold:       Min cosine similarity to accept (default: 0.30)
        embedder:        Embedder instance needed only if anchors not yet initialized

    Returns:
        (is_in_domain, best_similarity_score, matched_prototype_label)

    Fail-Open: Returns (True, 1.0, "Fallback On Error") on any exception to
    ensure availability. LLM router provides secondary domain defense.
    """
    clean_query = query.strip()
    if not clean_query:
        return False, 0.0, "Empty Query"

    try:
        if embedder is not None and getattr(embedder, "is_fallback", False) is True:
            logger.info("DOMAIN_FILTER_FAIL_OPEN_FALLBACK_EMBEDDER", query_prefix=clean_query[:80])
            return True, 1.0, "Fallback Embedder Fail Open"

        if not _anchor_embeddings:
            # Anchors not ready (embedder not initialized at startup) — fail open
            logger.warning("DOMAIN_ANCHORS_NOT_INITIALIZED_FAIL_OPEN", query_prefix=clean_query[:80])
            return True, 1.0, "Anchors Not Initialized"

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return False, 0.0, "Zero Norm Query"

        q_unit = q_vec / q_norm

        best_score: float = -1.0
        best_label: str = "None"

        for label, a_unit in _anchor_embeddings:
            # Anchors are pre-normalized; dot product = cosine similarity
            cos_sim = float(np.dot(q_unit, a_unit))
            if cos_sim > best_score:
                best_score = cos_sim
                best_label = label

        is_in_domain = best_score >= threshold

        with logfire.span(
            "🛡️ Domain Vector Filter",
            query=clean_query[:80],
            best_score=round(best_score, 4),
            best_domain=best_label,
            allowed=is_in_domain,
        ):
            if is_in_domain:
                logger.info(
                    "DOMAIN_FILTER_PASSED",
                    score=round(best_score, 4),
                    threshold=threshold,
                    domain=best_label,
                )
            else:
                logger.warning(
                    "DOMAIN_FILTER_BLOCKED",
                    score=round(best_score, 4),
                    threshold=threshold,
                    best_domain=best_label,
                )

            return is_in_domain, best_score, best_label

    except Exception as e:
        logger.warning("DOMAIN_FILTER_FAIL_OPEN", error=str(e))
        return True, 1.0, "Fallback On Error"


def reset_anchors() -> None:
    """Reset cached anchors (used in tests to force re-initialization)."""
    global _anchor_embeddings
    with _anchor_lock:
        _anchor_embeddings = None
