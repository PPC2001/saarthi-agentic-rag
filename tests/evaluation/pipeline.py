"""RAGAS Evaluation Pipeline for Saarthi AI.

Inspired by enterprise-rag's evals/pipeline.py and evals/metrics.py.
Adapted for the Indian Life Insurance domain with SBI Life / IRDAI knowledge context.

Two-Phase Evaluation:
  Phase 1 — Live Pipeline:
    Calls the running FastAPI /api/v1/query endpoint for each golden sample.
    Captures: actual_response, actual_contexts (citations), actual_tools_called (query_type).

  Phase 2 — RAGAS Scoring:
    Runs 6 evaluation experiments using RAGAS framework:
      1. Faithfulness        — Is the answer grounded in retrieved context?
      2. Answer Relevancy    — Does the answer address the actual question?
      3. Context Precision   — Are retrieved contexts relevant to the question?
      4. Context Recall      — Did retrieval capture all relevant reference context?
      5. Answer Correctness  — How close is the answer to the reference ground truth?
      6. Tool Correctness    — Did the router choose the correct execution path?
         (Jaccard: actual_tools_called ∩ expected_tools / actual ∪ expected)

Usage:
  # Run full evaluation (requires backend running on port 8000)
  uv run python tests/evaluation/pipeline.py

  # Or run as pytest (for CI)
  uv run pytest tests/evaluation/ -v -s

Configuration:
  API_URL       = http://localhost:8000/api/v1/query
  DEV_API_KEY   = dev-test-key-12345
  RAGAS_LLM     = Set LANGCHAIN_API_KEY and OPENAI_API_KEY for RAGAS judge LLM
"""

import copy
import json
import time
from pathlib import Path

import logfire
import requests

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000/api/v1/query"
DEV_API_KEY = "dev-test-key-12345"
RESPONSE_TRUNCATE = 2000  # chars — keep concise for RAGAS judge token budget
DELAY_BETWEEN_CALLS = 5  # seconds — avoid rate limiting on LLM backend
REQUEST_TIMEOUT = 120  # seconds — full LangGraph pipeline can take >30s

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_OUTPUT_PATH = Path(__file__).parent / "eval_results.json"


# ── Tool detection ─────────────────────────────────────────────────────────────


def detect_tool(query_type: str) -> str:
    """Map query_type from API response to an expected_tools-compatible label.

    Taxonomy:
      greeting             → greeting
      off_topic            → guardrail_filter
      security_blocked     → guardrail_filter
      general_on_topic     → synthesis
      structured_rag       → retrieval + synthesis
      actuarial_calculation→ actuarial_calculation
      web_search           → web_search
      attachment_rag       → retrieval + synthesis
      domain_vector_filter → guardrail_filter
    """
    mapping = {
        "greeting": "greeting",
        "off_topic": "guardrail_filter",
        "security_blocked": "guardrail_filter",
        "domain_vector_filter": "guardrail_filter",
        "general_on_topic": "synthesis",
        "structured_rag": "retrieval",
        "actuarial_calculation": "actuarial_calculation",
        "web_search": "web_search",
        "attachment_rag": "retrieval",
    }
    return mapping.get(query_type, "unknown")


# ── Phase 1: Live Pipeline ─────────────────────────────────────────────────────


def load_golden_dataset() -> dict:
    """Load the golden evaluation dataset."""
    with open(GOLDEN_DATASET_PATH) as f:
        return json.load(f)


def run_live_pipeline(
    golden_dataset: dict,
    progress_callback=None,
) -> dict:
    """Phase 1: Enrich each golden sample with actual API responses.

    Calls the live /api/v1/query endpoint for each sample and captures:
      - actual_response:     LLM-generated answer text
      - actual_contexts:     Citation snippets (retrieved chunks)
      - actual_tools_called: Detected execution path (from query_type)
    """
    dataset = copy.deepcopy(golden_dataset)
    samples = dataset["rag_samples"]
    n = len(samples)

    with logfire.span("🚀 Eval Phase 1 — Live Pipeline", total_samples=n):
        for i, sample in enumerate(samples):
            question = sample["question"]

            if progress_callback:
                progress_callback(i, n, question, "calling")

            with logfire.span(
                f"📤 Live Query {i + 1}/{n}",
                question=question[:80],
                domain=sample.get("domain", ""),
            ):
                try:
                    resp = requests.post(
                        API_URL,
                        json={
                            "query": question,
                            "include_citations": True,
                        },
                        headers={
                            "X-API-Key": DEV_API_KEY,
                            "Content-Type": "application/json",
                        },
                        timeout=REQUEST_TIMEOUT,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    raw_answer = data.get("answer", "")
                    query_type = data.get("query_type", "unknown")
                    citations = data.get("citations") or []

                    # Extract citation snippets as context strings for RAGAS
                    context_snippets = [
                        c.get("snippet", c.get("text", ""))
                        for c in citations[:5]
                        if c.get("snippet") or c.get("text")
                    ]

                    sample["actual_response"] = raw_answer[:RESPONSE_TRUNCATE]
                    sample["actual_contexts"] = context_snippets
                    sample["actual_tools_called"] = [detect_tool(query_type)]
                    sample["actual_query_type"] = query_type

                    logfire.info(
                        "✅ Response captured",
                        tool=sample["actual_tools_called"][0],
                        query_type=query_type,
                        response_chars=len(raw_answer),
                        context_count=len(context_snippets),
                    )

                except requests.exceptions.ConnectionError:
                    logfire.error("❌ Cannot reach FastAPI — is the backend running on port 8000?")
                    _fill_empty(sample)

                except Exception as e:
                    logfire.error(f"❌ Query {i + 1} failed: {e}")
                    _fill_empty(sample)

            if progress_callback:
                progress_callback(i, n, question, "done", sample["actual_response"])

            if i < n - 1:
                time.sleep(DELAY_BETWEEN_CALLS)

    return dataset


def _fill_empty(sample: dict) -> None:
    """Fill a sample with empty values when API call fails."""
    sample["actual_response"] = ""
    sample["actual_contexts"] = sample.get("relevant_contexts", [])
    sample["actual_tools_called"] = ["unknown"]
    sample["actual_query_type"] = "unknown"


def save_results(dataset: dict, path: Path = RESULTS_OUTPUT_PATH) -> None:
    """Save enriched dataset to JSON for Phase 2 RAGAS scoring."""
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"✅ Phase 1 results saved to: {path}")


# ── Phase 2: RAGAS Scoring ─────────────────────────────────────────────────────


def run_ragas_evaluation(
    dataset: dict,
    status_cb=None,
) -> dict:
    """Phase 2: Score each sample using RAGAS evaluation framework.

    Runs 6 experiments:
      1. Faithfulness        (LLM judge)
      2. Answer Relevancy    (LLM judge + embeddings)
      3. Context Precision   (LLM judge)
      4. Context Recall      (LLM judge)
      5. Answer Correctness  (LLM judge + embeddings)
      6. Tool Correctness    (Jaccard — zero LLM calls)

    Returns:
        Dict mapping metric_name → list of (question, score) records
    """
    try:
        from ragas import evaluate  # type: ignore[import-untyped]
        from ragas.metrics import (  # type: ignore[import-untyped]
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as e:
        print(f"❌ RAGAS not installed: {e}")
        print("Install with: pip install ragas")
        return {}

    samples = dataset["rag_samples"]

    # Filter samples that have actual responses (skip failed API calls)
    valid_samples = [s for s in samples if s.get("actual_response")]
    print(f"📊 Running RAGAS on {len(valid_samples)}/{len(samples)} valid samples")

    results: dict = {}

    # ── Experiment 6: Tool Correctness (Jaccard — zero LLM) ──────────────────
    if status_cb:
        status_cb("⚡ Tool Correctness (Jaccard — no LLM)...")
    tool_rows = []
    for s in samples:
        called = set(s.get("actual_tools_called") or ["unknown"])
        expected = set(s.get("expected_tools") or [])
        union = len(called | expected)
        jaccard = len(called & expected) / union if union > 0 else 0.0
        tool_rows.append(
            {
                "question": s["question"][:65],
                "domain": s.get("domain", ""),
                "expected": list(expected),
                "actual": list(called),
                "tool_correctness": round(jaccard, 3),
            }
        )
    results["tool_correctness"] = tool_rows
    avg_tool = sum(r["tool_correctness"] for r in tool_rows) / len(tool_rows) if tool_rows else 0
    print(f"   Tool Correctness avg: {avg_tool:.3f}")

    # For RAGAS LLM-based metrics, we need valid samples with non-empty context
    rag_samples = [
        s
        for s in valid_samples
        if s.get("actual_contexts")
        and s.get("actual_response")
        and s.get("domain") not in ("off_topic", "greeting")  # Skip guardrail samples
    ]

    if not rag_samples:
        print("⚠️ No valid RAG samples for RAGAS LLM metrics (all were guardrail routes)")
        return results

    # Build RAGAS dataset
    ragas_data = {
        "question": [s["question"] for s in rag_samples],
        "answer": [s["actual_response"] for s in rag_samples],
        "contexts": [s["actual_contexts"] for s in rag_samples],
        "ground_truth": [s["reference"] for s in rag_samples],
    }

    try:
        from datasets import Dataset  # type: ignore[import-untyped]

        ragas_dataset = Dataset.from_dict(ragas_data)

        if status_cb:
            status_cb(f"🧪 Running RAGAS on {len(rag_samples)} samples...")

        scored = evaluate(
            ragas_dataset,
            metrics=[
                Faithfulness(),
                AnswerRelevancy(),
                ContextPrecision(),
                ContextRecall(),
                AnswerCorrectness(),
            ],
        )

        df = scored.to_pandas()

        for metric in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "answer_correctness",
        ]:
            if metric in df.columns:
                avg = df[metric].mean()
                results[metric] = df[["question", metric]].to_dict("records")
                print(f"   {metric} avg: {avg:.3f}")

    except Exception as e:
        print(f"⚠️ RAGAS LLM scoring failed: {e}")
        print("   Make sure OPENAI_API_KEY or LANGCHAIN_API_KEY is set.")

    return results


def print_summary(results: dict) -> None:
    """Print a summary table of all evaluation results."""
    print("\n" + "=" * 60)
    print("📊 SAARTHI AI — RAGAS EVALUATION SUMMARY")
    print("=" * 60)

    metric_order = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
        "tool_correctness",
    ]

    for metric in metric_order:
        if metric in results:
            rows = results[metric]
            scores = [r.get(metric, r.get("tool_correctness", 0)) for r in rows]
            avg = sum(scores) / len(scores) if scores else 0.0
            status = "✅" if avg >= 0.7 else "⚠️" if avg >= 0.5 else "❌"
            print(f"  {status} {metric:<25} avg: {avg:.3f} ({len(scores)} samples)")

    print("=" * 60)


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🚀 Starting Saarthi AI RAGAS Evaluation Pipeline")
    print(f"   API: {API_URL}")
    print(f"   Golden dataset: {GOLDEN_DATASET_PATH}")
    print()

    # Phase 1: Live Pipeline
    print("📤 Phase 1: Calling live API for each golden sample...")
    golden = load_golden_dataset()
    enriched = run_live_pipeline(
        golden,
        progress_callback=lambda i, n, q, stage, r="": print(f"  [{i + 1}/{n}] {stage}: {q[:60]}"),
    )
    save_results(enriched)

    # Phase 2: RAGAS Scoring
    print("\n🧪 Phase 2: Running RAGAS evaluation metrics...")
    results = run_ragas_evaluation(enriched, status_cb=print)
    print_summary(results)

    # Save scored results
    scored_path = GOLDEN_DATASET_PATH.parent / "eval_scored.json"
    with open(scored_path, "w") as f:
        json.dump({"results": results, "timestamp": time.time()}, f, indent=2)
    print(f"\n✅ Scored results saved to: {scored_path}")
