# Saarthi AI — Enterprise Indian Life Insurance Policy Servicing Platform

> **Production-grade Agentic RAG (Retrieval-Augmented Generation) backend** built for SBI Life's internal corporate knowledge system. Enables advisors, bancassurance partners, CSRs, underwriters, and compliance officers to query master policy contracts, IRDAI regulations, and proprietary product brochures with legal-grade accuracy.

[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-orange)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red)](https://qdrant.tech)
[![Tests](https://img.shields.io/badge/Tests-61%20passed-brightgreen)](#testing)

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [System Components](#system-components)
3. [LangGraph State Machine (Nodes)](#langgraph-state-machine)
4. [Intent Classification Router](#intent-classification-router)
5. [Regulatory & Policy Scope](#regulatory--policy-scope)
6. [Tech Stack](#tech-stack)
7. [Project Structure](#project-structure)
8. [Configuration Reference](#configuration-reference)
9. [Quick Start](#quick-start)
10. [API Endpoints](#api-endpoints)
11. [Testing](#testing)
12. [Docker Deployment](#docker-deployment)
13. [Observability](#observability)
14. [Security & Guardrails](#security--guardrails)

---

## 🏛️ Architecture Overview

The platform uses a **Corrective Agentic RAG (CRAG)** state machine orchestrated via **LangGraph**. At its core is a **Structural Intent Classification Router Node** that uses an LLM to classify every incoming query into one of four predefined corporate taxonomy categories, then routes it to the optimal execution path:

```
                          ┌──────────────────────────────────┐
                          │         Employee Query           │
                          └───────────────┬──────────────────┘
                                          ▼
                          ┌──────────────────────────────────┐
                          │   LLM Router Node (Intent)       │
                          │  [greeting | off_topic |         │
                          │   general_on_topic |             │
                          │   structured_rag]                │
                          │  + PII Scrub + Injection Check   │
                          └──┬─────────┬──────────┬──────────┘
                             │         │          │
              ┌──────────────┘  ┌──────┘          └────────────────┐
              ▼                 ▼                                   ▼
  ┌────────────────┐   ┌────────────────────┐            ┌─────────────────────┐
  │  Fast Path END │   │ synthesis_node     │            │  retrieval_node     │
  │ (greeting /    │   │ (general_on_topic  │            │  (structured_rag /  │
  │  off_topic)    │   │  skips retrieval)  │            │  actuarial /        │
  └────────────────┘   └────────────────────┘            │  web_search)        │
                                                         └────────┬────────────┘
                                                                  ▼
                                                      ┌───────────────────────┐
                                                      │  document_grader_node │
                                                      │  (CRAG relevance)     │
                                                      └────────┬──────────────┘
                                                               │
                                              ┌────────────────┴────────────────┐
                                              ▼                                 ▼
                                    ┌──────────────────┐             ┌────────────────────┐
                                    │  synthesis_node  │             │  web_search_node   │
                                    │  (LLM answer)    │◄────────────│  (IRDAI portal /   │
                                    └────────┬─────────┘             │  SBI Life web)     │
                                             ▼                       └────────────────────┘
                                    ┌──────────────────┐
                                    │  compliance_guard │
                                    │  (ULIP Notice /   │
                                    │   Section 45)     │
                                    └────────┬──────────┘
                                             ▼
                                    ┌──────────────────┐
                                    │  Verified Answer  │
                                    │  + Citations      │
                                    └──────────────────┘
```

---

## 🧩 System Components

### 1. Structural Intent Classification Router (`router_node`)
The entry point of every query. Uses `openai/gpt-oss-120b` via Portkey to classify the query before any retrieval happens. Includes:
- **Prompt Injection Detection** (blocks override attempts)
- **PII Scrubbing** (Aadhaar, PAN, phone numbers via Microsoft Presidio)
- **4-Category LLM Taxonomy** (see [Intent Classification Router](#intent-classification-router))
- **Heuristic Fallback** (regex-based classifier when LLM is unavailable)

### 2. Qdrant Vector Retrieval (`retrieval_node`)
- **Dense vector search** with `Qwen3-Embedding-8B` (4096-dimension embeddings)
- **Payload filtering** by `product_code`, `state`, and `section`
- Automatic catalog pre-seeding on startup
- Batch ingestion pipeline for uploaded policy bonds and endorsements

### 3. Document Relevance Grader (`document_grader_node`)
- Corrective RAG (CRAG) grading: evaluates retrieved chunks for relevance
- Routes to `web_search` if retrieved context quality is **insufficient**
- Routes to `synthesis` if context quality is **adequate**

### 4. Actuarial Calculator (`actuarial_calculator_node`)
Deterministic, hallucination-free calculations for:
- **Policy Loan Eligibility** — max 80% of Special Surrender Value after 2 full premium years
- **Surrender Value Computation** — Guaranteed Surrender Value (GSV) vs Special Surrender Value (SSV)
- **Section 45 Incontestability Timeline** — 3-year bar from policy inception/revival
- **Grace Period & Revival Deadlines** — 15/30 days grace, 5-year revival window from FUP

### 5. Web Search (`web_search_node`)
- Live IRDAI portal scraping for latest regulatory circulars
- SBI Life official website for current product disclosures
- Domain allowlist enforcement (only trusted regulatory/insurer URLs)

### 6. LLM Answer Synthesizer (`synthesis_node`)
- Multi-source context aggregation (vault docs + web + actuarial results)
- Conflict detection when user-uploaded documents contradict official SBI Life terms
- Role-specific persona prompts (advisor, CSR, underwriter, compliance, operations)
- Mandatory statutory disclaimer injection

### 7. Compliance Output Guard (`compliance_guard_node`)
- Enforces mandatory ULIP market-risk disclosures
- Injects Section 45 claim settlement statutory notices
- Blocks legally non-compliant output patterns

---

## 🧠 LangGraph State Machine

The graph is defined in [`src/app/core/graphs/retrieval/`](src/app/core/graphs/retrieval/) and follows strict SOLID principles:

| File | Responsibility |
|------|----------------|
| [`state.py`](src/app/core/graphs/retrieval/state.py) | Pydantic `AgenticRAGState` model and regex constants |
| [`nodes/router_node.py`](src/app/core/graphs/retrieval/nodes/router_node.py) | LLM-based intent classification + PII scrub + guardrails |
| [`nodes/actuarial.py`](src/app/core/graphs/retrieval/nodes/actuarial.py) | Deterministic actuarial calculation tools |
| [`nodes/retriever.py`](src/app/core/graphs/retrieval/nodes/retriever.py) | Qdrant vector retrieval with payload filtering |
| [`nodes/grader.py`](src/app/core/graphs/retrieval/nodes/grader.py) | CRAG document relevance grading |
| [`nodes/web_search.py`](src/app/core/graphs/retrieval/nodes/web_search.py) | Live regulatory web search |
| [`nodes/synthesizer.py`](src/app/core/graphs/retrieval/nodes/synthesizer.py) | LLM answer synthesis with persona prompts |
| [`nodes/compliance.py`](src/app/core/graphs/retrieval/nodes/compliance.py) | Output compliance guardrail injection |
| [`router.py`](src/app/core/graphs/retrieval/router.py) | Conditional edge routing functions |
| [`builder.py`](src/app/core/graphs/retrieval/builder.py) | Graph assembly and compilation |
| [`graph.py`](src/app/core/graphs/retrieval/graph.py) | Public facade module (re-exports all) |

---

## 🎯 Intent Classification Router

Every query is classified by the LLM into **exactly one** of four corporate taxonomy categories:

| Category | Criteria | Execution Path |
|----------|----------|----------------|
| `"greeting"` | Basic pleasantries, system handshakes, identity questions | → `END` (fast path, friendly greeting) |
| `"off_topic"` | Requests unrelated to life insurance, IRDAI, or policy frameworks | → `END` (fast path, scoped refusal) |
| `"general_on_topic"` | Generic conceptual definitions (e.g., "What is a ULIP?", "What is Sum Assured?") | → `synthesis` (bypasses empty vector search) |
| `"structured_rag"` | High-precision queries requiring specific document context, product matrices, or compliance lookups | → `retrieval` → `document_grader` → `synthesis` |

**Sub-routing within `structured_rag`:**
- Calculation keywords → `actuarial_calculator` → `retrieval`
- IRDAI portal/circular keywords → `web_search`
- Uploaded attachment references → `attachment_rag`

**Critical guardrail constraints:**
- Mixed-intent queries (e.g., "Hello, what is the surrender fee?") → `structured_rag` (operational intent wins)
- Insurance term with off-topic context (e.g., "policy on building a website") → `off_topic`

**Live test results:**

| Query | Classified As | Result |
|-------|--------------|--------|
| `"what is ulip policy ?"` | `general_on_topic` | Full structured ULIP explainer with IRDAI clauses |
| `"what is today's cricket score ?"` | `off_topic` | Polite scoped-refusal with insurance domain redirect |
| `"hello good morning"` | `greeting` | Saarthi AI welcome message |
| `"What is surrender penalty for Smart Platina Plus in 3rd year?"` | `structured_rag` | Qdrant + Web retrieval with citations |
| `"Calculate policy loan for surrender value of 500000"` | `actuarial_calculation` | Deterministic calculation (80% of SSV) |

---

## ⚖️ Regulatory & Policy Scope

Aligned with Indian insurance law and IRDAI regulations:

### Insurance Act, 1938
| Section | Provision |
|---------|-----------|
| **Section 45** | Incontestability — no policy can be challenged after **3 years** from inception or revival date |
| **Section 39** | Nomination — beneficial nominees (spouse, children, parents) take policy proceeds in order |
| **Section 38** | Assignment — absolute vs conditional assignment rules and insurer's acknowledgment requirements |

### IRDAI (Protection of Policyholders' Interests) Regulations
| Regulation | Provision |
|------------|-----------|
| **Grace Period** | 15 calendar days for monthly mode; 30 calendar days for quarterly/half-yearly/annual |
| **Free-Look Period** | 15 days from receipt of physical policy; 30 days for electronic / distance-marketing issuance |
| **ULIP 5-Year Lock-in** | Discontinued Policy Fund (DPF) guarantees minimum 4% p.a. during lock-in |
| **Suicide Clause** | Minimum 80% refund of all premiums paid if death occurs within 12 months of policy issuance/revival |
| **Surrender Value** | GSV (%) of total premiums paid, or SSV (asset-share backing), whichever is higher |
| **Revival Period** | Up to 5 consecutive years from date of first unpaid premium (FUP) |

### SBI Life Product Lines Covered
- **Term Assurance**: Saral Term, eShield Next (level/increasing SA, TROP, terminal illness accelerator)
- **ULIP**: Dual benefit (protection + NAV-linked investment), fund switching, partial withdrawals post lock-in
- **Guaranteed Savings & Endowment**: Smart Platina Plus, guaranteed additions, reversionary bonuses, policy loans
- **Annuity & Pension**: Immediate annuity (with/without ROP), deferred pension schemes

---

## 🛠 Tech Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **API Framework** | FastAPI | ≥0.115 | Async REST API, OpenAPI docs, middleware |
| **LLM Orchestration** | LangGraph | ≥1.0 | Agentic state machine, conditional routing |
| **LLM Client** | LangChain OpenAI | ≥0.2 | LLM invocation, message formatting |
| **LLM Gateway** | Portkey AI | ≥1.9 | Virtual keys, fallback, observability |
| **LLM Model (DEV)** | `openai/gpt-oss-120b` | — | Via Portkey → Groq routing |
| **LLM Model (PROD)** | Claude Sonnet 4 (Bedrock) | — | Via Portkey → AWS Bedrock |
| **Embeddings** | Qwen3-Embedding-8B | 4096d | Dense vector embeddings via Portkey |
| **Vector Database** | Qdrant | ≥1.12 | Payload-filtered cosine similarity search |
| **Cache** | Redis | ≥7.2 | Semantic query caching (95% similarity TTL) |
| **Document Parsing** | PyMuPDF + pdfplumber + python-docx | — | Policy bond PDF/DOCX extraction |
| **PII Masking** | Microsoft Presidio | ≥2.2.364 | Aadhaar, PAN, phone, email scrubbing |
| **Input Guardrails** | Guardrails AI | ≥0.11 | Prompt injection, jailbreak, topic boundary |
| **Observability** | Logfire + structlog | ≥2.0 / ≥24 | Structured logging, traces, spans |
| **AWS SDK** | boto3 + aioboto3 | ≥1.35 / ≥13 | S3 (doc storage), DynamoDB (API keys) |
| **Testing** | pytest + pytest-asyncio | ≥8.3 / ≥0.24 | 61 unit tests, async test support |
| **Linting** | Ruff | ≥0.6 | Fast Python linter + formatter |
| **Type Checking** | mypy | ≥1.11 | Strict static type analysis |
| **Package Manager** | Astral `uv` | — | Ultra-fast Python package management |
| **Containerization** | Docker + Compose | — | Local stack (App + Qdrant + Redis) |

---

## 📁 Project Structure

```
saarthi-ai/
├── src/
│   └── app/
│       ├── main.py                        # FastAPI entrypoint + middleware stack
│       ├── config.py                      # Pydantic Settings (ENV-driven config)
│       ├── dependencies.py                # DI container (LLM, Qdrant, Redis, Embedder)
│       │
│       ├── api/
│       │   ├── api_v1/
│       │   │   ├── router.py              # API v1 router aggregation
│       │   │   └── endpoints/
│       │   │       ├── health.py          # /health, /live, /ready probes
│       │   │       ├── info.py            # / root info endpoint
│       │   │       ├── query.py           # POST /v1/query — main RAG endpoint
│       │   │       ├── ingest.py          # POST /v1/ingest — document ingestion
│       │   │       └── session.py         # POST /v1/session/attach, DELETE /v1/session/{id}/clear
│       │   ├── deps.py                    # FastAPI dependency providers
│       │   ├── middleware/
│       │   │   ├── auth.py                # API Key + Google OAuth Bearer validation
│       │   │   ├── audit.py               # HIPAA compliance audit trail middleware
│       │   │   ├── logging.py             # X-Request-ID correlation middleware
│       │   │   └── rate_limit.py          # Sliding window rate limiter (60 req/min)
│       │   ├── schemas/
│       │   │   ├── query.py               # PolicyQueryRequest, PolicyAnswerResponse, Citation
│       │   │   ├── common.py              # APIErrorResponse, ErrorDetail
│       │   │   └── exceptions.py          # BaseDomainException (RFC 7807 Problem Details)
│       │   └── utils.py                   # Response helpers
│       │
│       ├── core/
│       │   ├── graphs/
│       │   │   └── retrieval/
│       │   │       ├── state.py           # AgenticRAGState + regex constants
│       │   │       ├── router.py          # Conditional edge routing functions
│       │   │       ├── builder.py         # LangGraph assembly + compilation
│       │   │       ├── graph.py           # Public facade (re-exports all)
│       │   │       └── nodes/
│       │   │           ├── router_node.py # ⭐ LLM intent classification (4-taxonomy)
│       │   │           ├── actuarial.py   # Deterministic actuarial tools
│       │   │           ├── retriever.py   # Qdrant vector retrieval
│       │   │           ├── grader.py      # CRAG relevance grading
│       │   │           ├── web_search.py  # Live regulatory web search
│       │   │           ├── synthesizer.py # LLM answer synthesis
│       │   │           ├── compliance.py  # Output guardrail injection
│       │   │           └── input_guard.py # Thin delegate → router_node
│       │   │
│       │   ├── document_processing/
│       │   │   ├── parser.py              # PDF/DOCX/TXT parser
│       │   │   └── session_docs.py        # In-session document store
│       │   │
│       │   ├── guardrails/
│       │   │   ├── input_guardrails.py    # PII masking, injection detection, off-topic check
│       │   │   ├── output_guardrails.py   # ULIP notice, claim settlement disclaimer injection
│       │   │   └── guardrails_ai_service.py # Guardrails AI integration
│       │   │
│       │   ├── llm_gateway/
│       │   │   ├── client.py              # Portkey LLM client factory (smart/fast)
│       │   │   ├── embeddings.py          # Qwen3-Embedding-8B via Portkey
│       │   │   └── prompts.py             # System prompts + persona guidelines
│       │   │
│       │   ├── retrieval/
│       │   │   ├── classifier.py          # Heuristic query classifier (regex-based)
│       │   │   ├── qdrant_service.py      # Qdrant async client + ingestion pipeline
│       │   │   └── web_retrieval.py       # Domain-allowlisted web scraper
│       │   │
│       │   ├── tools/
│       │   │   └── actuarial_calculator.py # Loan / surrender / Section45 / grace calculations
│       │   │
│       │   └── http_clients/              # Shared async HTTP client (httpx)
│       │
│       ├── services/
│       │   └── query_service.py           # QueryService — orchestrates LangGraph execution
│       │
│       └── observability/
│           ├── logging.py                 # structlog + compliance audit logger
│           ├── logfire_setup.py           # Logfire telemetry initialization
│           └── audit.py                  # HIPAA audit trail writer
│
├── tests/
│   ├── unit/
│   │   ├── test_actuarial_tools.py        # 8 tests: loan, surrender, section45, grace
│   │   ├── test_agentic_rag_graph.py      # 7 tests: router node, graph routing, LLM mock
│   │   ├── test_auth.py                   # 4 tests: API key hashing, auth validation
│   │   ├── test_config.py                 # 3 tests: ENV config, DEV/PROD settings
│   │   ├── test_enterprise_standards.py   # 8 tests: SOLID, modular graph, API structure
│   │   ├── test_guardrails_ai.py          # 6 tests: jailbreak, PII, off-topic, API block
│   │   ├── test_logging.py                # 4 tests: PII masking (Aadhaar, PAN, phone, CC)
│   │   ├── test_saarthi_assistant.py      # 11 tests: intent routing, session, web retrieval
│   │   └── test_sarathi_ai_domain.py      # 8 tests: refusal phrase, statutory disclaimer, isolation
│   ├── integration/                       # Integration test suite (E2E against live stack)
│   └── evaluation/                        # RAG evaluation (RAGAS metrics)
│
├── .env                                   # Local environment variables (never commit)
├── .env.example                           # Environment variable template with documentation
├── pyproject.toml                         # Project metadata + dependency declarations
├── requirements.txt                       # Pinned runtime + dev dependencies (pip-compatible)
├── Makefile                               # Developer CLI shortcuts
├── Dockerfile                             # Production container build
├── docker-compose.yml                     # Local dev stack (App + Qdrant + Redis)
└── .python-version                        # Python 3.12 pin (for pyenv / uv)
```

---

## ⚙️ Configuration Reference

All settings are driven by environment variables. Copy `.env.example` to `.env` and configure:

### Core Application

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `DEV` | Runtime environment: `DEV` or `PROD` |
| `CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed frontend origins |

### LLM Provider

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | `groq` (DEV) or `bedrock` (PROD) |
| `PORTKEY_API_KEY` | — | Portkey account API key |
| `PORTKEY_VIRTUAL_KEY` | — | Active provider virtual key (`groq-vk-dev` in DEV, `bedrock-vk-prod` in PROD) |
| `PORTKEY_BASE_URL` | `https://api.portkey.ai/v1` | Portkey gateway base URL |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Unified model for synthesis, routing, and grading |

> **Note:** Groq retired `llama-3.3-70b-versatile` on August 16, 2026. Use `openai/gpt-oss-120b` via Portkey's `groq-vk-dev` virtual key for DEV environments.

### Embeddings

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_PROVIDER` | `qwen` | `qwen` (DEV) or `bedrock` (PROD) |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-8B` | Embedding model |
| `EMBEDDING_DIMENSIONS` | `4096` | Vector dimensions (must match Qdrant collection) |

### Vector Database (Qdrant)

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | — | Full Qdrant Cloud URL |
| `QDRANT_API_KEY` | — | API key for Qdrant Cloud |

### Redis Semantic Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |

### AWS (PROD)

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region for SDK calls |
| `AWS_ACCESS_KEY_ID` | — | AWS credentials (use IAM roles in PROD) |
| `AWS_SECRET_ACCESS_KEY` | — | AWS credentials |
| `S3_BUCKET_NAME` | `insurance-rag-dev-documents` | S3 bucket for document storage |
| `DYNAMODB_TABLE_PREFIX` | `insurance-rag-dev` | DynamoDB table name prefix |

### Authentication & User Accounts

Saarthi AI supports **Google Sign-In with signed JWT tokens** (HMAC-SHA256) alongside API Key authentication.

| Variable | Default | Description |
|----------|---------|-------------|
| `JWT_SECRET_KEY` | `saarthi-ai-enterprise-jwt-signing-secret-key-32chars` | Secret key used to sign and verify JWT access tokens |
| `JWT_ALGORITHM` | `HS256` | JWT signing algorithm (HS256) |
| `JWT_EXPIRATION_HOURS` | `24` | Expiration window for signed JWT access tokens |
| `DEV_API_KEY` | `dev-test-key-12345` | Accepted API key in DEV mode |
| `API_KEY_HASH_SALT` | `dev-salt-change-me` | Salt for PROD API key hashing |
| `RATE_LIMIT_PER_MINUTE` | `60` | Max requests per IP per minute |

#### Pre-configured Google Accounts

- **Employee**: `pratik@saarthi.com` (Pratik Chavan) — Role: `["employee"]`, Dept: Operations. Has full access to policy Q&A, streaming, citations, and personal Document Library.
- **Admin**: `saarthi.admin@saarthi.com` (Saarthi Admin) — Roles: `["admin", "kb_admin", "compliance_reviewer"]`, Dept: Compliance. Has complete access to the Admin & Compliance Console (`/admin/*`).
- **Guest Mode**: Unauthenticated visitors can query the knowledge base with ephemeral sessions without signing in.

### Observability

| Variable | Default | Description |
|----------|---------|-------------|
| `LOGFIRE_TOKEN` | — | Pydantic Logfire write token (`pylf_v1_...`) |
| `LOGFIRE_ENVIRONMENT` | `dev` | Environment label in Logfire |
| `LOGFIRE_SEND_TO_LOGFIRE` | `true` | Stream traces to Logfire cloud |
| `LANGCHAIN_TRACING_V2` | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | — | LangSmith API key |

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.12+**
- **Astral `uv`** package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Docker & Docker Compose** (for local Qdrant + Redis)

### 1. Clone & Install

```bash
# Clone the repository
git clone <repo-url>
cd rag-with-aws/saarthi-ai

# Install all dependencies (runtime + dev extras)
uv sync --all-extras

# Or install from requirements.txt (pip-compatible)
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy and edit the environment template
cp .env.example .env
```

**Minimum required for DEV:**
```bash
# .env (minimum DEV configuration)
ENV=DEV
LLM_PROVIDER=groq
PORTKEY_API_KEY=pk-...
PORTKEY_VIRTUAL_KEY=groq-vk-dev   # Create in Portkey dashboard
LLM_MODEL=openai/gpt-oss-120b
```

### 3. Start Infrastructure

```bash
# Start Qdrant Vector DB + Redis (detached)
make docker-up
# Equivalent to: docker compose up -d
```

Verify services are healthy:
```bash
curl http://localhost:6333/collections   # Qdrant
redis-cli ping                           # Redis → PONG
```

### 4. Start FastAPI Server

```bash
# Development server with hot reload (port 8000)
make dev
# Equivalent to: uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# Or bind to all interfaces (for Docker / remote access):
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Verify Backend is Running

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Test a query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-test-key-12345" \
  -d '{"query": "what is a ULIP policy ?"}'
```

### 6. Ingest Policy Documents

```bash
# Upload a PDF policy document to Qdrant
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "X-API-Key: dev-test-key-12345" \
  -F "file=@/path/to/policy.pdf" \
  -F "product_code=SMART-PLATINA-PLUS" \
  -F "document_type=master_policy"
```

---

## 📡 API Endpoints

All endpoints are available at both `/v1/...` and `/api/v1/...`.

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | None | Aggregate diagnostic health check (Qdrant, Redis, LLM) |
| `GET` | `/live` | None | Liveness probe (always 200 OK if app is running) |
| `GET` | `/ready` | None | Readiness probe (checks all dependency connections) |
| `GET` | `/` | None | Application info and version |
| `POST` | `/api/v1/query` | `X-API-Key` | Submit natural language policy question via LangGraph |
| `POST` | `/api/v1/ingest` | `X-API-Key` | Upload and index a policy document into Qdrant |
| `POST` | `/api/v1/session/attach` | `X-API-Key` | Attach a document for in-chat session RAG |
| `DELETE` | `/api/v1/session/{id}/clear` | `X-API-Key` | Clear all session document attachments |
| `GET` | `/api/v1/collections` | `X-API-Key` | List Qdrant collections and vector counts |

### Query Request Schema

```json
{
  "query": "What is the surrender penalty for Smart Platina Plus in year 3?",
  "product_code": "SMART-PLATINA-PLUS",
  "state": "MH",
  "session_id": "session-abc-123",
  "has_attachment": false,
  "include_citations": true
}
```

### Query Response Schema

```json
{
  "answer": "Based on the retrieved contract terms...",
  "query_type": "structured_rag",
  "citations": [
    {
      "document_id": "doc-001",
      "document_title": "Smart Platina Plus Policy Contract",
      "section": "Surrender Benefits",
      "clause_number": "7.3",
      "page_number": 12,
      "snippet": "In the third policy year, the surrender charge is...",
      "score": 0.92
    }
  ],
  "disclaimer": "Statutory Notice: ...",
  "latency_ms": 1842.5,
  "model_used": "openai/gpt-oss-120b"
}
```

**Authentication:**
- **DEV**: Include header `X-API-Key: dev-test-key-12345`
- **PROD**: Include header `X-API-Key: <hashed-production-key>` or `Authorization: Bearer <google-oauth-token>`

---

## 🧪 Testing

### Run All Unit Tests

```bash
make test-unit
# Equivalent to: uv run pytest tests/unit/ -v
```

### Run 24 E2E Comprehensive Scenarios & Scorecard

```bash
make test-scenarios
# Equivalent to: uv run python scripts/run_e2e_scenarios.py
```

### Run Integration Tests

```bash
make test-integration
# Equivalent to: uv run pytest tests/integration/ -v
```

### Run Full Test Suite with Coverage

```bash
make test
# Equivalent to: uv run pytest tests/ -v --cov=src/app --cov-report=term-missing
```

### Run Linter

```bash
make lint           # Check only
make format         # Auto-fix
```

### Run Type Checker

```bash
make typecheck      # uv run mypy src/
```

### Test Results (CI baseline)

```
61 passed, 51 warnings in 13.55s
```

| Test File | Tests | Coverage Area |
|-----------|-------|---------------|
| `test_actuarial_tools.py` | 8 | Policy loan, surrender value, Section 45, grace period |
| `test_agentic_rag_graph.py` | 7 | Router node, 4-taxonomy LLM classification, graph routing |
| `test_auth.py` | 4 | API key hashing, auth validation |
| `test_config.py` | 3 | Settings loading, DEV/PROD switching |
| `test_enterprise_standards.py` | 8 | SOLID principles, modular graph structure |
| `test_guardrails_ai.py` | 6 | Jailbreak blocking, PII masking, off-topic filtering |
| `test_logging.py` | 4 | PII pattern masking (Aadhaar, PAN, phone, CC) |
| `test_saarthi_assistant.py` | 11 | Intent routing, session docs, web retrieval |
| `test_sarathi_ai_domain.py` | 8 | Refusal phrases, statutory disclaimers, doc isolation |

---

## 🐳 Docker Deployment

### Local Development Stack

```bash
# Start all services (App + Qdrant + Redis)
make docker-up

# Stop all services
make docker-down

# View logs
docker compose logs -f app
docker compose logs -f qdrant
```

Services exposed:
| Service | Port | Purpose |
|---------|------|---------|
| FastAPI App | `:8000` | REST API |
| Qdrant HTTP | `:6333` | Vector DB HTTP API + Dashboard |
| Qdrant gRPC | `:6334` | gRPC interface |
| Redis | `:6379` | Semantic cache |

### Production Container

```bash
# Build production image
docker build -t saarthi-ai:latest .

# Run with production env
docker run -p 8000:8000 --env-file .env.prod saarthi-ai:latest
```

---

## 📊 Observability

### Logfire Tracing

The platform emits structured traces to [Pydantic Logfire](https://logfire.pydantic.dev/) with spans for:
- `graph.router_node` — intent classification with query and reasoning
- `graph.retrieval_node` — Qdrant query with latency and hit count
- `graph.synthesis_node` — LLM synthesis with token count
- `rag.agentic_pipeline` — end-to-end pipeline span per query

Set `LOGFIRE_TOKEN` and `LOGFIRE_SEND_TO_LOGFIRE=true` to stream to the cloud dashboard.

### Structured Logs

All logs use `structlog` JSON formatting with:
- `X-Request-ID` correlation across middleware chain
- `user_id` propagation via context variable
- Compliance audit events via `log_compliance_audit()`

### Health Check Response

```json
{
  "status": "healthy",
  "environment": "DEV",
  "version": "0.1.0",
  "llm_provider": "groq",
  "embedding_provider": "qwen",
  "components": {
    "qdrant": { "status": "healthy", "latency_ms": 4.2 },
    "redis": { "status": "healthy", "latency_ms": 1.1 },
    "llm_gateway": { "status": "healthy", "message": "Configured for groq (model: openai/gpt-oss-120b)" },
    "embedding_provider": { "status": "healthy", "message": "Configured for qwen (4096d)" }
  }
}
```

---

## 🔐 Security & Guardrails

### Input Security Layers
1. **Rate Limiting** — 60 requests/minute per IP (sliding window)
2. **API Key Authentication** — SHA-256 hashed key validation; Google OAuth Bearer in PROD
3. **Prompt Injection Detection** — Regex + pattern-based jailbreak interception
4. **PII Scrubbing** — Microsoft Presidio: Aadhaar (12-digit), PAN card, +91 mobile, email, credit card
5. **Off-Topic Guardrail** — Scoped to life insurance domain; hard refusal for unrelated queries

### Output Security Layers
1. **ULIP Market Risk Notice** — Mandatory statutory IRDAI disclosure injected for all ULIP responses
2. **Section 45 Notice** — Incontestability advisory for claim-related responses
3. **Prompt Injection Defense** — Context chunks treated as inert DATA, never as instructions
4. **Stack Trace Protection** — Global exception handler prevents internal detail disclosure

### Security Architecture Note
In production, replace `DEV_API_KEY` with a DynamoDB-backed API key management system. Never commit `.env` to version control.

---

## 🗺️ Roadmap

- [ ] AWS Bedrock integration (Claude Sonnet 4 + Titan Embeddings) for PROD
- [ ] RAGAS evaluation pipeline for RAG quality metrics
- [ ] Multi-language support (Hindi, Marathi) for regional agent desks
- [ ] Real-time IRDAI circular monitoring with webhook ingestion
- [ ] DynamoDB API key management with per-user rate limits
- [ ] Kubernetes Helm chart for cloud-native deployment

---

## ⚖️ License

Internal SBI Life & Statutory IRDAI Educational Reference. Not for public distribution.
