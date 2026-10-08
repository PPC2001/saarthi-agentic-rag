# ==============================================================================
# Production Multi-Stage Dockerfile for Saarthi AI - Life Insurance Assistant
# Following enterprise production standards (non-root execution, minimal attack surface)
# ==============================================================================

# ── Stage 1: Dependency Builder ──
FROM python:3.12-slim AS builder

WORKDIR /build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency configuration
COPY saarthi-ai/pyproject.toml saarthi-ai/uv.lock* ./

# Install production dependencies into a virtual environment
RUN uv sync --frozen --no-dev --no-editable


# ── Stage 2: Final Production Runtime ──
FROM python:3.12-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    PATH="/app/.venv/bin:$PATH" \
    PORT=8000

# Install runtime dependencies for healthchecks & document processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root system user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appgroup /build/.venv /app/.venv

# Copy backend source code
COPY --chown=appuser:appgroup saarthi-ai/src /app/src

# Set permissions and switch to non-root user
USER appuser

EXPOSE 8000

# Production Healthcheck Probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
