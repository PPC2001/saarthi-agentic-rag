# ==============================================================================
# Makefile — Life Insurance RAG Platform Developer Shortcuts
# ==============================================================================

.PHONY: help install dev ui lint format typecheck test test-unit test-integration docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make install        - Install all dependencies with UV"
	@echo "  make dev            - Run FastAPI dev server (127.0.0.1:8000)"
	@echo "  make ui             - Run Streamlit UI (127.0.0.1:8501)"
	@echo "  make lint           - Run Ruff linter"
	@echo "  make format         - Format code with Ruff"
	@echo "  make typecheck      - Run mypy strict type checker"
	@echo "  make test           - Run full test suite with coverage"
	@echo "  make docker-up      - Start local stack (App + Qdrant + Redis)"
	@echo "  make docker-down    - Stop local stack"
	@echo "  make clean          - Remove temporary files and caches"

install:
	uv sync --all-extras

dev:
	uv run uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

test:
	uv run pytest tests/ -v --cov=src/app --cov-report=term-missing

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-scenarios:
	uv run python scripts/run_e2e_scenarios.py

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov/
