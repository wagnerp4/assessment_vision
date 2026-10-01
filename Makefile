.PHONY: install install-dev test lint format clean run help

help:
	@echo "Assessment App - Development Tasks"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install       Install dependencies"
	@echo "  install-dev   Install dev dependencies"
	@echo "  test          Run tests"
	@echo "  test-cov      Run tests with coverage"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code with black"
	@echo "  clean         Clean up cache and temp files"
	@echo "  run           Run the app"
	@echo ""

install:
	uv sync

install-dev:
	uv sync --all-extras

test:
	uv run pytest tests/ -v

test-cov:
	uv run pytest tests/ -v --cov=src/assessment_app --cov-report=term-missing

lint:
	uv run ruff check src/ scripts/
	uv run black --check src/ scripts/

format:
	uv run black src/ scripts/
	uv run ruff check --fix src/ scripts/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info .pytest_cache .coverage htmlcov
	rm -rf .mypy_cache .ruff_cache

run:
	uv run python -m assessment_app.app
