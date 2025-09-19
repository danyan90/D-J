# Makefile for D-J Project
# Convenient commands for development tasks

.PHONY: help install install-dev test test-cov lint format type-check clean build docs serve-docs

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e .
	pip install -r requirements-dev.txt
	pre-commit install

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage report
	pytest --cov=dj_project --cov-report=html --cov-report=term

lint:  ## Run all linting tools
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:  ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

type-check:  ## Run type checking with mypy
	mypy src/

quality: format lint type-check  ## Run all code quality checks

clean:  ## Clean up build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:  ## Build the package
	python -m build

demo:  ## Run the CLI demo
	dj-cli demo

setup-dev:  ## Complete development environment setup
	python -m venv venv || true
	@echo "Virtual environment created. Activate it with:"
	@echo "  source venv/bin/activate  # On Linux/Mac"
	@echo "  venv\\Scripts\\activate     # On Windows"
	@echo "Then run: make install-dev"

check-deps:  ## Check for outdated dependencies
	pip list --outdated

update-deps:  ## Update dependencies (be careful!)
	pip install --upgrade -r requirements.txt
	pip install --upgrade -r requirements-dev.txt

# Development workflow commands
dev-check: format lint type-check test  ## Run full development check (format, lint, type-check, test)

pre-commit:  ## Run pre-commit hooks manually
	pre-commit run --all-files

release-check: clean build test-cov lint type-check  ## Check if ready for release

# Quick development commands
quick-test:  ## Run tests without coverage (faster)
	pytest -x

watch-test:  ## Run tests in watch mode (requires pytest-watch)
	ptw -- -x

# Documentation commands (if using Sphinx in the future)
docs:  ## Generate documentation (placeholder)
	@echo "Documentation generation not yet implemented"
	@echo "See docs/ directory for manual documentation"

serve-docs:  ## Serve documentation locally (placeholder)
	@echo "Documentation serving not yet implemented"
	@echo "Open docs/README.md in your browser"