# RxVision2025 Development Automation
# Professional pharmaceutical computer vision system

.PHONY: help install install-dev install-gpu clean lint format type-check security test test-fast test-integration test-gpu coverage build docs serve-docs docker-build docker-run benchmark data-download data-synthetic validate-model export-models release

# Default target
help: ## Show this help message
	@echo "RxVision2025 Development Commands"
	@echo "================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment setup
install: ## Install production dependencies
	python -m pip install --upgrade pip
	pip install -e .

install-dev: ## Install development dependencies
	python -m pip install --upgrade pip
	pip install -e ".[dev]"
	pre-commit install

install-gpu: ## Install GPU dependencies
	python -m pip install --upgrade pip
	pip install -e ".[gpu]"

install-data: ## Install data processing dependencies
	python -m pip install --upgrade pip
	pip install -e ".[data]"

install-all: ## Install all dependencies
	python -m pip install --upgrade pip
	pip install -e ".[dev,gpu,data]"
	pre-commit install

# Code quality
clean: ## Clean up temporary files and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ dist/ build/ .tox/ .coverage.*

lint: ## Run all linting checks
	pre-commit run --all-files

format: ## Format code with black and isort
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

type-check: ## Run type checking with mypy
	mypy src/ --config-file pyproject.toml

security: ## Run security checks
	bandit -r src/ -f json -o reports/bandit-report.json
	bandit -r src/ -f txt
	safety check --json --output reports/safety-report.json
	safety check

# Testing
test: ## Run all tests with coverage
	pytest tests/ -v --cov=src/rxvision2025 --cov-report=xml --cov-report=html --cov-report=term-missing --junit-xml=reports/pytest-results.xml

test-fast: ## Run fast tests only (exclude slow and GPU tests)
	pytest tests/ -v -m "not slow and not gpu" --junit-xml=reports/pytest-fast.xml

test-integration: ## Run integration tests
	pytest tests/ -v -m "integration" --junit-xml=reports/pytest-integration.xml

test-gpu: ## Run GPU tests (requires GPU)
	pytest tests/ -v -m "gpu" --junit-xml=reports/pytest-gpu.xml

coverage: ## Generate coverage report
	pytest tests/ --cov=src/rxvision2025 --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

# Building and distribution
build: ## Build package for distribution
	python -m build
	twine check dist/*

upload-test: ## Upload to test PyPI
	twine upload --repository testpypi dist/*

upload: ## Upload to PyPI
	twine upload dist/*

# Documentation
docs: ## Build documentation
	sphinx-build -b html docs/ docs/_build/html

docs-clean: ## Clean documentation build
	rm -rf docs/_build/

serve-docs: docs ## Build and serve documentation locally
	cd docs/_build/html && python -m http.server 8000

# Docker operations
docker-build: ## Build Docker image
	docker build -t rxvision2025:latest .

docker-build-dev: ## Build development Docker image
	docker build -f Dockerfile.dev -t rxvision2025:dev .

docker-run: ## Run Docker container
	docker run --rm -p 8000:8000 rxvision2025:latest

docker-run-dev: ## Run development Docker container
	docker run --rm -it -v $(PWD):/app -p 8000:8000 rxvision2025:dev bash

docker-test: ## Run tests in Docker
	docker run --rm rxvision2025:latest python -m pytest tests/

# Data management
data-download: ## Download real pharmaceutical dataset
	python scripts/download_data_modern.py --full --validate

data-sample: ## Download sample dataset for testing
	python scripts/download_data_modern.py --sample --classes 10

data-synthetic: ## Create synthetic dataset for development
	python scripts/download_data_modern.py --synthetic --classes 5 --images-per-class 100

data-clean: ## Clean downloaded data
	rm -rf data/raw/ data/processed/
	mkdir -p data/raw data/processed

# Model operations
train: ## Train model with default configuration
	python scripts/train_model.py --config configs/default.yaml

train-gpu: ## Train model on GPU
	python scripts/train_model.py --config configs/gpu.yaml --device gpu

validate-model: ## Validate trained model performance
	python scripts/validate_model.py --data-dir data/ --min-accuracy 0.8

export-models: ## Export trained models to different formats
	python scripts/export_models.py --output-dir models/ --formats tensorflow,onnx,tflite

benchmark: ## Run performance benchmarks
	python scripts/run_benchmarks.py --comprehensive --output reports/benchmarks.json

# API operations
serve: ## Start inference API server
	uvicorn src.inference.api:app --host 0.0.0.0 --port 8000 --reload

serve-prod: ## Start production API server
	uvicorn src.inference.api:app --host 0.0.0.0 --port 8000

# Development workflow
dev-setup: install-dev data-sample ## Complete development environment setup
	@echo "Development environment ready!"
	@echo "Run 'make test-fast' to verify installation"

ci-check: lint type-check security test ## Run all CI checks locally
	@echo "All CI checks passed!"

pre-commit: format lint type-check ## Run pre-commit checks
	pre-commit run --all-files

# Monitoring and profiling
profile: ## Profile model performance
	python scripts/profile_model.py --output reports/profile.json

monitor: ## Start monitoring dashboard
	python scripts/monitoring_dashboard.py

# Release workflow
version-patch: ## Bump patch version
	bumpversion patch

version-minor: ## Bump minor version
	bumpversion minor

version-major: ## Bump major version
	bumpversion major

release-check: clean lint type-check security test build ## Pre-release validation
	@echo "Release candidate ready!"

# Research and experimentation
experiment: ## Run experiment with MLflow tracking
	python scripts/run_experiment.py --config configs/experiment.yaml

tensorboard: ## Launch TensorBoard
	tensorboard --logdir=logs/

jupyter: ## Start Jupyter notebook server
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Utilities
requirements: ## Generate requirements.txt from pyproject.toml
	pip-compile pyproject.toml --output-file requirements.txt
	pip-compile pyproject.toml --extra dev --output-file requirements-dev.txt

check-deps: ## Check for dependency updates
	pip list --outdated

update-deps: ## Update dependencies
	pip-compile --upgrade pyproject.toml --output-file requirements.txt
	pip-compile --upgrade pyproject.toml --extra dev --output-file requirements-dev.txt

security-scan: ## Comprehensive security scan
	bandit -r src/ -f json -o reports/bandit-report.json
	safety check --json --output reports/safety-report.json
	pip-audit --format=json --output=reports/pip-audit.json

# Database operations (if applicable)
db-migrate: ## Run database migrations
	alembic upgrade head

db-seed: ## Seed database with sample data
	python scripts/seed_database.py

# Cloud operations
deploy-staging: ## Deploy to staging environment
	@echo "Deploying to staging..."
	# Add your staging deployment commands here

deploy-prod: ## Deploy to production environment
	@echo "Deploying to production..."
	# Add your production deployment commands here

# Performance testing
load-test: ## Run load tests on API
	locust -f tests/load_test.py --host=http://localhost:8000

stress-test: ## Run stress tests
	python scripts/stress_test.py

# Backup and restore
backup: ## Backup models and data
	tar -czf backups/rxvision-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz models/ data/ configs/

restore: ## Restore from backup (specify BACKUP_FILE)
	tar -xzf $(BACKUP_FILE)

# Health checks
health-check: ## Check system health
	python scripts/health_check.py

status: ## Show project status
	@echo "RxVision2025 Project Status"
	@echo "==========================="
	@echo "Git branch: $(shell git branch --show-current 2>/dev/null || echo 'unknown')"
	@echo "Git status: $(shell git status --porcelain 2>/dev/null | wc -l | tr -d ' ') files changed"
	@echo "Python version: $(shell python --version)"
	@echo "Package installed: $(shell pip show rxvision2025 >/dev/null 2>&1 && echo 'Yes' || echo 'No')"
	@echo "Pre-commit installed: $(shell pre-commit --version >/dev/null 2>&1 && echo 'Yes' || echo 'No')"
	@echo "Data directory: $(shell [ -d data ] && echo 'Exists' || echo 'Missing')"
	@echo "Models directory: $(shell [ -d models ] && echo 'Exists' || echo 'Missing')"

# Directory setup
setup-dirs: ## Create necessary directories
	mkdir -p data/{raw,processed,train,val,test}
	mkdir -p models/{checkpoints,exports}
	mkdir -p logs/{training,inference}
	mkdir -p reports/{coverage,security,benchmarks}
	mkdir -p configs/{training,inference,experiments}
	mkdir -p backups
	@echo "Project directories created"

# All-in-one commands
fresh-start: clean setup-dirs install-dev data-sample ## Complete fresh environment setup
	@echo "Fresh development environment ready!"

full-ci: clean install-dev lint type-check security test build docs ## Full CI pipeline locally
	@echo "Full CI pipeline completed successfully!"