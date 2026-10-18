# Movie Recommender - build, test and run automation.
# Local targets use uv and read Neo4j credentials from .env (never committed;
# copy .env.example). Docker targets drive the compose stack.

.DEFAULT_GOAL := help
COMPOSE := docker compose
DOCKER_USER ?= your-dockerhub-user
IMAGE := $(DOCKER_USER)/movie-recommender-api
# Pass .env to uv only when it exists: a fresh clone must still run make test
# (integration tests skip themselves when NEO4J_TEST_URI is unset).
ENV_FILE := $(if $(wildcard .env),--env-file .env)

.PHONY: help install run seed gds test lint benchmark \
        docker-build docker-run docker-stop docker-push clean

help: ## List the available targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Create .venv and install the locked dependencies
	uv sync

run: ## Run the API locally with reload (needs a running Neo4j and .env)
	uv run $(ENV_FILE) uvicorn app.main:app --reload

seed: ## Load ml-latest-small into the Neo4j of .env (downloads it if needed)
	uv run python scripts/download_data.py
	uv run $(ENV_FILE) python -m scripts.seed_data

gds: ## Compute SIMILAR, communities and PageRank (run after every seed)
	uv run $(ENV_FILE) python -m scripts.run_gds

test: ## Run the test suite with coverage (integration tests need NEO4J_TEST_URI)
	uv run $(ENV_FILE) pytest --cov=app --cov=scripts --cov-report=term-missing

lint: ## Pylint over app, scripts and tests; fails under 9.5
	uv run pylint app scripts tests --fail-under 9.5

benchmark: ## Latency per endpoint against the database of .env
	uv run $(ENV_FILE) python -m scripts.benchmark --repeats 50 --warmup 5

docker-build: ## Build the API image
	docker build -t $(IMAGE) .

docker-run: ## Start the full stack (Neo4j + API + Nginx) in the background
	$(COMPOSE) up --build -d

docker-stop: ## Stop the stack (data volume is kept)
	$(COMPOSE) down

docker-push: ## Push the API image to Docker Hub (needs docker login, set DOCKER_USER=you)
	docker push $(IMAGE)

clean: ## Remove caches and coverage artifacts
	rm -rf .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
