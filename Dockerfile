# API image: uv installs the locked runtime dependencies, uvicorn serves.
# scripts/ ships too: the app imports scripts.db and scripts.movielens, and it
# lets the stack be seeded with `docker compose exec api uv run python -m
# scripts.seed_data` without a local Python.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1

# Dependency layer first, so code edits do not reinstall anything.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY app/ app/
COPY scripts/ scripts/

EXPOSE 8000

# The slim image has no curl; the healthcheck probes with the venv's python.
HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=15s \
  CMD ["/app/.venv/bin/python", "-c", \
       "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=3)"]

CMD ["uv", "run", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
