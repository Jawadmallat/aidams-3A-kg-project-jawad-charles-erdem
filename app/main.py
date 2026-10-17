"""Application factory: driver lifecycle, error mapping and the health endpoint.

Run locally with:
    uv run --env-file .env uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from neo4j import Driver
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from app.routers import movies
from scripts.db import get_driver


def create_app(driver: Driver | None = None) -> FastAPI:
    """Build the FastAPI app.

    Without arguments the lifespan creates a driver from NEO4J_URI, NEO4J_USER
    and NEO4J_PASSWORD and closes it on shutdown. Tests and scripts pass their
    own driver instead; the caller keeps ownership and closes it itself.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        owned = driver is None
        if owned:
            application.state.driver = get_driver()
        yield
        if owned:
            application.state.driver.close()

    application = FastAPI(
        title="Movie Recommender",
        description="Graph-backed movie recommendations over Neo4j (MovieLens).",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.exception_handler(ServiceUnavailable)
    @application.exception_handler(SessionExpired)
    async def database_down(_request: Request, _error: Exception) -> JSONResponse:
        """Map Neo4j connectivity failures to 503 instead of a 500 traceback."""
        return JSONResponse(status_code=503, content={"detail": "database unavailable"})

    @application.get("/api/health", tags=["health"])
    async def health() -> dict[str, str]:
        """Liveness probe for container healthchecks; does not touch the database."""
        return {"status": "ok"}

    if driver is not None:
        # An injected driver is usable immediately, without waiting for the
        # lifespan; the owner closes it (tests, benchmark).
        application.state.driver = driver

    application.include_router(movies.router)
    return application


app = create_app()
