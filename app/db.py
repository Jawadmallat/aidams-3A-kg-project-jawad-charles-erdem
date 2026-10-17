"""Access to the Neo4j driver owned by the FastAPI application.

The driver is created (or injected) by the lifespan in app.main and stored on
`app.state.driver`; routers receive it through the `get_driver` dependency, so
tests can inject a driver for a disposable database instead.
"""

from fastapi import Request
from neo4j import Driver


def get_driver(request: Request) -> Driver:
    """Return the application-wide Neo4j driver (FastAPI dependency)."""
    return request.app.state.driver
