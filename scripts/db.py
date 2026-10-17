"""Neo4j connection settings shared by the data scripts."""

import os

from neo4j import Driver, GraphDatabase


def get_driver(uri: str | None = None) -> Driver:
    """Create a driver from NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD.

    Run the scripts with `uv run --env-file .env ...` to load them from .env.
    """
    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("NEO4J_PASSWORD is not set (copy .env.example to .env)")
    return GraphDatabase.driver(
        uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), password),
    )
