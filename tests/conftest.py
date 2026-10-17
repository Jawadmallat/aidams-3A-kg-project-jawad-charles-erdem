"""Shared pytest fixtures."""

import os

import pytest
from neo4j.exceptions import AuthError, ServiceUnavailable

from scripts.db import get_driver


@pytest.fixture(scope="session")
def neo4j_driver():
    """Driver for a disposable test database, or skip when none is configured.

    Integration tests only run when NEO4J_TEST_URI is set. It is kept separate
    from NEO4J_URI on purpose: these tests may wipe the database they point at.
    """
    uri = os.environ.get("NEO4J_TEST_URI")
    if not uri:
        pytest.skip("NEO4J_TEST_URI not set")
    driver = get_driver(uri)
    try:
        driver.verify_connectivity()
    except (ServiceUnavailable, AuthError) as error:
        driver.close()
        pytest.skip(f"Neo4j test database unreachable: {error}")
    yield driver
    driver.close()
