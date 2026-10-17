"""Shared pytest fixtures."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from neo4j.exceptions import AuthError, ServiceUnavailable

from app.main import create_app
from scripts.db import get_driver
from scripts.run_gds import run_gds
from scripts.seed_data import read_dataset, reset_graph, seed

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "movielens"


@pytest.fixture(scope="session", name="neo4j_driver")
def fixture_neo4j_driver():
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


@pytest.fixture(scope="session", name="fixture_frames")
def fixture_fixture_frames():
    """The committed MovieLens test fixture as DataFrames."""
    return read_dataset(FIXTURE_DIR)


@pytest.fixture(scope="session")
def seeded_graph(neo4j_driver, fixture_frames):
    """The test database wiped and loaded with the fixture, once per test session."""
    reset_graph(neo4j_driver)
    seed(neo4j_driver, fixture_frames)
    return neo4j_driver


@pytest.fixture(scope="session", name="recommender_graph")
def fixture_recommender_graph(seeded_graph):  # pylint: disable=redefined-outer-name
    """The seeded graph with GDS results written.

    pytest injects the seeded_graph fixture through the parameter name, which
    pylint reads as shadowing; the disable is scoped to this line. Safe at
    session scope: run_gds replaces its previous output and is deterministic
    on the same graph (proven by the GDS integration tests).
    """
    run_gds(seeded_graph)
    return seeded_graph


@pytest.fixture(scope="session", name="api_client")
def fixture_api_client(recommender_graph):
    """A TestClient over the app wired to the seeded test database."""
    application = create_app(driver=recommender_graph)
    with TestClient(application) as client:
        yield client
