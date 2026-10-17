"""API error handling that must hold without any database.

These tests run everywhere (no NEO4J_TEST_URI needed): the app is built with a
stub driver whose every query fails the way the neo4j driver fails when the
server is unreachable. Endpoint-specific 404 behaviour lives with the endpoint
tests, which need real data.
"""

from fastapi.testclient import TestClient
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from app.main import create_app


class FailingDriver:
    """Stands in for a neo4j Driver whose server is down."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def execute_query(self, *_args, **_kwargs):
        """Raise the configured connectivity error, like a dead server would."""
        raise self.error


def client_with_dead_database(error: Exception | None = None) -> TestClient:
    """A TestClient over an app whose driver raises on every query."""
    application = create_app(driver=FailingDriver(error or ServiceUnavailable("down")))
    return TestClient(application, raise_server_exceptions=False)


def test_health_answers_without_database():
    """/api/health is a liveness probe: 200 even when every query would fail."""
    response = client_with_dead_database().get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_route_is_404():
    """Routes outside the API surface return 404, not an error from the driver."""
    assert client_with_dead_database().get("/api/nope").status_code == 404


def test_similar_validation_rejects_bad_parameters():
    """Validation answers 422 before any query runs (the driver here would raise)."""
    client = client_with_dead_database()
    for url in (
        "/api/movies/abc/similar",      # non-integer id
        "/api/movies/1/similar?limit=0",    # below ge=1
        "/api/movies/1/similar?limit=999",  # above le=50
    ):
        assert client.get(url).status_code == 422, url


def test_similar_with_dead_database_is_503():
    """The real endpoint maps a dead database to 503."""
    response = client_with_dead_database().get("/api/movies/1/similar")
    assert response.status_code == 503
    assert response.json() == {"detail": "database unavailable"}


def make_probe(error: Exception):
    """A parameterless route body standing in for any endpoint that queries Neo4j."""
    async def probe():
        raise error
    return probe


def test_connectivity_errors_map_to_503():
    """ServiceUnavailable and SessionExpired surface as 503, not a 500 traceback."""
    for error in (ServiceUnavailable("down"), SessionExpired("gone")):
        application = create_app(driver=FailingDriver(error))
        application.get("/api/_probe")(make_probe(error))

        response = TestClient(application, raise_server_exceptions=False).get("/api/_probe")
        assert response.status_code == 503, f"{type(error).__name__} was not mapped"
        assert response.json() == {"detail": "database unavailable"}
