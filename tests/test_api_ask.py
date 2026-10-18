"""HTTP behaviour of POST /api/ask that holds without a database or a key.

The LLM is always stubbed (CI has no key) and the driver is a stub that
fails like an unreachable server, so every status code the endpoint can
answer is pinned down here; the real Neo4j layers are in
test_ask_integration.py.
"""

import pytest
from fastapi.testclient import TestClient
from neo4j.exceptions import ServiceUnavailable

from app import llm
from app.main import create_app
from app.models import CypherDraft
from app.services import ask as service


class DeadDriver:
    """A driver whose server is down: sessions and queries fail."""

    def session(self, **_kwargs):
        """Opening a session fails like an unreachable server."""
        raise ServiceUnavailable("down")

    def execute_query(self, *_args, **_kwargs):
        """Queries fail like an unreachable server."""
        raise ServiceUnavailable("down")


@pytest.fixture(name="client")
def fixture_client(monkeypatch):
    """A client over the app with a dead database and the LLM switched on."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    return TestClient(create_app(driver=DeadDriver()), raise_server_exceptions=False)


def answer_with(monkeypatch, draft: CypherDraft) -> None:
    """Make the stubbed model answer `draft` to any question."""
    monkeypatch.setattr(llm, "complete_json", lambda *_: draft)


def test_without_a_key_ask_is_503_and_says_how_to_fix_it(monkeypatch):
    """No LLM_API_KEY: a clear 503, not a crash; the rest of the API is unaffected."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    client = TestClient(create_app(driver=DeadDriver()), raise_server_exceptions=False)
    response = client.post("/api/ask", json={"question": "Which movies are popular?"})
    assert response.status_code == 503
    assert "LLM_API_KEY" in response.json()["detail"]
    assert client.get("/api/health").status_code == 200


@pytest.mark.parametrize(
    "body", [{}, {"question": ""}, {"question": "hi"}, {"question": "x" * 501}, {"q": "x"}]
)
def test_invalid_bodies_are_422(client, body):
    """Missing, too short or too long questions fail validation first."""
    assert client.post("/api/ask", json=body).status_code == 422


def test_a_write_draft_is_400_with_the_query_and_the_failed_checks(client, monkeypatch):
    """The guardrail rejection of the live demo: what the model wrote, and why not."""
    answer_with(monkeypatch, CypherDraft(
        cypher="MATCH (u:User) DETACH DELETE u", explanation="Deletes all users."
    ))
    response = client.post("/api/ask", json={"question": "Remove all the users"})
    assert response.status_code == 400
    body = response.json()
    assert body["cypher"] == "MATCH (u:User) DETACH DELETE u"
    assert body["explanation"] == "Deletes all users."
    failed = {check["name"]: check["detail"] for check in body["checks"] if not check["passed"]}
    assert failed["read_only_keywords"] == "forbidden: DELETE, DETACH"


def test_a_hallucinated_label_is_400(client, monkeypatch):
    """A label outside the schema is rejected before reaching the database."""
    answer_with(monkeypatch, CypherDraft(
        cypher="MATCH (d:Director)-[:DIRECTED]->(m:Movie) RETURN d.name", explanation="x"
    ))
    response = client.post("/api/ask", json={"question": "Who directed Heat?"})
    assert response.status_code == 400
    failed = [check for check in response.json()["checks"] if not check["passed"]]
    assert failed == [{"name": "known_schema", "passed": False,
                       "detail": "not in the schema: Director, DIRECTED"}]


def test_a_provider_failure_is_502(client, monkeypatch):
    """A failing or off-schema LLM answer is a bad gateway, not a 500."""
    def failing(*_):
        raise llm.LLMError("LLM provider error: APITimeoutError")
    monkeypatch.setattr(llm, "complete_json", failing)
    response = client.post("/api/ask", json={"question": "Which movies are popular?"})
    assert response.status_code == 502
    assert response.json() == {"detail": "LLM provider error: APITimeoutError"}


def test_a_slow_query_is_504(client, monkeypatch):
    """A query stopped by the transaction timeout answers 504."""
    def slow(*_):
        raise service.QueryTimeout("query exceeded 10 s")
    monkeypatch.setattr(service, "answer", slow)
    response = client.post("/api/ask", json={"question": "Which movies are popular?"})
    assert response.status_code == 504
    assert response.json() == {"detail": "query exceeded 10 s"}


def test_a_dead_database_is_503(client, monkeypatch):
    """A valid draft against an unreachable Neo4j maps to the app-wide 503."""
    answer_with(monkeypatch, CypherDraft(
        cypher="MATCH (m:Movie) RETURN m.title AS title LIMIT 3", explanation="x"
    ))
    response = client.post("/api/ask", json={"question": "Name three movies"})
    assert response.status_code == 503
    assert response.json() == {"detail": "database unavailable"}


def test_a_declined_question_is_200_with_no_query(client, monkeypatch):
    """The model may decline: 200, answered false, and nothing ran."""
    answer_with(monkeypatch, CypherDraft(cypher="", explanation="No box office data."))
    response = client.post("/api/ask", json={"question": "What did Titanic earn?"})
    assert response.status_code == 200
    body = response.json()
    assert (body["answered"], body["cypher"], body["rows"]) == (False, None, [])
    assert body["explanation"] == "No box office data."


def test_ask_is_documented_in_openapi(client):
    """Swagger shows the endpoint with its error codes."""
    operation = client.get("/openapi.json").json()["paths"]["/api/ask"]["post"]
    assert {"200", "400", "422", "502", "503", "504"} <= set(operation["responses"])
