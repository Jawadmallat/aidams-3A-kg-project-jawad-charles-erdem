"""The server-side layers of /api/ask against a real Neo4j (fixture graph).

Layers 2 and 3 are only worth something if Neo4j enforces them on its own,
so these tests feed them queries the static guardrails would have stopped
and check the database refuses them too. The LLM is still a stub. Skipped
without NEO4J_TEST_URI, like every integration test.
"""

import pytest
from neo4j.exceptions import ClientError

from app import llm
from app.models import CypherDraft, MovieRef
from app.services import ask, guardrails
from app.services.guardrails import Validation


def checks_by_name(checks) -> dict[str, bool]:
    """Verdicts keyed by check name."""
    return {check.name: check.passed for check in checks}


def test_explain_passes_a_valid_read_query(recommender_graph):
    """A correct query compiles, is read-only and raises no blocking warning."""
    checks = ask.explain(
        recommender_graph, "MATCH (m:Movie {movieId: $id}) RETURN m.title AS title", {"id": 1}
    )
    assert checks_by_name(checks) == {
        "explain_dry_run": True, "read_only_plan": True, "database_warnings": True,
    }


def test_explain_classifies_a_write_as_not_read_only(recommender_graph):
    """Layer 2 alone spots a write: Neo4j reports the plan type, nothing runs."""
    checks = ask.explain(recommender_graph, "MATCH (m:Movie) SET m.flag = true RETURN m", {})
    assert checks_by_name(checks)["read_only_plan"] is False
    flags = recommender_graph.execute_query(
        "MATCH (m:Movie) WHERE m.flag IS NOT NULL RETURN count(m) AS n"
    ).records[0]["n"]
    assert flags == 0


def test_explain_reports_a_syntax_error(recommender_graph):
    """A query that does not compile fails the dry run with Neo4j's code."""
    checks = ask.explain(recommender_graph, "MATCH (m:Movie) RETURN m.title LIMT 3", {})
    assert len(checks) == 1
    assert checks[0].name == "explain_dry_run"
    assert not checks[0].passed
    assert "SyntaxError" in checks[0].detail


@pytest.mark.parametrize(
    ("cypher", "params", "warning"),
    [
        ("MATCH (m:Movie) RETURN m.director AS d", {}, "director"),
        ("MATCH (m:Movie {movieId: $id}) RETURN m.title", {}, "parameter"),
    ],
)
def test_explain_blocks_unknown_properties_and_missing_parameters(
    recommender_graph, cypher, params, warning
):
    """Hallucinated property keys and unbound parameters fail the warnings check."""
    checks = {check.name: check for check in ask.explain(recommender_graph, cypher, params)}
    assert checks["database_warnings"].passed is False
    assert warning in checks["database_warnings"].detail


def test_execute_returns_rows(recommender_graph):
    """Layer 3 runs a read query and returns plain JSON rows."""
    rows, truncated = ask.execute(
        recommender_graph,
        "MATCH (m:Movie) WHERE m.movieId IN $ids RETURN m.title AS title, m.genres AS genres "
        "ORDER BY m.movieId",
        {"ids": [1, 2571]},
    )
    assert [row["title"] for row in rows] == ["Toy Story", "The Matrix"]
    assert isinstance(rows[0]["genres"], list)
    assert truncated is False


def test_execute_refuses_a_write_on_its_own(recommender_graph):
    """Even with layers 1 and 2 skipped, the read transaction cannot write."""
    with pytest.raises(ClientError) as caught:
        ask.execute(recommender_graph, "CREATE (:Movie {movieId: -1}) RETURN 1", {})
    assert "AccessMode" in caught.value.code
    assert not recommender_graph.execute_query(
        "MATCH (m:Movie {movieId: -1}) RETURN m"
    ).records


def test_execute_caps_rows_beyond_the_final_limit(recommender_graph):
    """A UNION can exceed its last LIMIT; execution still stops at MAX_LIMIT."""
    rows, truncated = ask.execute(
        recommender_graph,
        "UNWIND range(1, 150) AS x RETURN x UNION UNWIND range(151, 300) AS x RETURN x",
        {},
    )
    assert len(rows) == guardrails.MAX_LIMIT
    assert truncated is True


def test_execute_stops_a_slow_query(recommender_graph, monkeypatch):
    """The transaction timeout aborts a long query server-side."""
    monkeypatch.setattr(ask, "QUERY_TIMEOUT_S", 0.2)
    with pytest.raises(ClientError) as caught:
        ask.execute(
            recommender_graph,
            "UNWIND range(1, 100000000) AS x WITH x WHERE x < 0 RETURN count(x) AS n", {},
        )
    assert "TransactionTimedOut" in caught.value.code


def test_layer_two_catches_what_layer_one_would_miss(recommender_graph, monkeypatch):
    """With the static guardrails switched off, a write is still never run."""
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "m")
    draft = CypherDraft(cypher="MATCH (m:Movie {movieId: 1}) SET m.title = 'x' RETURN m",
                        explanation="x")
    monkeypatch.setattr(llm, "complete_json", lambda *_: draft)
    monkeypatch.setattr(
        ask, "validate", lambda cypher, params: Validation(cypher, params or {}, [])
    )
    with pytest.raises(ask.QueryRejected) as caught:
        ask.answer(recommender_graph, "rename Toy Story")
    assert checks_by_name(caught.value.checks)["read_only_plan"] is False
    title = recommender_graph.execute_query(
        "MATCH (m:Movie {movieId: 1}) RETURN m.title AS t"
    ).records[0]["t"]
    assert title == "Toy Story"


def test_ask_end_to_end_on_the_fixture(api_client, monkeypatch):
    """Through HTTP: a scripted draft runs on the fixture and returns its rows."""
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "m")
    draft = CypherDraft(
        cypher="CALL db.index.fulltext.queryNodes('movie_title_fulltext', $text) "
               "YIELD node RETURN node.movieId AS movieId, node.title AS title",
        params={"text": "matrix"},
        explanation="Movies whose title matches 'matrix'.",
    )
    monkeypatch.setattr(llm, "complete_json", lambda *_: draft)
    response = api_client.post("/api/ask", json={"question": "Find The Matrix"})
    assert response.status_code == 200
    body = response.json()
    assert {"movieId": 2571, "title": "The Matrix"} in body["rows"]
    assert body["cypher"].endswith(f"LIMIT {guardrails.MAX_LIMIT}")
    assert all(check["passed"] for check in body["checks"])


def test_ask_resolves_named_movies_on_the_server(api_client, monkeypatch):
    """Through HTTP: the model names The Matrix, the server binds its movieId."""
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "m")
    draft = CypherDraft(
        cypher="MATCH (m:Movie {movieId: $m1})-[:IN_GENRE]->(g:Genre) RETURN g.name AS genre",
        movies=[MovieRef(param="m1", title="the matrix", year=1999)],
        explanation="Genres of The Matrix.",
    )
    monkeypatch.setattr(llm, "complete_json", lambda *_: draft)
    body = api_client.post("/api/ask", json={"question": "Genres of The Matrix?"}).json()
    assert body["params"] == {"m1": 2571}
    assert body["resolvedMovies"][0]["matchedTitle"] == "The Matrix"
    assert {row["genre"] for row in body["rows"]} == {"Action", "Sci-Fi", "Thriller"}


def test_a_repeated_unknown_property_is_reported_once(recommender_graph):
    """Neo4j warns per occurrence; the check states each problem once."""
    checks = {check.name: check for check in ask.explain(
        recommender_graph,
        "MATCH (m:Movie) WHERE m.avgRating > 4 RETURN m.avgRating ORDER BY m.avgRating", {},
    )}
    assert checks["database_warnings"].detail.count("avgRating` does not exist") == 1
