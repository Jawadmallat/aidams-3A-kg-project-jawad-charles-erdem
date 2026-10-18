"""Control flow of the NL-to-Cypher pipeline, with the LLM and Neo4j stubbed.

`complete_json` is replaced by a script of drafts, and the EXPLAIN and
execution layers by stubs, so these tests pin down the decisions (reject,
retry once, never retry an unsafe plan, decline) without a key or a database.
The layers themselves run against a real Neo4j in test_ask_integration.py.
"""

import pytest
from neo4j.exceptions import ClientError

from app import llm
from app.models import CypherDraft, MovieRef
from app.services import ask
from app.services.guardrails import Check

GOOD = CypherDraft(
    cypher="MATCH (m:Movie) RETURN m.title AS title LIMIT 3", params={},
    explanation="Three movie titles.",
)
COMPILES = [
    Check("explain_dry_run", True, "the query compiles"),
    Check("read_only_plan", True, "read-only"),
    Check("database_warnings", True, "no warning"),
]


class CodedClientError(ClientError):
    """A ClientError with the code and message Neo4j would send."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self._code, self._message = code, message

    @property
    def code(self) -> str:
        """The Neo4j status code."""
        return self._code

    @property
    def message(self) -> str:
        """The server's message."""
        return self._message


class ScriptedLLM:
    """Returns the scripted drafts in order and records every prompt."""

    def __init__(self, *drafts: CypherDraft) -> None:
        self.drafts = list(drafts)
        self.prompts: list[list[dict[str, str]]] = []

    def __call__(self, messages, schema):
        assert schema is CypherDraft
        self.prompts.append(list(messages))
        return self.drafts.pop(0)


@pytest.fixture(name="stub")
def fixture_stub(monkeypatch):
    """Configure a fake model; returns a helper to script the layers."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    def script(*drafts, explains=None, rows=None, movie_ids=None):
        model = ScriptedLLM(*drafts)
        explain_results = list(explains or [COMPILES])
        monkeypatch.setattr(llm, "complete_json", model)
        monkeypatch.setattr(ask, "explain", lambda *_: explain_results.pop(0))
        ids = dict(movie_ids or {})
        monkeypatch.setattr(ask, "resolve_movies", lambda _driver, refs: [
            {"param": ref.param, "title": ref.title, "year": ref.year,
             "movieId": ids.get(ref.title), "matchedTitle": ref.title if ref.title in ids
             else None, "matchedYear": None, "score": None, "ambiguous": False}
            for ref in refs
        ])
        if isinstance(rows, Exception):
            def failing_execute(*_):
                raise rows
            monkeypatch.setattr(ask, "execute", failing_execute)
        else:
            monkeypatch.setattr(ask, "execute", lambda *_: (rows or [], False))
        return model

    return script


def test_a_valid_draft_runs_and_reports_every_layer(stub):
    """Rows come back with the executed query and every check passed."""
    stub(GOOD, rows=[{"title": "Toy Story"}])
    result = ask.answer(driver=None, question="three movies")
    assert result["answered"] is True
    assert result["rows"] == [{"title": "Toy Story"}]
    assert result["rowCount"] == 1
    assert result["attempts"] == 1
    assert result["model"] == "test-model"
    assert result["cypher"] == GOOD.cypher
    names = [check["name"] for check in result["checks"]]
    assert names[-4:] == ["explain_dry_run", "read_only_plan", "database_warnings",
                          "read_transaction"]
    assert all(check["passed"] for check in result["checks"])


def test_the_prompt_carries_the_schema_and_the_question(stub):
    """The model sees the real schema and conventions, then the question."""
    model = stub(GOOD)
    ask.answer(driver=None, question="three movies")
    system, user = model.prompts[0]
    assert "(:User)-[:RATED]->(:Movie)" in system["content"]
    assert "rating >= 4.0" in system["content"]
    assert user == {"role": "user", "content": "three movies"}


def test_a_declined_question_runs_nothing(stub):
    """An empty cypher is the model saying the graph cannot answer."""
    stub(CypherDraft(cypher="", explanation="No director data in the graph."))
    result = ask.answer(driver=None, question="who directed The Matrix?")
    assert result["answered"] is False
    assert result["cypher"] is None
    assert result["rows"] == []
    assert result["explanation"] == "No director data in the graph."


def test_a_static_guardrail_failure_is_rejected_without_retry(stub):
    """A write draft is rejected at layer 1; the model is not asked again."""
    model = stub(CypherDraft(cypher="MATCH (n) DETACH DELETE n", explanation="x"))
    with pytest.raises(ask.QueryRejected) as caught:
        ask.answer(driver=None, question="delete everything")
    failed = {check.name for check in caught.value.checks if not check.passed}
    assert "read_only_keywords" in failed
    assert len(model.prompts) == 1


def test_a_fixable_explain_failure_is_retried_once_with_feedback(stub):
    """A syntax error goes back to the model with Neo4j's message."""
    broken = CypherDraft(cypher="MATCH (m:Movie) RETURN m.title LIMT 3", explanation="x")
    syntax_error = [Check("explain_dry_run", False, "SyntaxError: Invalid input 'LIMT'")]
    model = stub(broken, GOOD, explains=[syntax_error, COMPILES], rows=[{"title": "Heat"}])
    result = ask.answer(driver=None, question="three movies")
    assert result["attempts"] == 2
    assert result["rows"] == [{"title": "Heat"}]
    feedback = model.prompts[1][-1]["content"]
    assert "Invalid input 'LIMT'" in feedback
    assert model.prompts[1][-2]["role"] == "assistant"


def test_every_rejected_draft_is_logged_even_when_a_retry_fixes_it(stub, caplog):
    """The audit trail keeps the rejected first draft the response does not show."""
    broken = CypherDraft(cypher="MATCH (m:Movie) RETURN m.avgRating AS r", explanation="x")
    unknown = [Check("database_warnings", False, "property `avgRating` does not exist")]
    stub(broken, GOOD, explains=[COMPILES[:2] + unknown, COMPILES])
    with caplog.at_level("WARNING", logger="app.services.ask"):
        ask.answer(driver=None, question="movies by avgRating")
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "sent back to the model" in message
    assert "m.avgRating" in message
    assert "avgRating` does not exist" in message


def test_a_static_rejection_is_logged_as_final(stub, caplog):
    """A draft stopped by layer 1 is logged with the failed rule."""
    stub(CypherDraft(cypher="MATCH (n) DETACH DELETE n", explanation="x"))
    with caplog.at_level("WARNING", logger="app.services.ask"), \
            pytest.raises(ask.QueryRejected):
        ask.answer(driver=None, question="delete everything")
    assert "(final)" in caplog.records[0].getMessage()
    assert "read_only_keywords" in caplog.records[0].getMessage()


def test_a_second_explain_failure_is_rejected(stub):
    """After the one retry, a still-broken query is rejected."""
    missing = [Check("database_warnings", False, "unknown property key")]
    model = stub(GOOD, GOOD, explains=[COMPILES[:2] + missing, COMPILES[:2] + missing])
    with pytest.raises(ask.QueryRejected) as caught:
        ask.answer(driver=None, question="q")
    assert len(model.prompts) == 2
    assert any(check.name == "database_warnings" and not check.passed
               for check in caught.value.checks)


def test_a_write_plan_is_never_retried(stub):
    """A plan Neo4j classifies as writing is a safety failure: no second chance."""
    writes = [Check("explain_dry_run", True, "ok"), Check("read_only_plan", False, "'rw'")]
    model = stub(GOOD, GOOD, explains=[writes, COMPILES])
    with pytest.raises(ask.QueryRejected):
        ask.answer(driver=None, question="q")
    assert len(model.prompts) == 1


def test_a_refusal_by_the_read_transaction_is_a_rejection(stub):
    """Layer 3: Neo4j's AccessMode error becomes a failed read_transaction check."""
    refused = CodedClientError(
        "Neo.ClientError.Statement.AccessMode", "Writing in read access mode not allowed"
    )
    stub(GOOD, rows=refused)
    with pytest.raises(ask.QueryRejected) as caught:
        ask.answer(driver=None, question="q")
    last = caught.value.checks[-1]
    assert (last.name, last.passed) == ("read_transaction", False)
    assert "AccessMode" in last.detail


def test_a_timeout_is_reported_as_such(stub):
    """Neo4j's transaction timeout becomes QueryTimeout, not a rejection."""
    timed_out = CodedClientError(
        "Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration", "timed out"
    )
    stub(GOOD, rows=timed_out)
    with pytest.raises(ask.QueryTimeout):
        ask.answer(driver=None, question="q")


def test_rejection_body_shows_the_draft_and_the_checks():
    """The 400 body carries what the model wrote and every verdict."""
    error = ask.QueryRejected(
        CypherDraft(cypher="MATCH (n) DELETE n", params={"x": 1}, explanation="e"),
        [Check("read_only_keywords", False, "forbidden: DELETE")],
    )
    body = ask.rejection_body(error)
    assert body["cypher"] == "MATCH (n) DELETE n"
    assert body["params"] == {"x": 1}
    assert body["checks"] == [
        {"name": "read_only_keywords", "passed": False, "detail": "forbidden: DELETE"}
    ]


def test_jsonable_turns_driver_values_into_json():
    """Nested containers are kept; unknown driver types become strings."""
    class Temporal:
        """Stands in for a neo4j.time value."""

        def __str__(self) -> str:
            return "2018-09-24T00:00:00Z"

    value = {"a": (1, 2.5, None), "b": [True, {"c": Temporal()}]}
    assert ask._jsonable(value) == {  # pylint: disable=protected-access
        "a": [1, 2.5, None], "b": [True, {"c": "2018-09-24T00:00:00Z"}]
    }


NAMED = CypherDraft(
    cypher="MATCH (m:Movie {movieId: $m1})<-[r:RATED]-() RETURN count(r) AS n",
    params={"m1": "the model's guess"},
    movies=[MovieRef(param="m1", title="Heat")],
    explanation="Ratings of Heat.",
)


def test_named_movies_are_bound_by_the_server(stub):
    """The resolved movieId replaces whatever the model put in that parameter."""
    stub(NAMED, movie_ids={"Heat": 6}, rows=[{"n": 110}])
    result = ask.answer(driver=None, question="How many ratings does Heat have?")
    assert result["params"] == {"m1": 6}
    assert result["resolvedMovies"][0]["movieId"] == 6
    assert result["rows"] == [{"n": 110}]


def test_an_unknown_movie_is_not_answered_and_nothing_runs(stub):
    """A named movie absent from the graph: answered false, with the reason."""
    stub(NAMED, movie_ids={}, rows=ValueError("must not run"))
    result = ask.answer(driver=None, question="How many ratings does Heat have?")
    assert result["answered"] is False
    assert result["cypher"] is None
    assert result["explanation"] == "No movie in the graph matches 'Heat'."
    assert result["resolvedMovies"][0]["movieId"] is None


def test_the_missing_movie_explanation_names_the_year():
    """Unresolved movies are listed with their year when given."""
    resolved = [{"title": "Heat", "year": 1995, "movieId": None, "ambiguous": False},
                {"title": "Alien", "year": None, "movieId": 1214, "ambiguous": False}]
    assert ask._unresolved_explanation(resolved) == (  # pylint: disable=protected-access
        "No movie in the graph matches 'Heat' (1995)."
    )


def test_an_ambiguous_title_asks_for_the_year_and_runs_nothing(stub, monkeypatch):
    """A title shared by several films: answered false, the candidates listed."""
    stub(NAMED, rows=ValueError("must not run"))
    monkeypatch.setattr(ask, "resolve_movies", lambda _driver, refs: [{
        "param": "m1", "title": "Titanic", "year": None, "movieId": None,
        "matchedTitle": None, "matchedYear": None, "score": None, "ambiguous": True,
        "candidates": [{"movieId": 1721, "title": "Titanic", "year": 1997},
                       {"movieId": 2157, "title": "Titanic", "year": 1953}],
    }])
    result = ask.answer(driver=None, question="How many ratings does Titanic have?")
    assert result["answered"] is False
    assert result["cypher"] is None
    assert result["explanation"] == (
        "'Titanic' matches several movies: Titanic (1997, movie id 1721); "
        "Titanic (1953, movie id 2157). Ask again with the year or the movie id."
    )
    assert len(result["resolvedMovies"][0]["candidates"]) == 2


def test_a_rejection_reports_the_resolved_movies(stub):
    """The 400 body shows which movies the server matched."""
    bad = NAMED.model_copy(update={"cypher": "MATCH (m:Movie {movieId: $m1}) DETACH DELETE m"})
    stub(bad, movie_ids={"Heat": 6})
    with pytest.raises(ask.QueryRejected) as caught:
        ask.answer(driver=None, question="delete Heat")
    assert ask.rejection_body(caught.value)["resolvedMovies"][0]["movieId"] == 6
