"""Unit tests of the static Cypher guardrails (no database, no LLM).

Two kinds of cases: queries an attacker or a confused model could produce,
which must be rejected by the named check, and legitimate queries that look
dangerous to a naive filter ('Set It Off', a `// delete` comment), which must
pass. Both matter: a guardrail that rejects good answers is useless too.
"""

import pytest

from app.services import guardrails
from app.services.guardrails import MAX_LIMIT, validate

GOOD_QUERY = """
MATCH (u:User)-[r:RATED]->(m:Movie)
WHERE r.rating >= 4.0
RETURN m.title AS title, count(*) AS likes
ORDER BY likes DESC
LIMIT 5
"""


def failed(cypher: str, params: dict | None = None) -> dict[str, str]:
    """The failed checks of a query, by name."""
    return {
        check.name: check.detail
        for check in validate(cypher, params).checks if not check.passed
    }


def test_a_plain_read_query_passes_every_check():
    """A typical answer to "most liked movies" passes unchanged."""
    validation = validate(GOOD_QUERY)
    assert validation.passed
    assert [check.name for check in validation.checks] == [
        "single_statement", "read_clause_first", "read_only_keywords",
        "allowed_namespaces", "allowed_procedures", "known_schema", "row_limit",
    ]
    assert validation.cypher == GOOD_QUERY.strip()


@pytest.mark.parametrize(
    ("cypher", "check", "reason"),
    [
        ("MATCH (n) DETACH DELETE n", "read_only_keywords", "DELETE, DETACH"),
        ("MATCH (m:Movie) SET m.title = 'x' RETURN m", "read_only_keywords", "SET"),
        ("CREATE (u:User {userId: 1}) RETURN u", "read_clause_first", "CREATE"),
        ("MERGE (g:Genre {name: 'x'}) RETURN g", "read_only_keywords", "MERGE"),
        ("MATCH (m:Movie) REMOVE m.year RETURN m", "read_only_keywords", "REMOVE"),
        ("DROP INDEX movie_year", "read_only_keywords", "DROP"),
        ("LOAD CSV FROM 'file:///etc/passwd' AS row RETURN row", "read_only_keywords", "LOAD"),
        ("MATCH (m:Movie) FOREACH (x IN [1] | SET m.y = x)", "read_only_keywords", "FOREACH"),
        ("MATCH (m:Movie) CALL (m) { SET m.x = 1 } IN TRANSACTIONS RETURN m",
         "read_only_keywords", "TRANSACTIONS"),
        ("SHOW USERS", "read_only_keywords", "SHOW"),
        ("USE system MATCH (n) RETURN n", "read_only_keywords", "USE"),
        ("MATCH (m:Movie) RETURN m; MATCH (n) DETACH DELETE n", "single_statement", "';'"),
        ("CALL apoc.cypher.runWrite('MATCH (n) DELETE n', {}) YIELD value RETURN value",
         "allowed_namespaces", "apoc"),
        ("RETURN apoc.cypher.runFirstColumnSingle('MATCH (n) RETURN n', {})",
         "allowed_namespaces", "apoc"),
        ("CALL gds.graph.drop('g') YIELD graphName RETURN graphName",
         "allowed_namespaces", "gds"),
        ("CALL dbms.components() YIELD name RETURN name", "allowed_namespaces", "dbms"),
        ("CALL db.labels() YIELD label RETURN label", "allowed_procedures", "db.labels"),
        ("CALL db.index.fulltext.queryNodes('secret_index', 'x') YIELD node RETURN node",
         "allowed_procedures", "secret_index"),
        ("MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a", "known_schema", "Actor, ACTED_IN"),
        ("MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre) RETURN g", "known_schema", "HAS_GENRE"),
        ("MATCH (m:movie) RETURN m", "known_schema", "movie"),
        ("MATCH (m:Movie:Film) RETURN m", "known_schema", "Film"),
        ("MATCH (u:User)-[:RATED|LIKED]->(m:Movie) RETURN m", "known_schema", "LIKED"),
        ("MATCH (m:`Secret Label`) RETURN m", "known_schema", "Secret Label"),
        ("MATCH (m:Movie) RETURN m LIMIT $n", "row_limit", "$n"),
    ],
)
def test_unsafe_or_invalid_queries_are_rejected(cypher, check, reason):
    """Each query fails the named check, and the detail names the culprit."""
    failures = failed(cypher)
    assert check in failures, failures
    assert reason in failures[check]
    assert not validate(cypher).passed


@pytest.mark.parametrize(
    "cypher",
    [
        # keywords inside string literals are data, not clauses
        "MATCH (m:Movie) WHERE m.title = 'Set It Off' RETURN m LIMIT 5",
        'MATCH (u:User)-[t:TAGGED]->(m:Movie) WHERE t.tag = "delete scene" RETURN m LIMIT 5',
        # ... and inside comments
        "MATCH (m:Movie) // never DELETE anything\nRETURN m LIMIT 5",
        "/* CREATE nothing */ MATCH (m:Movie) RETURN m LIMIT 5",
        # properties and parameters named like keywords
        "MATCH (m:Movie) WHERE m.year = $set RETURN m.title LIMIT 5",
        # read subqueries, list comprehensions, label predicates, inline WHERE
        "MATCH (m:Movie) CALL (m) { MATCH (m)-[:IN_GENRE]->(g:Genre) RETURN count(g) AS n } "
        "RETURN m.title, n LIMIT 5",
        "MATCH (m:Movie) RETURN [(m)-[:IN_GENRE]->(g:Genre) | g.name] AS genres LIMIT 5",
        "MATCH (m:Movie WHERE m.year > 2000) RETURN m.title LIMIT 5",
        "MATCH (m) WHERE m:Movie RETURN m.title LIMIT 5",
        "MATCH (m:Movie)-[s:SIMILAR*1..2]->(o:Movie) RETURN o.title LIMIT 5",
        "MATCH (u:User)-[:RATED|TAGGED]->(m:Movie) RETURN m.title LIMIT 5",
        "MATCH (m:Movie {movieId: 1}) RETURN {title: m.title, year: m.year} AS movie",
        # the one allowed procedure, on a declared index
        "CALL db.index.fulltext.queryNodes('movie_title_fulltext', 'matrix') "
        "YIELD node, score RETURN node.title, score LIMIT 5",
        "WITH 1 AS one MATCH (m:Movie) RETURN m.title, one LIMIT 5",
        "UNWIND [1, 2] AS id MATCH (m:Movie {movieId: id}) RETURN m.title",
        "MATCH (m:Movie) RETURN m.title LIMIT 5;",
    ],
)
def test_legitimate_queries_pass(cypher):
    """Read queries that only look suspicious to a naive filter are accepted."""
    assert failed(cypher) == {}


def test_missing_limit_is_appended():
    """No LIMIT: the query runs with LIMIT MAX_LIMIT, and the check says so."""
    validation = validate("MATCH (m:Movie) RETURN m.title")
    assert validation.passed
    assert validation.cypher.endswith(f"LIMIT {MAX_LIMIT}")
    assert "appended" in validation.checks[-1].detail


def test_a_limit_inside_a_subquery_is_not_the_final_limit():
    """Only the LIMIT closing the query bounds the rows returned."""
    validation = validate(
        "MATCH (m:Movie) CALL (m) { MATCH (m)<-[:RATED]-(u:User) RETURN u LIMIT 3 } "
        "RETURN m.title, u.userId"
    )
    assert validation.cypher.endswith(f"LIMIT {MAX_LIMIT}")


def test_large_literal_limit_is_lowered():
    """LIMIT 100000 becomes LIMIT MAX_LIMIT instead of a rejection."""
    validation = validate("MATCH (m:Movie) RETURN m.title LIMIT 100000")
    assert validation.passed
    assert validation.cypher == f"MATCH (m:Movie) RETURN m.title LIMIT {MAX_LIMIT}"
    assert "lowered" in validation.checks[-1].detail


def test_large_parameter_limit_is_lowered_in_the_params():
    """LIMIT $n keeps the text and caps the parameter value."""
    validation = validate("MATCH (m:Movie) RETURN m LIMIT $n", {"n": 5000, "x": 1})
    assert validation.passed
    assert validation.params == {"n": MAX_LIMIT, "x": 1}


def test_small_parameter_limit_is_kept():
    """A LIMIT parameter within bounds is left alone."""
    validation = validate("MATCH (m:Movie) RETURN m LIMIT $n", {"n": 3})
    assert validation.passed
    assert validation.params == {"n": 3}


@pytest.mark.parametrize("value", ["10", -1, True, None, 2.5])
def test_non_integer_parameter_limit_is_rejected(value):
    """LIMIT $n with a string, negative, boolean, null or float value fails."""
    assert "row_limit" in failed("MATCH (m:Movie) RETURN m LIMIT $n", {"n": value})


def test_trailing_comment_cannot_swallow_the_appended_limit():
    """Comments are removed from the query that runs."""
    validation = validate("MATCH (m:Movie) RETURN m.title // top movies")
    assert validation.cypher == f"MATCH (m:Movie) RETURN m.title\nLIMIT {MAX_LIMIT}"


def test_every_failure_is_reported_at_once():
    """All checks run, so one rejection lists every broken rule."""
    failures = failed("CREATE (a:Actor) ; MATCH (n) DETACH DELETE n")
    assert {"single_statement", "read_clause_first", "read_only_keywords",
            "known_schema"} <= set(failures)


def test_empty_query_is_rejected():
    """An empty answer from the model is not a query."""
    assert "read_clause_first" in failed("   ")


def test_strip_literals_keeps_quotes_and_drops_contents():
    """String contents and comments vanish; the structure stays."""
    assert guardrails.strip_literals("RETURN 'a;b' // c\n, \"d\"") == "RETURN ''  \n, \"\""
