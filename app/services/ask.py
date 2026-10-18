"""NL-to-Cypher: question -> LLM draft -> three validation layers -> rows.

The LLM only writes a query; Neo4j produces the answer. A draft runs only
after it passes, in order:

1. the static guardrails (app/services/guardrails.py): read-only keywords,
   allowed procedures, labels and types from the schema, a bounded LIMIT;
2. an EXPLAIN dry run: the plan must compile, be read-only (query type "r")
   and raise no warning about unknown labels, types, properties or missing
   parameters;
3. execution in a read transaction with a timeout, so even a write that got
   past both layers is refused by the server.

A draft that fails layer 2 for a fixable reason (syntax, unknown name) is
sent back to the model once with the database's error; a safety failure is
never retried. Every step is reported in `checks`.
"""

import json
import logging
from dataclasses import asdict

import neo4j
from neo4j import Driver
from neo4j.exceptions import ClientError

from app import llm
from app.models import CypherDraft
from app.services.ask_schema import describe_schema
from app.services.guardrails import MAX_LIMIT, Check, Validation, validate
from app.services.titles import resolve_movies

# Every rejected draft is logged (WARNING) with the checks it failed, including
# drafts that a retry later fixed and that the response therefore does not
# show: `docker compose logs api` is the audit trail of what the LLM wrote.
logger = logging.getLogger(__name__)

# Seconds a generated query may run before Neo4j aborts it.
QUERY_TIMEOUT_S = 10.0

# Extra attempts after a draft fails the EXPLAIN dry run for a fixable reason.
MAX_RETRIES = 1

# GQL status codes of EXPLAIN warnings that mean the query cannot answer the
# question as written: unknown label, relationship type, property key, or a
# parameter the draft uses but does not provide.
BLOCKING_WARNINGS = {
    "01N50": "unknown label",
    "01N51": "unknown relationship type",
    "01N52": "unknown property key",
    "01N60": "missing parameter",
}

SYSTEM_PROMPT = f"""You translate questions about a MovieLens movie graph into one
read-only Cypher query for Neo4j 5.

{describe_schema()}

Rules:
- Only read: MATCH, OPTIONAL MATCH, WITH, UNWIND, WHERE, RETURN, ORDER BY,
  SKIP, LIMIT, read-only CALL {{}} subqueries, and the procedure
  db.index.fulltext.queryNodes. Never write, never use apoc, gds or dbms.
- Put every value taken from the question in `params` and reference it as
  $name in the query (tags, genres, ids, years, thresholds, search text).
- Never look up a movie the question names by its title in the query. List
  it in "movies" as {{"param": "m1", "title": "Toy Story", "year": 1995}}
  (year null when not given) and match it as (m:Movie {{movieId: $m1}}): the
  server resolves the title and binds its movieId to $m1.
- Return named columns (RETURN m.title AS title), not whole nodes, and end
  with ORDER BY and LIMIT (at most {MAX_LIMIT}). Return as many rows as the
  question asks for: "which movie" is LIMIT 1, "which 5 movies" is LIMIT 5.
- If the graph cannot answer the question (actors, directors, box office,
  plot...), set "cypher" to "" and say why in "explanation".
- "explanation" is one or two sentences saying what the query computes.

Example: "Which movies are tagged 'atmospheric'?" ->
{{"cypher": "MATCH (:User)-[t:TAGGED]->(m:Movie) WHERE t.tag = $tag \
RETURN m.title AS title, count(*) AS taggings ORDER BY taggings DESC, title LIMIT 10",
 "params": {{"tag": "atmospheric"}}, "movies": [],
 "explanation": "Movies users tagged 'atmospheric', most tagged first."}}

Example: "How many users rated Heat?" ->
{{"cypher": "MATCH (m:Movie {{movieId: $m1}})<-[r:RATED]-(:User) RETURN count(r) AS ratings",
 "params": {{}}, "movies": [{{"param": "m1", "title": "Heat", "year": null}}],
 "explanation": "Counts the ratings of the movie Heat."}}
"""


class QueryRejected(Exception):
    """A draft failed a validation layer; it never ran (or was refused)."""

    def __init__(self, draft: CypherDraft, checks: list[Check],
                 resolved: list[dict] | None = None) -> None:
        super().__init__("generated query rejected")
        self.draft = draft
        self.checks = checks
        self.resolved = resolved or []


class QueryTimeout(Exception):
    """The generated query ran longer than QUERY_TIMEOUT_S."""


def build_messages(question: str) -> list[dict[str, str]]:
    """The chat messages that ask the model for a CypherDraft."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def explain(driver: Driver, cypher: str, params: dict) -> list[Check]:
    """Layer 2: compile the query with EXPLAIN, without running it.

    Returns the "explain_dry_run", "read_only_plan" and "database_warnings"
    checks; only the first one when the query does not compile.
    """
    with driver.session(
        default_access_mode=neo4j.READ_ACCESS, notifications_min_severity="WARNING"
    ) as session:
        try:
            summary = session.run("EXPLAIN " + cypher, params).consume()
        except ClientError as error:
            return [Check("explain_dry_run", False, f"{error.code}: {error.message}")]
    checks = [Check("explain_dry_run", True, "the query compiles")]
    if summary.query_type == "r":
        checks.append(Check("read_only_plan", True, "Neo4j classifies the plan as read-only"))
    else:
        checks.append(Check("read_only_plan", False,
                            f"Neo4j classifies the plan as '{summary.query_type}', not read-only"))
    blocking, other = [], []
    for status in summary.gql_status_objects:
        if status.raw_severity != "WARNING":
            continue
        if status.gql_status in BLOCKING_WARNINGS:
            blocking.append(status.status_description)
        else:
            other.append(status.status_description)
    # One warning per occurrence: `m.x` used twice is reported twice.
    blocking, other = list(dict.fromkeys(blocking)), list(dict.fromkeys(other))
    if blocking:
        checks.append(Check("database_warnings", False, " | ".join(blocking)))
    else:
        checks.append(Check("database_warnings", True,
                            " | ".join(other) if other else "no warning"))
    return checks


def _jsonable(value):
    """Neo4j values as JSON: nodes become property maps, temporals strings."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def execute(driver: Driver, cypher: str, params: dict) -> tuple[list[dict], bool]:
    """Layer 3: run the query in a read transaction with a timeout.

    Returns at most MAX_LIMIT rows and whether more were available (a UNION
    can return more rows than its final LIMIT).
    """
    @neo4j.unit_of_work(timeout=QUERY_TIMEOUT_S)
    def read(tx):
        return [record.data() for record in tx.run(cypher, params).fetch(MAX_LIMIT + 1)]

    with driver.session(default_access_mode=neo4j.READ_ACCESS) as session:
        rows = session.execute_read(read)
    return [_jsonable(row) for row in rows[:MAX_LIMIT]], len(rows) > MAX_LIMIT


def _feedback(draft: CypherDraft, failures: list[Check]) -> list[dict[str, str]]:
    """Messages that show the model its draft and why Neo4j refused it."""
    return [
        {"role": "assistant", "content": draft.model_dump_json()},
        {"role": "user", "content": "Neo4j refused this query:\n"
         + "\n".join(f"- {check.name}: {check.detail}" for check in failures)
         + "\nFix the query, following the same rules."},
    ]


def _log_rejection(question: str, draft: CypherDraft, checks: list[Check], outcome: str) -> None:
    """Record a rejected draft and why, whatever happens next."""
    failed = "; ".join(f"{c.name}: {c.detail}" for c in checks if not c.passed)
    logger.warning("ask: draft rejected (%s) | question=%r | cypher=%r | failed=%s",
                   outcome, question, draft.cypher, failed)


def _execute_checked(driver: Driver, validation: Validation, rejection: QueryRejected):
    """Layer 3 with its errors mapped: timeout, or a refusal added to `rejection`."""
    try:
        return execute(driver, validation.cypher, validation.params)
    except ClientError as error:
        if "TransactionTimedOut" in (error.code or ""):
            raise QueryTimeout(f"query exceeded {QUERY_TIMEOUT_S:g} s") from error
        rejection.checks = rejection.checks + [
            Check("read_transaction", False, f"{error.code}: {error.message}")
        ]
        raise rejection from error


def _not_answered(question: str, explanation: str, attempt: int, model: str,
                  resolved: list[dict]) -> dict:
    """The response when no query runs (declined, or a named movie is missing)."""
    return {
        "question": question, "answered": False, "cypher": None, "params": {},
        "explanation": explanation, "checks": [], "rows": [], "rowCount": 0,
        "truncated": False, "attempts": attempt, "model": model, "resolvedMovies": resolved,
    }


def _named(movie: dict) -> str:
    """'Title' or 'Title' (year), as the question named the movie."""
    return f"'{movie['title']}'" + (f" ({movie['year']})" if movie["year"] else "")


def _unresolved_explanation(resolved: list[dict]) -> str | None:
    """Why no query can run: a named movie is missing, or its title is ambiguous.

    An ambiguous title lists its candidates so the user can ask again with a
    year or a movie id, instead of getting an answer about an arbitrary film.
    """
    reasons = []
    for movie in resolved:
        if movie["ambiguous"]:
            options = "; ".join(
                f"{c['title']} ({c['year'] if c['year'] else 'year unknown'}, "
                f"movie id {c['movieId']})" for c in movie["candidates"]
            )
            reasons.append(f"{_named(movie)} matches several movies: {options}. "
                           "Ask again with the year or the movie id.")
        elif movie["movieId"] is None:
            reasons.append(f"No movie in the graph matches {_named(movie)}.")
    return " ".join(reasons) or None


def answer(driver: Driver, question: str) -> dict:
    """Answer a question with LLM-written, validated, read-only Cypher.

    Raises llm.LLMNotConfigured / llm.LLMError (no or failing model),
    QueryRejected (a validation layer failed) and QueryTimeout.
    """
    settings = llm.get_settings()
    messages = build_messages(question)
    for attempt in range(1, MAX_RETRIES + 2):
        draft = llm.complete_json(messages, CypherDraft)
        if not draft.cypher.strip():
            return _not_answered(question, draft.explanation, attempt, settings.model, [])
        resolved = resolve_movies(driver, draft.movies)
        if missing := _unresolved_explanation(resolved):
            return _not_answered(question, missing, attempt, settings.model, resolved)
        validation = validate(draft.cypher, {
            **draft.params, **{movie["param"]: movie["movieId"] for movie in resolved}
        })
        if not validation.passed:
            _log_rejection(question, draft, validation.checks, "final")
            raise QueryRejected(draft, validation.checks, resolved)
        dry_run = explain(driver, validation.cypher, validation.params)
        failures = [check for check in dry_run if not check.passed]
        if not failures:
            break
        if attempt > MAX_RETRIES or any(c.name == "read_only_plan" for c in failures):
            _log_rejection(question, draft, dry_run, "final")
            raise QueryRejected(draft, validation.checks + dry_run, resolved)
        _log_rejection(question, draft, dry_run, "sent back to the model")
        messages += _feedback(draft, failures)

    checks = validation.checks + dry_run
    rows, truncated = _execute_checked(driver, validation, QueryRejected(draft, checks, resolved))
    checks.append(Check("read_transaction", True,
                        f"ran in a read transaction, timeout {QUERY_TIMEOUT_S:g} s"))
    return {
        "question": question, "answered": True, "cypher": validation.cypher,
        "params": validation.params, "explanation": draft.explanation,
        "checks": [asdict(check) for check in checks], "rows": rows,
        "rowCount": len(rows), "truncated": truncated, "attempts": attempt,
        "model": settings.model, "resolvedMovies": resolved,
    }


def rejection_body(error: QueryRejected) -> dict:
    """The 400 body for a rejected draft: what was generated and what failed."""
    return {
        "detail": "the generated query was rejected by the guardrails",
        "cypher": error.draft.cypher,
        "params": json.loads(json.dumps(error.draft.params, default=str)),
        "explanation": error.draft.explanation,
        "checks": [asdict(check) for check in error.checks],
        "resolvedMovies": error.resolved,
    }
