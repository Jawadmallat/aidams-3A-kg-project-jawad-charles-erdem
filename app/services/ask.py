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
from dataclasses import asdict

import neo4j
from neo4j import Driver
from neo4j.exceptions import ClientError

from app import llm
from app.models import CypherDraft
from app.services.ask_schema import describe_schema
from app.services.guardrails import MAX_LIMIT, Check, validate

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
  $name in the query (titles, tags, genres, ids, years, thresholds).
- Return named columns (RETURN m.title AS title), not whole nodes, and end
  with ORDER BY and LIMIT (at most {MAX_LIMIT}). Return as many rows as the
  question asks for: "which movie" is LIMIT 1, "which 5 movies" is LIMIT 5.
- If the graph cannot answer the question (actors, directors, box office,
  plot...), set "cypher" to "" and say why in "explanation".
- "explanation" is one or two sentences saying what the query computes.

Example: "Which movies are tagged 'atmospheric'?" ->
{{"cypher": "MATCH (:User)-[t:TAGGED]->(m:Movie) WHERE t.tag = $tag \
RETURN m.title AS title, count(*) AS taggings ORDER BY taggings DESC, title LIMIT 10",
 "params": {{"tag": "atmospheric"}},
 "explanation": "Movies users tagged 'atmospheric', most tagged first."}}
"""


class QueryRejected(Exception):
    """A draft failed a validation layer; it never ran (or was refused)."""

    def __init__(self, draft: CypherDraft, checks: list[Check]) -> None:
        super().__init__("generated query rejected")
        self.draft = draft
        self.checks = checks


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
            return {
                "question": question, "answered": False, "cypher": None, "params": {},
                "explanation": draft.explanation, "checks": [], "rows": [], "rowCount": 0,
                "truncated": False, "attempts": attempt, "model": settings.model,
            }
        validation = validate(draft.cypher, draft.params)
        if not validation.passed:
            raise QueryRejected(draft, validation.checks)
        dry_run = explain(driver, validation.cypher, validation.params)
        failures = [check for check in dry_run if not check.passed]
        if not failures:
            break
        unsafe = any(check.name == "read_only_plan" for check in failures)
        if unsafe or attempt > MAX_RETRIES:
            raise QueryRejected(draft, validation.checks + dry_run)
        messages += _feedback(draft, failures)

    checks = validation.checks + dry_run
    try:
        rows, truncated = execute(driver, validation.cypher, validation.params)
    except ClientError as error:
        if "TransactionTimedOut" in (error.code or ""):
            raise QueryTimeout(f"query exceeded {QUERY_TIMEOUT_S:g} s") from error
        failed = Check("read_transaction", False, f"{error.code}: {error.message}")
        raise QueryRejected(draft, checks + [failed]) from error
    checks.append(Check("read_transaction", True,
                        f"ran in a read transaction, timeout {QUERY_TIMEOUT_S:g} s"))
    return {
        "question": question, "answered": True, "cypher": validation.cypher,
        "params": validation.params, "explanation": draft.explanation,
        "checks": [asdict(check) for check in checks], "rows": rows,
        "rowCount": len(rows), "truncated": truncated, "attempts": attempt,
        "model": settings.model,
    }


def rejection_body(error: QueryRejected) -> dict:
    """The 400 body for a rejected draft: what was generated and what failed."""
    return {
        "detail": "the generated query was rejected by the guardrails",
        "cypher": error.draft.cypher,
        "params": json.loads(json.dumps(error.draft.params, default=str)),
        "explanation": error.draft.explanation,
        "checks": [asdict(check) for check in error.checks],
    }
