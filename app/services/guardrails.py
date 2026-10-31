"""Static checks on LLM-generated Cypher, before it ever reaches Neo4j.

The model is untrusted: whatever it writes goes through these checks, and a
query runs only if all of them pass. They are the first of three layers in
app/services/ask.py; the other two are server-side (an EXPLAIN dry run that
must report a read-only plan, and execution in a read transaction), so a
keyword that slips past this text-level filter still cannot write.

Every check reports a name, a verdict and a reason, so a rejection can show
exactly which rule fired (the live demo of a guardrail rejection).
"""

import re
from dataclasses import dataclass

from app.services.ask_schema import (
    ALLOWED_PROCEDURES,
    FULLTEXT_INDEXES,
    NODE_PROPERTIES,
    RELATIONSHIPS,
)

# Upper bound on the rows any generated query may return. Deliberately larger
# than the REST page size (MAX_LIMIT = 50 in app/routers): an LLM answer is
# one response with no next page, so it gets more room.
MAX_LIMIT = 100

# Clauses and keywords that write, change the schema or administer the
# server. Matched as whole words on the query with string literals and
# comments removed, so a title like 'Set It Off' does not trigger them.
FORBIDDEN_KEYWORDS = (
    "CREATE", "MERGE", "DELETE", "DETACH", "SET", "REMOVE", "DROP", "FOREACH",
    "LOAD", "TRANSACTIONS", "GRANT", "REVOKE", "DENY", "ALTER", "RENAME",
    "START", "STOP", "TERMINATE", "ENABLE", "SHOW", "USE",
)

# Namespaces whose functions can run arbitrary Cypher or reach outside the
# graph (apoc.cypher.*, apoc.load.*, gds.*, dbms.*); none are needed to answer
# questions about this schema.
FORBIDDEN_NAMESPACES = ("apoc", "gds", "dbms")

# A query must open with a clause that reads.
READ_OPENERS = ("MATCH", "OPTIONAL", "WITH", "UNWIND", "CALL", "RETURN")

_LITERAL_OR_COMMENT = re.compile(
    r"'(?:\\.|[^'\\])*'"      # single-quoted string
    r'|"(?:\\.|[^"\\])*"'     # double-quoted string
    r"|//[^\n]*"              # line comment
    r"|/\*.*?\*/",            # block comment
    re.DOTALL,
)
_NAME = r"`[^`]+`|\w+"
# A label or type expression: names joined by `:`, `|` or `&`, each maybe `!`.
_EXPRESSION = rf"!?(?:{_NAME})(?:\s*[:|&]\s*!?(?:{_NAME}))*"
_NODE_LABELS = re.compile(rf"\(\s*(?:{_NAME})?\s*:\s*({_EXPRESSION})")
_REL_TYPES = re.compile(rf"\[\s*(?:{_NAME})?\s*:\s*({_EXPRESSION})")
_PROCEDURE = re.compile(r"\bCALL\s+([A-Za-z_][\w.]*)\s*\(", re.IGNORECASE)
_FULLTEXT_INDEX = re.compile(
    r"db\.index\.fulltext\.query(?:Nodes|Relationships)\s*\(\s*'([^']*)'", re.IGNORECASE
)
_FINAL_LIMIT = re.compile(r"\bLIMIT\s+(\d+|\$\w+)\s*;?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class Check:
    """The verdict of one guardrail on one query."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Validation:
    """All verdicts, plus the query and parameters to run if they all pass.

    `cypher` and `params` may differ from the model's: the LIMIT guardrail
    appends or lowers the final LIMIT instead of rejecting the query.
    """

    cypher: str
    params: dict
    checks: list[Check]

    @property
    def passed(self) -> bool:
        """True when every check passed: only then may the query run."""
        return all(check.passed for check in self.checks)


def strip_literals(cypher: str) -> str:
    """The query with string contents and comments removed (code only).

    Strings keep their quotes but lose their contents, so keyword and label
    checks see the code the database will execute and not the data in it.
    """
    def blank(match: re.Match) -> str:
        text = match.group(0)
        return text[0] * 2 if text[0] in "'\"" else " "

    return _LITERAL_OR_COMMENT.sub(blank, cypher)


def remove_comments(cypher: str) -> str:
    """The query without comments, string literals untouched.

    This is the text that runs: a trailing `// comment` would otherwise
    swallow the LIMIT the guardrail appends.
    """
    def blank(match: re.Match) -> str:
        text = match.group(0)
        return text if text[0] in "'\"" else " "

    return _LITERAL_OR_COMMENT.sub(blank, cypher)


def _names(group: str) -> set[str]:
    """Split a label or type expression (`A|B`, `A:B`, `!A`) into bare names."""
    return {name.strip("`") for name in re.findall(_NAME, group)}


def check_single_statement(code: str) -> Check:
    """One statement only: a `;` may only end the query."""
    body = code.strip().rstrip(";")
    if ";" in body:
        return Check("single_statement", False, "several statements separated by ';'")
    return Check("single_statement", True, "one statement")


def check_read_opener(code: str) -> Check:
    """The query starts with a reading clause (MATCH, WITH, RETURN, ...)."""
    first = code.strip().split(None, 1)[0].upper() if code.strip() else ""
    if first in READ_OPENERS:
        return Check("read_clause_first", True, f"starts with {first}")
    return Check("read_clause_first", False, f"starts with {first or 'nothing'}, not a read clause")


def check_forbidden_keywords(code: str) -> Check:
    """No clause that writes, changes the schema or administers the server."""
    found = sorted({
        keyword for keyword in FORBIDDEN_KEYWORDS
        if re.search(rf"(?<![\w.$]){keyword}\b", code, re.IGNORECASE)
    })
    if found:
        return Check("read_only_keywords", False, "forbidden: " + ", ".join(found))
    return Check("read_only_keywords", True, "no write or admin keyword")


def check_namespaces(code: str) -> Check:
    """No apoc.*, gds.* or dbms.* function or procedure."""
    found = sorted({
        namespace for namespace in FORBIDDEN_NAMESPACES
        if re.search(rf"\b{namespace}\s*\.", code, re.IGNORECASE)
    })
    if found:
        return Check("allowed_namespaces", False, "forbidden namespace: " + ", ".join(found))
    return Check("allowed_namespaces", True, "no apoc, gds or dbms call")


def check_procedures(code: str, cypher: str) -> Check:
    """CALL only the full-text procedures, on the indexes the schema declares.

    `cypher` (with literals) is needed to read the index name argument.
    """
    called = set(_PROCEDURE.findall(code))
    unknown = sorted(name for name in called if name not in ALLOWED_PROCEDURES)
    if unknown:
        return Check("allowed_procedures", False, "procedure not allowed: " + ", ".join(unknown))
    bad_indexes = sorted(
        name for name in _FULLTEXT_INDEX.findall(cypher) if name not in FULLTEXT_INDEXES
    )
    if bad_indexes:
        return Check("allowed_procedures", False, "unknown index: " + ", ".join(bad_indexes))
    detail = ", ".join(sorted(called)) if called else "no procedure call"
    return Check("allowed_procedures", True, detail)


def check_schema_names(code: str) -> Check:
    """Every label in a node pattern and every relationship type is in the schema."""
    labels = set().union(*(_names(group) for group in _NODE_LABELS.findall(code)))
    rel_types = set().union(*(_names(group) for group in _REL_TYPES.findall(code)))
    unknown = sorted(labels - set(NODE_PROPERTIES)) + sorted(rel_types - set(RELATIONSHIPS))
    if unknown:
        return Check("known_schema", False, "not in the schema: " + ", ".join(unknown))
    used = sorted(labels) + sorted(rel_types)
    return Check("known_schema", True, "uses " + (", ".join(used) if used else "no label"))


def enforce_limit(cypher: str, code: str, params: dict) -> tuple[str, dict, Check]:
    """Make the query end with LIMIT <= MAX_LIMIT: append, lower, or keep it.

    A missing or too large LIMIT is fixed rather than rejected; the check says
    what changed so the response stays honest about the query that ran.
    """
    query = remove_comments(cypher).strip().rstrip(";").rstrip()
    match = _FINAL_LIMIT.search(code.strip())
    if match is None:
        return (f"{query}\nLIMIT {MAX_LIMIT}", params,
                Check("row_limit", True, f"no final LIMIT: appended LIMIT {MAX_LIMIT}"))
    value = match.group(1)
    if value.startswith("$"):
        name = value[1:]
        given = params.get(name)
        if not isinstance(given, int) or isinstance(given, bool) or given < 0:
            return (query, params,
                    Check("row_limit", False, f"LIMIT ${name} is not a non-negative integer"))
        if given > MAX_LIMIT:
            return (query, {**params, name: MAX_LIMIT},
                    Check("row_limit", True, f"LIMIT ${name}={given} lowered to {MAX_LIMIT}"))
        return query, params, Check("row_limit", True, f"LIMIT ${name}={given}")
    if int(value) > MAX_LIMIT:
        lowered = re.sub(r"\d+\s*$", str(MAX_LIMIT), query)
        return (lowered, params,
                Check("row_limit", True, f"LIMIT {value} lowered to {MAX_LIMIT}"))
    return query, params, Check("row_limit", True, f"LIMIT {value}")


def validate(cypher: str, params: dict | None = None) -> Validation:
    """Run every static guardrail on a generated query.

    All checks run even after a failure, so the response lists every problem
    at once; the query may only be executed when `Validation.passed`.
    """
    params = dict(params or {})
    code = strip_literals(cypher)
    checks = [
        check_single_statement(code),
        check_read_opener(code),
        check_forbidden_keywords(code),
        check_namespaces(code),
        check_procedures(code, cypher),
        check_schema_names(code),
    ]
    final_cypher, final_params, limit_check = enforce_limit(cypher, code, params)
    checks.append(limit_check)
    return Validation(cypher=final_cypher, params=final_params, checks=checks)
