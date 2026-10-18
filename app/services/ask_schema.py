"""The graph contract the NL-to-Cypher endpoint is allowed to use.

One source for both sides of /api/ask: the prompt describes the graph from
these constants, and the guardrails reject any label, relationship type,
procedure or index not listed here. They restate docs/decisions.md (the
contract owned by Part A); an integration test compares them with the live
database so the two cannot drift apart silently.
"""

from typing import NamedTuple

from scripts.movielens import LIKED_THRESHOLD

# Node labels and the properties each one carries.
NODE_PROPERTIES: dict[str, dict[str, str]] = {
    "User": {
        "userId": "integer, unique",
        "community": "integer or absent; Louvain community of the user's liked movies",
    },
    "Movie": {
        "movieId": "integer, unique",
        "title": "string without the year, article in front ('The Matrix'); NOT unique",
        "year": "integer or null",
        "genres": "list of strings, same values as the IN_GENRE links",
        "imdbId": "string like 'tt0133093'",
        "tmdbId": "integer or null",
        "community": "integer or absent (movies with fewer than 10 likes)",
        "pagerank": "float or absent (treat absent as 0)",
    },
    "Genre": {"name": "string, unique, e.g. 'Comedy', 'Sci-Fi', 'IMAX'"},
}



class Relationship(NamedTuple):
    """A relationship type: the labels it connects and its properties."""

    start: str
    end: str
    properties: dict[str, str]


RELATIONSHIPS: dict[str, Relationship] = {
    "RATED": Relationship("User", "Movie", {
        "rating": "float, 0.5 to 5.0 in half stars",
        "timestamp": "integer, Unix epoch seconds",
    }),
    "TAGGED": Relationship("User", "Movie", {
        "tag": "string, lower-cased",
        "timestamp": "integer, Unix epoch seconds",
    }),
    "IN_GENRE": Relationship("Movie", "Genre", {}),
    "SIMILAR": Relationship("Movie", "Movie", {
        "score": "float in (0, 1], Jaccard of the users who liked both; top 10 per movie",
    }),
}

# Full-text indexes, queried through the one procedure the guardrails allow.
FULLTEXT_INDEXES: dict[str, str] = {
    "movie_title_fulltext": "Movie.title: find movies by title words",
    "tagged_tag_fulltext": "TAGGED.tag: find tag applications by words",
}

# Procedures generated queries may CALL; anything else is rejected.
ALLOWED_PROCEDURES = frozenset({
    "db.index.fulltext.queryNodes",
    "db.index.fulltext.queryRelationships",
})

# The query conventions of docs/decisions.md that change what a correct
# answer is; a generic text-to-Cypher prompt would get each of them wrong.
CONVENTIONS = (
    "Ids are integers: {movieId: 1}, never {movieId: '1'}.",
    f"\"Liked\" means a RATED relationship with rating >= {LIKED_THRESHOLD}.",
    "\"Recent\" is measured from the newest RATED.timestamp in the data, not from "
    "today (the data ends in 2018): anchor windows on max(r.timestamp).",
    "Titles are not unique and have no year. Find a movie by title with "
    "CALL db.index.fulltext.queryNodes('movie_title_fulltext', $text) "
    "YIELD node, score, and filter on node.year when the year is known.",
    "Filter a movie you already matched by genre with 'Comedy' IN m.genres; use "
    "(m)-[:IN_GENRE]->(:Genre {name: ...}) when the genre is part of the path.",
    "Tags are a property of TAGGED, not nodes: (u:User)-[t:TAGGED]->(m:Movie) "
    "WHERE t.tag = 'dark comedy'.",
    "SIMILAR is directed and only exists for movies with at least 10 likes: "
    "query (m)-[s:SIMILAR]->(other) ORDER BY s.score DESC.",
)


def describe_schema() -> str:
    """The schema and conventions as prompt text, generated from the constants."""
    lines = ["Node labels and properties:"]
    for label, properties in NODE_PROPERTIES.items():
        lines.append(f"  (:{label})")
        lines += [f"    {name}: {kind}" for name, kind in properties.items()]
    lines.append("Relationships:")
    for rel_type, (start, end, properties) in RELATIONSHIPS.items():
        lines.append(f"  (:{start})-[:{rel_type}]->(:{end})")
        lines += [f"    {name}: {kind}" for name, kind in properties.items()]
    lines.append("Full-text indexes:")
    lines += [f"  {name}: {use}" for name, use in FULLTEXT_INDEXES.items()]
    lines.append("Conventions:")
    lines += [f"  - {rule}" for rule in CONVENTIONS]
    return "\n".join(lines)
