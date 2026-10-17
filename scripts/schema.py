"""Constraints and indexes of the movie graph.

Every statement uses IF NOT EXISTS, so applying the schema is safe to repeat.
The seed script applies it before loading data: the uniqueness constraints are
what make its MERGE lookups index-backed instead of full label scans.

Usage:
    uv run --env-file .env python -m scripts.schema
"""

import sys

from neo4j import Driver

from scripts.db import get_driver

CONSTRAINTS = {
    # Primary keys from MovieLens. Each one also creates the index MERGE relies on.
    "user_id_unique": "CREATE CONSTRAINT user_id_unique IF NOT EXISTS "
                      "FOR (u:User) REQUIRE u.userId IS UNIQUE",
    "movie_id_unique": "CREATE CONSTRAINT movie_id_unique IF NOT EXISTS "
                       "FOR (m:Movie) REQUIRE m.movieId IS UNIQUE",
    "genre_name_unique": "CREATE CONSTRAINT genre_name_unique IF NOT EXISTS "
                         "FOR (g:Genre) REQUIRE g.name IS UNIQUE",
}

INDEXES = {
    # Title search by words ("matrix" finds "The Matrix"); titles are not unique,
    # so this is a search index, not a key.
    "movie_title_fulltext": "CREATE FULLTEXT INDEX movie_title_fulltext IF NOT EXISTS "
                            "FOR (m:Movie) ON EACH [m.title]",
    # Release-year filters and ranges ("movies from the 1990s").
    "movie_year": "CREATE INDEX movie_year IF NOT EXISTS FOR (m:Movie) ON (m.year)",
    # Time windows over ratings: trending volume and the temporal train/test split.
    "rated_timestamp": "CREATE INDEX rated_timestamp IF NOT EXISTS "
                       "FOR ()-[r:RATED]-() ON (r.timestamp)",
    # "Movies tagged X" without scanning every TAGGED relationship.
    "tagged_tag": "CREATE INDEX tagged_tag IF NOT EXISTS FOR ()-[t:TAGGED]-() ON (t.tag)",
    # Word search inside tags ("dark" finds "dark comedy"), for free-text questions.
    # Idea taken from the AI-generated schema (docs/anti_genericity.md).
    "tagged_tag_fulltext": "CREATE FULLTEXT INDEX tagged_tag_fulltext IF NOT EXISTS "
                           "FOR ()-[t:TAGGED]-() ON EACH [t.tag]",
    # Written by scripts/run_gds.py. Community lookups ("other movies in this
    # user's community") and PageRank ordering for trending.
    "movie_community": "CREATE INDEX movie_community IF NOT EXISTS FOR (m:Movie) ON (m.community)",
    "movie_pagerank": "CREATE INDEX movie_pagerank IF NOT EXISTS FOR (m:Movie) ON (m.pagerank)",
    "user_community": "CREATE INDEX user_community IF NOT EXISTS FOR (u:User) ON (u.community)",
}


def apply_schema(driver: Driver) -> list[str]:
    """Create every constraint, then every index; return the names applied."""
    applied = []
    for name, statement in {**CONSTRAINTS, **INDEXES}.items():
        driver.execute_query(statement)
        applied.append(name)
    driver.execute_query("CALL db.awaitIndexes(300)")
    return applied


def existing_schema(driver: Driver) -> set[str]:
    """Return the names of the constraints and indexes present in the database."""
    constraints = driver.execute_query("SHOW CONSTRAINTS YIELD name").records
    indexes = driver.execute_query("SHOW INDEXES YIELD name").records
    return {record["name"] for record in [*constraints, *indexes]}


def main() -> int:
    """Apply the schema and report what exists afterwards."""
    with get_driver() as driver:
        applied = apply_schema(driver)
        present = existing_schema(driver)
    for name in applied:
        print(f"{'ok' if name in present else 'MISSING'}  {name}")
    return 0 if set(applied) <= present else 1


if __name__ == "__main__":
    sys.exit(main())
