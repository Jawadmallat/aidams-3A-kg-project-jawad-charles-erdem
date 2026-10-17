"""Tests for the graph constraints and indexes."""

import re

from scripts.schema import CONSTRAINTS, INDEXES, apply_schema, existing_schema


def test_every_statement_is_idempotent_and_named_after_its_key():
    """Each statement uses IF NOT EXISTS and creates the object named by its key."""
    for name, statement in {**CONSTRAINTS, **INDEXES}.items():
        assert "IF NOT EXISTS" in statement
        assert re.search(rf"\b(CONSTRAINT|INDEX) {name} IF NOT EXISTS", statement)


def test_required_keys_are_constrained():
    """The three MovieLens keys are unique; titles are deliberately not."""
    statements = " ".join(CONSTRAINTS.values())
    for key in ("u.userId", "m.movieId", "g.name"):
        assert f"REQUIRE {key} IS UNIQUE" in statements
    assert "m.title IS UNIQUE" not in statements


def test_apply_schema_twice_on_neo4j(neo4j_driver):
    """Applying the schema twice succeeds and leaves every object in place."""
    apply_schema(neo4j_driver)
    applied = apply_schema(neo4j_driver)
    assert set(applied) <= existing_schema(neo4j_driver)
