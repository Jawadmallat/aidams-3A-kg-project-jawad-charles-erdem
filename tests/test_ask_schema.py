"""The schema the NL-to-Cypher prompt and guardrails rely on.

The unit tests check the prompt text covers every constant; the integration
test checks the constants against a seeded database, so a schema change in
Part A's scripts fails here instead of silently confusing the LLM.
"""

from app.services import ask_schema


def test_description_names_every_label_relationship_and_index():
    """Nothing the guardrails accept is missing from what the model is told."""
    text = ask_schema.describe_schema()
    for label in ask_schema.NODE_PROPERTIES:
        assert f"(:{label})" in text
    for rel_type in ask_schema.RELATIONSHIPS:
        assert f"[:{rel_type}]" in text
    for index in ask_schema.FULLTEXT_INDEXES:
        assert index in text


def test_description_states_the_project_conventions():
    """The conventions a generic prompt gets wrong are spelled out."""
    text = ask_schema.describe_schema()
    assert "rating >= 4.0" in text
    assert "not from today" in text
    assert "movie_title_fulltext" in text


def test_relationship_ends_are_known_labels():
    """Every relationship connects labels the schema declares."""
    for relationship in ask_schema.RELATIONSHIPS.values():
        assert {relationship.start, relationship.end} <= set(ask_schema.NODE_PROPERTIES)


def test_constants_match_the_seeded_database(recommender_graph):
    """Labels, relationship types and full-text indexes match the live graph."""
    labels = {
        record["label"]
        for record in recommender_graph.execute_query("CALL db.labels() YIELD label").records
    }
    rel_types = {
        record["relationshipType"]
        for record in recommender_graph.execute_query(
            "CALL db.relationshipTypes() YIELD relationshipType"
        ).records
    }
    fulltext = {
        record["name"]
        for record in recommender_graph.execute_query(
            "SHOW FULLTEXT INDEXES YIELD name"
        ).records
    }
    assert labels == set(ask_schema.NODE_PROPERTIES)
    assert rel_types == set(ask_schema.RELATIONSHIPS)
    assert fulltext == set(ask_schema.FULLTEXT_INDEXES)
