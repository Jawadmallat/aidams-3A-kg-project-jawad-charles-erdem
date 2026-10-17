"""Integration tests: load the committed fixture into a disposable Neo4j."""

from scripts.seed_data import build_tag_rows, graph_counts, seed


def expected_counts(frames):
    """Counts the graph must hold after loading the fixture."""
    return {
        "users": 20,
        "movies": 50,
        "genres": 17,
        "ratings": len(frames["ratings"]),
        "tags": len(build_tag_rows(frames["tags"])),
        "inGenre": sum(
            len([g for g in raw.split("|") if g != "(no genres listed)"])
            for raw in frames["movies"].genres
        ),
    }


def test_fixture_loads_with_expected_counts(seeded_graph, fixture_frames):
    """Every node and relationship of the fixture lands in the graph."""
    assert graph_counts(seeded_graph) == expected_counts(fixture_frames)


def test_seeding_twice_creates_no_duplicates(seeded_graph, fixture_frames):
    """Re-running the seed leaves every count unchanged."""
    before = graph_counts(seeded_graph)
    seed(seeded_graph, fixture_frames)
    assert graph_counts(seeded_graph) == before


def test_one_rating_per_user_and_movie(seeded_graph):
    """MERGE never duplicates a RATED relationship between the same pair."""
    records = seeded_graph.execute_query(
        "MATCH (u:User)-[r:RATED]->(m:Movie) "
        "WITH u, m, count(r) AS n WHERE n > 1 RETURN count(*) AS duplicates"
    ).records
    assert records[0]["duplicates"] == 0


def test_movie_properties_and_genre_nodes_agree(seeded_graph):
    """Toy Story is cleaned, and its genre list matches its IN_GENRE links."""
    record = seeded_graph.execute_query(
        "MATCH (m:Movie {movieId: 1}) "
        "RETURN m.title AS title, m.year AS year, m.imdbId AS imdbId, m.genres AS genres, "
        "COLLECT { MATCH (m)-[:IN_GENRE]->(g) RETURN g.name ORDER BY g.name } AS linked"
    ).records[0]
    assert (record["title"], record["year"], record["imdbId"]) == ("Toy Story", 1995, "tt0114709")
    assert sorted(record["genres"]) == record["linked"]


def test_fulltext_index_finds_cleaned_title(seeded_graph):
    """The full-text index matches the article-first title."""
    records = seeded_graph.execute_query(
        "CALL db.index.fulltext.queryNodes('movie_title_fulltext', 'matrix') "
        "YIELD node RETURN node.title AS title LIMIT 1"
    ).records
    assert records[0]["title"] == "The Matrix"
