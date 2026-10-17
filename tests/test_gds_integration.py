"""Integration tests for scripts/run_gds.py on the committed fixture."""

from collections import Counter

import pytest

from scripts.movielens import LIKED_THRESHOLD
from scripts.run_gds import DEGREE_CUTOFF, TOP_K, run_gds


@pytest.fixture(name="gds_graph", scope="module")
def fixture_gds_graph(seeded_graph):
    """The seeded test graph with GDS results written, plus the run statistics."""
    return seeded_graph, run_gds(seeded_graph)


def rows(driver, query):
    """Run a read query and return its records as dicts."""
    return [record.data() for record in driver.execute_query(query).records]


def eligible_movies(fixture_frames):
    """Movies with at least DEGREE_CUTOFF likes: the only ones Node Similarity compares."""
    ratings = fixture_frames["ratings"]
    likes = ratings[ratings.rating >= LIKED_THRESHOLD].groupby("movieId").size()
    return set(likes[likes >= DEGREE_CUTOFF].index)


def test_similar_links_only_eligible_movies(gds_graph, fixture_frames):
    """SIMILAR connects distinct eligible movies, at most TOP_K per movie, with scores in (0, 1]."""
    driver, _ = gds_graph
    links = rows(driver, "MATCH (a:Movie)-[s:SIMILAR]->(b:Movie) "
                         "RETURN a.movieId AS a, b.movieId AS b, s.score AS score")
    eligible = eligible_movies(fixture_frames)
    assert links
    assert {link["a"] for link in links} == eligible
    assert {link["b"] for link in links} <= eligible
    assert all(link["a"] != link["b"] for link in links)
    assert all(0 < link["score"] <= 1 for link in links)
    assert max(Counter(link["a"] for link in links).values()) <= TOP_K


def test_community_and_pagerank_written_on_eligible_movies(gds_graph, fixture_frames):
    """Exactly the movies in the similarity graph get a community and a PageRank."""
    driver, stats = gds_graph
    scored = rows(driver, "MATCH (m:Movie) WHERE m.community IS NOT NULL "
                          "AND m.pagerank IS NOT NULL RETURN m.movieId AS id, m.community AS c")
    assert {row["id"] for row in scored} == eligible_movies(fixture_frames)
    assert len({row["c"] for row in scored}) == stats["louvain"]["communityCount"]


def test_user_community_is_their_most_liked_community(gds_graph):
    """Each user's community is the most frequent one among their liked movies."""
    driver, _ = gds_graph
    likes = rows(driver, f"""
        MATCH (u:User)-[r:RATED]->(m:Movie)
        WHERE r.rating >= {LIKED_THRESHOLD} AND m.community IS NOT NULL
        RETURN u.userId AS user, u.community AS assigned, collect(m.community) AS communities
    """)
    assert likes
    for row in likes:
        counts = Counter(row["communities"])
        best = max(counts.values())
        assert row["assigned"] == min(c for c, n in counts.items() if n == best)


def test_rerun_gives_identical_results(gds_graph):
    """Running GDS again replaces, rather than adds to, SIMILAR and the properties."""
    driver, _ = gds_graph
    snapshot = ("MATCH (m:Movie) RETURN m.movieId AS id, m.community AS c, "
                "m.pagerank AS pr, COUNT { (m)-[:SIMILAR]->() } AS k ORDER BY id")
    before = rows(driver, snapshot)
    run_gds(driver)
    assert rows(driver, snapshot) == before
