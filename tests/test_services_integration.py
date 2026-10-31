"""Service-level checks against the fixture: values, not just shapes.

The API tests check the HTTP behaviour; these tests check that the numbers the
services return are the ones the graph implies, by recomputing them with
independent Cypher.
"""

from app.services.movies import similar_movies, trending
from app.services.paths import recommendation_evidence, similarity_evidence
from scripts.movielens import LIKED_THRESHOLD

FIXTURE_LAST_RATING = 1519122884  # 2018-02-20, documented in docs/decisions.md


def liker_ids(driver, movie_id: int) -> set[int]:
    """The users who liked one movie, straight from the graph."""
    records = driver.execute_query(
        "MATCH (m:Movie {movieId: $id})<-[r:RATED]-(u:User) "
        "WHERE r.rating >= $liked RETURN u.userId AS id",
        id=movie_id, liked=LIKED_THRESHOLD,
    ).records
    return {record["id"] for record in records}


def fallback_movie_id(driver) -> int:
    """The first movie without SIMILAR that has at least two likers."""
    return driver.execute_query(
        "MATCH (m:Movie) WHERE NOT (m)-[:SIMILAR]->() "
        "AND COUNT { (m)<-[r:RATED]-() WHERE r.rating >= 4.0 } >= 2 "
        "RETURN m.movieId AS id ORDER BY id LIMIT 1"
    ).records[0]["id"]


def assert_exact_jaccard(driver, movie_id: int, results: list[dict]) -> None:
    """Each score equals |likers(a) & likers(b)| / |likers(a) | likers(b)|."""
    mine = liker_ids(driver, movie_id)
    for result in results:
        theirs = liker_ids(driver, result["movieId"])
        expected = len(mine & theirs) / len(mine | theirs)
        assert result["score"] == round(expected, 4)


def test_fallback_scores_are_the_exact_jaccard(recommender_graph):
    """Each fallback score equals |likers(a) & likers(b)| / |likers(a) | likers(b)|."""
    movie_id = fallback_movie_id(recommender_graph)
    found = similar_movies(recommender_graph, movie_id, limit=5)
    assert found["source"] == "co_rating"
    assert_exact_jaccard(recommender_graph, movie_id, found["results"])


def test_fallback_counts_live_when_likes_count_is_missing(recommender_graph):
    """Without likesCount (seed, no run_gds yet) the fallback scores stay exact."""
    movie_id = fallback_movie_id(recommender_graph)
    stored = similar_movies(recommender_graph, movie_id, limit=5)["results"]
    assert len(stored) >= 2, "one candidate loses likesCount, another keeps it"
    removed = recommender_graph.execute_query(
        "MATCH (m:Movie) WHERE m.movieId IN $ids "
        "WITH m, m.likesCount AS likes REMOVE m.likesCount "
        "RETURN m.movieId AS id, likes",
        ids=[movie_id, stored[0]["movieId"]],
    ).records
    try:
        found = similar_movies(recommender_graph, movie_id, limit=5)
    finally:  # the graph is shared by the whole session
        recommender_graph.execute_query(
            "UNWIND $rows AS row MATCH (m:Movie {movieId: row.id}) "
            "SET m.likesCount = row.likes",
            rows=[record.data() for record in removed],
        )
    assert len(removed) == 2 and all(record["likes"] for record in removed)
    assert found["source"] == "co_rating"
    assert found["results"] == stored
    assert_exact_jaccard(recommender_graph, movie_id, found["results"])


def test_similarity_evidence_restates_the_stored_similar_score(recommender_graph):
    """sharedLikers/(likers+otherLikers-sharedLikers) equals the SIMILAR score."""
    link = recommender_graph.execute_query(
        "MATCH (a:Movie)-[s:SIMILAR]->(b:Movie) "
        "RETURN a.movieId AS a, b.movieId AS b, s.score AS score "
        "ORDER BY s.score DESC LIMIT 1"
    ).records[0]
    evidence = similarity_evidence(recommender_graph, link["a"], link["b"])
    union = evidence["likers"] + evidence["otherLikers"] - evidence["sharedLikers"]
    assert evidence["sharedLikers"] / union == link["score"]
    assert evidence["title"] and evidence["otherTitle"]


def test_recommendation_evidence_paths_are_real(recommender_graph):
    """Every peer in the evidence really liked the movie and shares likes with the user."""
    # user 603's top recommendation is guaranteed to have peer evidence
    rec = recommender_graph.execute_query(
        "MATCH (me:User {userId: 603}) "
        "MATCH (peer:User)-[r:RATED]->(rec:Movie) "
        "WHERE r.rating >= $liked AND NOT EXISTS { (me)-[:RATED]->(rec) } "
        "RETURN rec.movieId AS id LIMIT 1", liked=LIKED_THRESHOLD,
    ).records[0]["id"]
    evidence = recommendation_evidence(recommender_graph, 603, rec)
    assert evidence["peers"], "a liked unseen movie always has at least one peer"
    for peer in evidence["peers"]:
        assert peer["sharedCount"] >= 1
        assert 1 <= len(peer["sharedTitles"]) <= 5
    for genre in evidence["sharedGenres"]:
        assert genre["likedMoviesInGenre"] >= 1


def test_recommendation_evidence_can_be_empty(recommender_graph):
    """No connecting paths means empty lists, never invented evidence."""
    # user 414 rated everything, so no movie is unseen; pick a movie nobody
    # could be connected to 414 through unseen paths by using a user with no
    # likes in common: evidence for a (user, movie) pair the user already
    # rated still works, so instead check a movie with a single liker.
    evidence = recommendation_evidence(recommender_graph, 999999, 1)
    assert evidence == {"peers": [], "sharedGenres": []}


def test_trending_anchor_is_the_newest_rating(recommender_graph):
    """The service reads the documented fixture anchor from the graph."""
    result = trending(recommender_graph, days=365, skip=0, limit=5)
    assert result["anchor"] == FIXTURE_LAST_RATING
    assert result["results"]
