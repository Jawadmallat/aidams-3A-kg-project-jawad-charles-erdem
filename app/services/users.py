"""User queries: personalised recommendations now, discover later in the stage."""

from neo4j import Driver

from scripts.movielens import LIKED_THRESHOLD

# How much one shared top genre multiplies a recommendation's score: a movie
# matching 2 of the user's top 3 genres scores 1.5x its peer evidence.
GENRE_BOOST = 0.25

# How many of the most similar users vote. Bounds the traversal on large
# datasets: the peer stage reads at most 25 users' ratings, however popular
# the user's movies are.
PEER_LIMIT = 25

USER_QUERY = """
MATCH (u:User {userId: $userId})
RETURN u.userId AS userId
"""

# Collaborative filtering in one statement, staged with scoped CALL {}
# subqueries (the project convention since run_gds.py):
#   1. the user's top 3 genres, walked through IN_GENRE nodes;
#   2. the 25 peers who share the most liked movies with the user;
#   3. the peers' liked movies the user has not rated, scored by the summed
#      overlap of the peers recommending them, boosted when the movie's
#      genres list intersects the user's top genres (the list filters, the
#      nodes make paths: docs/decisions.md).
RECOMMENDATIONS_QUERY = """
MATCH (me:User {userId: $userId})
CALL (me) {
  MATCH (me)-[r:RATED]->(:Movie)-[:IN_GENRE]->(g:Genre)
  WHERE r.rating >= $liked
  WITH g.name AS genre, count(*) AS likes
  ORDER BY likes DESC, genre
  LIMIT 3
  RETURN collect(genre) AS topGenres
}
CALL (me) {
  MATCH (me)-[r1:RATED]->(m:Movie)<-[r2:RATED]-(peer:User)
  WHERE r1.rating >= $liked AND r2.rating >= $liked
  WITH peer, count(m) AS overlap
  ORDER BY overlap DESC, peer.userId
  LIMIT $peerLimit
  RETURN collect({peer: peer, overlap: overlap}) AS peers
}
UNWIND peers AS entry
WITH me, topGenres, entry.peer AS peer, entry.overlap AS overlap
MATCH (peer)-[r:RATED]->(rec:Movie)
WHERE r.rating >= $liked AND NOT EXISTS { (me)-[:RATED]->(rec) }
WITH rec, topGenres, sum(overlap) AS peerScore, count(DISTINCT peer) AS peerCount
WITH rec, peerScore, peerCount,
     size([genre IN rec.genres WHERE genre IN topGenres]) AS genreMatches
RETURN rec.movieId AS movieId, rec.title AS title, rec.year AS year,
       rec.genres AS genres,
       round(peerScore * (1 + $genreBoost * genreMatches), 3) AS score,
       peerCount, genreMatches
ORDER BY score DESC, movieId
LIMIT $limit
"""


def find_user(driver: Driver, user_id: int) -> dict | None:
    """Return a user's id, or None when the id is unknown."""
    records = driver.execute_query(USER_QUERY, userId=user_id).records
    return records[0].data() if records else None


def recommendations(driver: Driver, user_id: int, limit: int) -> dict | None:
    """Personal picks for one user; None when the user does not exist.

    A user who has rated everything (or has no peers) gets an empty list,
    not an error: the user is known, there is just nothing left to recommend.
    """
    user = find_user(driver, user_id)
    if user is None:
        return None

    records = driver.execute_query(
        RECOMMENDATIONS_QUERY,
        userId=user_id,
        limit=limit,
        liked=LIKED_THRESHOLD,
        genreBoost=GENRE_BOOST,
        peerLimit=PEER_LIMIT,
    ).records
    return {**user, "results": [record.data() for record in records]}
