"""Movie queries: similarity now, trending later in the stage.

Services return plain dicts (the driver's record data); routers turn them into
the response models. Every function takes the driver first, like run_gds, so
scripts and Stage C can reuse them directly.
"""

from neo4j import Driver

from scripts.movielens import LIKED_THRESHOLD

MOVIE_QUERY = """
MATCH (m:Movie {movieId: $movieId})
RETURN m.movieId AS movieId, m.title AS title
"""

# The precomputed path: each movie's top-10 nearest neighbours by Jaccard
# similarity of likers. SIMILAR is directed (b is in a's top 10), so it is
# always read outgoing (docs/decisions.md).
SIMILAR_QUERY = """
MATCH (:Movie {movieId: $movieId})-[s:SIMILAR]->(other:Movie)
RETURN other.movieId AS movieId, other.title AS title, other.year AS year,
       other.genres AS genres, s.score AS score
ORDER BY s.score DESC, other.movieId
LIMIT $limit
"""

# Fallback for movies with fewer than 10 likes, which get no SIMILAR from GDS:
# the same Jaccard measure (shared likers over the union of likers), computed
# live. Cost is bounded by the movie's audience, under 10 likers by definition.
SIMILAR_FALLBACK_QUERY = """
MATCH (m:Movie {movieId: $movieId})
WITH m, COUNT { (m)<-[r:RATED]-(:User) WHERE r.rating >= $liked } AS mLikes
MATCH (m)<-[r1:RATED]-(u:User)-[r2:RATED]->(other:Movie)
WHERE r1.rating >= $liked AND r2.rating >= $liked
WITH other, mLikes, count(DISTINCT u) AS shared
WITH other, shared, toFloat(shared) /
     (mLikes + COUNT { (other)<-[r:RATED]-(:User) WHERE r.rating >= $liked } - shared) AS score
ORDER BY score DESC, shared DESC, other.movieId
LIMIT $limit
RETURN other.movieId AS movieId, other.title AS title, other.year AS year,
       other.genres AS genres, round(score, 4) AS score
"""


def find_movie(driver: Driver, movie_id: int) -> dict | None:
    """Return a movie's id and title, or None when the id is unknown."""
    records = driver.execute_query(MOVIE_QUERY, movieId=movie_id).records
    return records[0].data() if records else None


def similar_movies(driver: Driver, movie_id: int, limit: int) -> dict | None:
    """Movies most similar to one movie; None when the movie does not exist.

    Reads the precomputed SIMILAR relationships and falls back to the live
    co-rating Jaccard for long-tail movies; `source` names the branch taken.
    """
    movie = find_movie(driver, movie_id)
    if movie is None:
        return None

    records = driver.execute_query(SIMILAR_QUERY, movieId=movie_id, limit=limit).records
    source = "gds"
    if not records:
        records = driver.execute_query(
            SIMILAR_FALLBACK_QUERY, movieId=movie_id, limit=limit, liked=LIKED_THRESHOLD
        ).records
        source = "co_rating"
    return {**movie, "source": source, "results": [record.data() for record in records]}
