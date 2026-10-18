"""Movie queries: similarity (with its live fallback) and trending.

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
# live. The traversal is bounded by the movie's audience (under 10 likers by
# definition); the denominators read the likesCount property run_gds stores,
# because counting a popular candidate's raters per request dominated the cost
# (520,865 db hits for one movie on the ml-25m sample - review finding on
# PR #2, evidence in docs/performance.md). Like every GDS-derived value, the
# counts are as fresh as the last run_gds run.
SIMILAR_FALLBACK_QUERY = """
MATCH (m:Movie {movieId: $movieId})<-[r1:RATED]-(u:User)-[r2:RATED]->(other:Movie)
WHERE r1.rating >= $liked AND r2.rating >= $liked
WITH m, other, count(DISTINCT u) AS shared
WITH other, shared, toFloat(shared) /
     (coalesce(m.likesCount, 0) + coalesce(other.likesCount, 0) - shared) AS score
ORDER BY score DESC, shared DESC, other.movieId
LIMIT $limit
RETURN other.movieId AS movieId, other.title AS title, other.year AS year,
       other.genres AS genres, round(score, 4) AS score
"""


# "Recent" is anchored on the newest rating in the data, not on today: the
# small dataset ends in 2018, so a window ending now would be empty
# (docs/decisions.md). The IS NOT NULL predicate is what lets the planner
# serve ORDER BY from the rated_timestamp range index: without it the plan
# is a full relationship scan (1,265,961 db hits on the ml-25m sample);
# with it, a DirectedRelationshipIndexScan reading 2 db hits. Evidence in
# docs/performance.md.
ANCHOR_QUERY = """
MATCH ()-[r:RATED]->()
WHERE r.timestamp IS NOT NULL
RETURN r.timestamp AS anchor
ORDER BY r.timestamp DESC
LIMIT 1
"""

# Trending blends the trend signal (rating volume inside the window) with
# structural importance: score = volume x (1 + pagerank). A movie without
# GDS results (missing pagerank counts as 0) can still trend on volume alone.
TRENDING_QUERY = """
WITH $anchor - $days * 86400 AS cutoff
MATCH (:User)-[r:RATED]->(m:Movie)
WHERE r.timestamp >= cutoff
WITH m, count(r) AS recentRatings
RETURN m.movieId AS movieId, m.title AS title, m.year AS year, m.genres AS genres,
       recentRatings, round(coalesce(m.pagerank, 0.0), 4) AS pagerank,
       round(recentRatings * (1 + coalesce(m.pagerank, 0.0)), 3) AS score
ORDER BY score DESC, movieId
SKIP $skip LIMIT $limit
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


def trending(driver: Driver, days: int, skip: int, limit: int) -> dict:
    """Popular and central movies of the window ending at the newest rating.

    The anchor is returned so "recent" is transparent: on the development
    dataset it is September 2018, not today.
    """
    anchor_records = driver.execute_query(ANCHOR_QUERY).records
    if not anchor_records:  # empty graph: nothing rated yet
        return {"anchor": None, "windowDays": days, "results": []}

    anchor = anchor_records[0]["anchor"]
    records = driver.execute_query(
        TRENDING_QUERY, anchor=anchor, days=days, skip=skip, limit=limit
    ).records
    return {
        "anchor": anchor,
        "windowDays": days,
        "results": [record.data() for record in records],
    }
