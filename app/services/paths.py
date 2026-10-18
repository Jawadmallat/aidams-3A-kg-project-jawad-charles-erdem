"""Graph evidence behind a recommendation, for the explanation endpoint.

Stage C's "Why was this recommended?" endpoint must explain with actual graph
paths, not free-form LLM text. These functions return the verifiable facts the
explanation is built from; the LLM's only job is to phrase them. No evidence
means the caller should say so rather than invent a reason.
"""

from neo4j import Driver

from scripts.movielens import LIKED_THRESHOLD

# Why are these two movies similar? The Jaccard evidence behind a SIMILAR
# relationship (or a fallback answer): how many users liked both, out of how
# many liked each, plus the genres the movies share through IN_GENRE paths.
SIMILARITY_EVIDENCE_QUERY = """
MATCH (a:Movie {movieId: $movieId}), (b:Movie {movieId: $otherId})
RETURN a.title AS title, b.title AS otherTitle,
       COUNT { (a)<-[r:RATED]-(u:User)-[r2:RATED]->(b)
               WHERE r.rating >= $liked AND r2.rating >= $liked } AS sharedLikers,
       COUNT { (a)<-[r:RATED]-(:User) WHERE r.rating >= $liked } AS likers,
       COUNT { (b)<-[r:RATED]-(:User) WHERE r.rating >= $liked } AS otherLikers,
       [(a)-[:IN_GENRE]->(g:Genre)<-[:IN_GENRE]-(b) | g.name] AS sharedGenres
"""

# Why this movie for this user? The like-minded users who liked it, each with
# the movies they share with the user as the connecting paths:
# (me)-[:RATED]->(shared)<-[:RATED]-(peer)-[:RATED]->(rec), all likes.
RECOMMENDATION_EVIDENCE_QUERY = """
MATCH (me:User {userId: $userId})-[r1:RATED]->(shared:Movie)
      <-[r2:RATED]-(peer:User)-[r3:RATED]->(rec:Movie {movieId: $movieId})
WHERE r1.rating >= $liked AND r2.rating >= $liked AND r3.rating >= $liked
WITH peer, collect(DISTINCT shared.title) AS sharedTitles
ORDER BY size(sharedTitles) DESC, peer.userId
LIMIT $limit
RETURN peer.userId AS peerId, size(sharedTitles) AS sharedCount,
       sharedTitles[..5] AS sharedTitles
"""

# The genre paths between a user's liked movies and the recommended movie:
# (me) likes (m)-[:IN_GENRE]->(g)<-[:IN_GENRE]-(rec).
GENRE_EVIDENCE_QUERY = """
MATCH (me:User {userId: $userId})-[r:RATED]->(m:Movie)
      -[:IN_GENRE]->(g:Genre)<-[:IN_GENRE]-(rec:Movie {movieId: $movieId})
WHERE r.rating >= $liked
WITH g.name AS genre, count(DISTINCT m) AS likedMoviesInGenre
ORDER BY likedMoviesInGenre DESC, genre
RETURN genre, likedMoviesInGenre
"""


def similarity_evidence(driver: Driver, movie_id: int, other_id: int) -> dict | None:
    """The facts behind "these two movies are similar"; None if either is unknown.

    sharedLikers over (likers + otherLikers - sharedLikers) restates the
    Jaccard score of the SIMILAR relationship, so the explanation can say
    "liked by N of the same people" and stay consistent with /similar.
    """
    records = driver.execute_query(
        SIMILARITY_EVIDENCE_QUERY, movieId=movie_id, otherId=other_id, liked=LIKED_THRESHOLD
    ).records
    return records[0].data() if records else None


def recommendation_evidence(
    driver: Driver, user_id: int, movie_id: int, limit: int = 5
) -> dict:
    """The paths behind "this movie suits this user": peers and shared genres.

    Both lists can be empty: that means there is no graph evidence, and the
    explanation endpoint must say so instead of inventing a reason.
    """
    peers = driver.execute_query(
        RECOMMENDATION_EVIDENCE_QUERY,
        userId=user_id, movieId=movie_id, liked=LIKED_THRESHOLD, limit=limit,
    ).records
    genres = driver.execute_query(
        GENRE_EVIDENCE_QUERY, userId=user_id, movieId=movie_id, liked=LIKED_THRESHOLD
    ).records
    return {
        "peers": [record.data() for record in peers],
        "sharedGenres": [record.data() for record in genres],
    }
