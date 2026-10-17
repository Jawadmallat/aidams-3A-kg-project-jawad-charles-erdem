"""Run the Graph Data Science algorithms and write their results to the graph.

1. Node Similarity on the "liked" bipartite graph (Movie -> User, rating >= 4.0)
   writes each movie's top-k neighbours as (:Movie)-[:SIMILAR {score}]->(:Movie).
2. Louvain on SIMILAR (undirected, weighted by score) writes Movie.community.
3. PageRank on SIMILAR (directed, weighted by score) writes Movie.pagerank:
   a movie ranks high when many movies list it among their nearest neighbours.
4. Cypher derives User.community: the community most of a user's liked movies
   belong to.

Projections are built in movieId order and Louvain runs single-threaded, so the
same graph always yields the same communities. Previous results are removed
first, so the script is safe to re-run. Movies
with fewer than DEGREE_CUTOFF likes get no SIMILAR, community or pagerank.

Usage:
    uv run --env-file .env python -m scripts.run_gds
"""

import sys
import time

from neo4j import Driver

from scripts.db import get_driver
from scripts.movielens import LIKED_THRESHOLD
from scripts.schema import apply_schema

LIKES_GRAPH = "movie-likes"
SIMILARITY_GRAPH = "movie-similarity"
TOP_K = 10          # neighbours kept per movie: keeps SIMILAR at about 10 x movies
# Minimum likes for a movie to be compared. At 5, niche movies liked by a few very
# active users became hubs that dozens of movies listed as "similar" (PageRank no
# longer tracked popularity at all); 10 removes that while keeping 74% of likes.
# See demo.ipynb, sections 1.4 and 3.
DEGREE_CUTOFF = 10

CLEAR_SIMILAR = """
MATCH ()-[s:SIMILAR]->()
CALL (s) { DELETE s } IN TRANSACTIONS OF 10000 ROWS
"""

CLEAR_PROPERTIES = """
MATCH (n) WHERE n:Movie OR n:User
CALL (n) { REMOVE n.community, n.pagerank } IN TRANSACTIONS OF 10000 ROWS
"""

PROJECT_LIKES = """
MATCH (m:Movie)<-[r:RATED]-(u:User)
WHERE r.rating >= $liked
WITH m, u ORDER BY m.movieId, u.userId
WITH gds.graph.project($graph, m, u, {sourceNodeLabels: 'Movie', targetNodeLabels: 'User'}) AS g
RETURN g.nodeCount AS nodes, g.relationshipCount AS relationships
"""

NODE_SIMILARITY = """
CALL gds.nodeSimilarity.write($graph, {
  topK: $topK,
  degreeCutoff: $degreeCutoff,
  writeRelationshipType: 'SIMILAR',
  writeProperty: 'score'
})
YIELD nodesCompared, relationshipsWritten
RETURN nodesCompared, relationshipsWritten
"""

PROJECT_SIMILARITY = """
MATCH (a:Movie)-[s:SIMILAR]->(b:Movie)
WITH a, b, s ORDER BY a.movieId, b.movieId
WITH gds.graph.project($graph, a, b, {
  sourceNodeLabels: 'Movie', targetNodeLabels: 'Movie',
  relationshipType: 'SIMILAR', relationshipProperties: {score: s.score}
}) AS g
RETURN g.nodeCount AS nodes, g.relationshipCount AS relationships
"""

TO_UNDIRECTED = """
CALL gds.graph.relationships.toUndirected($graph, {
  relationshipType: 'SIMILAR',
  mutateRelationshipType: 'SIMILAR_UNDIRECTED',
  aggregation: 'MAX'
})
YIELD relationshipsWritten
RETURN relationshipsWritten
"""

LOUVAIN = """
CALL gds.louvain.write($graph, {
  relationshipTypes: ['SIMILAR_UNDIRECTED'],
  relationshipWeightProperty: 'score',
  writeProperty: 'community',
  consecutiveIds: true,
  concurrency: 1
})
YIELD communityCount, modularity
RETURN communityCount, modularity
"""

PAGERANK = """
CALL gds.pageRank.write($graph, {
  relationshipTypes: ['SIMILAR'],
  relationshipWeightProperty: 'score',
  writeProperty: 'pagerank',
  dampingFactor: 0.85,
  maxIterations: 20
})
YIELD nodePropertiesWritten, ranIterations
RETURN nodePropertiesWritten, ranIterations
"""

USER_COMMUNITY = """
MATCH (u:User)
CALL (u) {
  MATCH (u)-[r:RATED]->(m:Movie)
  WHERE r.rating >= $liked AND m.community IS NOT NULL
  WITH m.community AS community, count(*) AS likes
  ORDER BY likes DESC, community ASC
  LIMIT 1
  SET u.community = community
} IN TRANSACTIONS OF 1000 ROWS
"""

SUMMARY = """
CALL () { MATCH ()-[s:SIMILAR]->() RETURN count(s) AS similar }
CALL () { MATCH (m:Movie) WHERE m.community IS NOT NULL RETURN count(m) AS moviesInCommunities }
CALL () { MATCH (u:User) WHERE u.community IS NOT NULL RETURN count(u) AS usersInCommunities }
RETURN similar, moviesInCommunities, usersInCommunities
"""


def drop_projections(driver: Driver) -> None:
    """Remove this script's in-memory graphs if they exist."""
    for name in (LIKES_GRAPH, SIMILARITY_GRAPH):
        driver.execute_query("CALL gds.graph.drop($name, false) YIELD graphName", name=name)


def run_auto_commit(driver: Driver, query: str, **params) -> None:
    """Run a query that batches itself with CALL ... IN TRANSACTIONS."""
    with driver.session() as session:
        session.run(query, **params).consume()


def single(driver: Driver, query: str, **params) -> dict:
    """Run a query returning one row and give it back as a dict."""
    return driver.execute_query(query, **params).records[0].data()


def run_gds(driver: Driver) -> dict:
    """Recompute SIMILAR, communities and PageRank; return per-step statistics."""
    apply_schema(driver)
    drop_projections(driver)
    run_auto_commit(driver, CLEAR_SIMILAR)
    run_auto_commit(driver, CLEAR_PROPERTIES)

    stats = {}
    try:
        stats["likes_projection"] = single(
            driver, PROJECT_LIKES, graph=LIKES_GRAPH, liked=LIKED_THRESHOLD)
        stats["node_similarity"] = single(
            driver, NODE_SIMILARITY, graph=LIKES_GRAPH, topK=TOP_K, degreeCutoff=DEGREE_CUTOFF)
        stats["similarity_projection"] = single(
            driver, PROJECT_SIMILARITY, graph=SIMILARITY_GRAPH)
        single(driver, TO_UNDIRECTED, graph=SIMILARITY_GRAPH)
        stats["louvain"] = single(driver, LOUVAIN, graph=SIMILARITY_GRAPH)
        stats["pagerank"] = single(driver, PAGERANK, graph=SIMILARITY_GRAPH)
    finally:
        drop_projections(driver)

    run_auto_commit(driver, USER_COMMUNITY, liked=LIKED_THRESHOLD)
    stats["graph"] = single(driver, SUMMARY)
    return stats


def report(driver: Driver) -> None:
    """Print sanity checks: Toy Story's neighbours, largest communities, top PageRank."""
    neighbours = driver.execute_query(
        "MATCH (:Movie {movieId: 1})-[s:SIMILAR]->(m) "
        "RETURN m.title AS title, round(s.score, 3) AS score ORDER BY score DESC LIMIT 5"
    ).records
    print("\nMost similar to Toy Story:")
    for record in neighbours:
        print(f"  {record['score']:.3f}  {record['title']}")

    communities = driver.execute_query(
        "MATCH (m:Movie) WHERE m.community IS NOT NULL "
        "WITH m.community AS community, m ORDER BY m.pagerank DESC "
        "WITH community, count(m) AS size, collect(m.title)[..3] AS top "
        "RETURN community, size, top ORDER BY size DESC LIMIT 5"
    ).records
    print("\nLargest communities (top movies by PageRank):")
    for record in communities:
        print(f"  #{record['community']:<4} {record['size']:>5} movies  {', '.join(record['top'])}")

    central = driver.execute_query(
        "MATCH (m:Movie) WHERE m.pagerank IS NOT NULL "
        "RETURN m.title AS title, round(m.pagerank, 3) AS pagerank "
        "ORDER BY m.pagerank DESC LIMIT 5"
    ).records
    print("\nHighest PageRank:")
    for record in central:
        print(f"  {record['pagerank']:.3f}  {record['title']}")


def main() -> int:
    """Entry point."""
    started = time.perf_counter()
    with get_driver() as driver:
        stats = run_gds(driver)
        for step, values in stats.items():
            print(f"{step:<22} {values}")
        print(f"done in {time.perf_counter() - started:.1f}s")
        report(driver)
    return 0


if __name__ == "__main__":
    sys.exit(main())
