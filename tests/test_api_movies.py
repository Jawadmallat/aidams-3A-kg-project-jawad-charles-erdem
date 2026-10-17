"""API tests for the movie endpoints, against the committed fixture.

Fixture facts used here (tests/fixtures/movielens, see docs/decisions.md):
movie 1 is Toy Story and 2571 is The Matrix; after run_gds, 29 of the 50
movies have SIMILAR relationships and the other 21 exercise the fallback.
"""


def movie_without_similar(driver) -> int:
    """A movie that has liked ratings but no outgoing SIMILAR (fallback case)."""
    return driver.execute_query(
        "MATCH (m:Movie) WHERE NOT (m)-[:SIMILAR]->() "
        "AND COUNT { (m)<-[r:RATED]-() WHERE r.rating >= 4.0 } >= 2 "
        "RETURN m.movieId AS id ORDER BY id LIMIT 1"
    ).records[0]["id"]


def test_similar_reads_the_precomputed_gds_results(api_client):
    """A movie with SIMILAR answers from GDS, scores descending in (0, 1]."""
    response = api_client.get("/api/movies/2571/similar")
    assert response.status_code == 200
    body = response.json()
    assert (body["movieId"], body["title"], body["source"]) == (2571, "The Matrix", "gds")
    scores = [result["score"] for result in body["results"]]
    assert 0 < len(scores) <= 10
    assert scores == sorted(scores, reverse=True)
    assert all(0 < score <= 1 for score in scores)


def test_similar_falls_back_to_co_rating(api_client, recommender_graph):
    """A movie without SIMILAR answers from the live Jaccard fallback."""
    movie_id = movie_without_similar(recommender_graph)
    response = api_client.get(f"/api/movies/{movie_id}/similar")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "co_rating"
    assert body["results"], "the dense fixture guarantees co-raters"
    assert all(0 < result["score"] <= 1 for result in body["results"])


def test_similar_respects_the_limit(api_client):
    """limit caps the number of results."""
    response = api_client.get("/api/movies/2571/similar?limit=3")
    assert response.status_code == 200
    assert len(response.json()["results"]) <= 3


def test_similar_unknown_movie_is_404(api_client):
    """An id that is not in the graph names the movie in a 404."""
    response = api_client.get("/api/movies/999999/similar")
    assert response.status_code == 404
    assert "999999" in response.json()["detail"]
