"""API tests for the user endpoints, against the committed fixture.

Fixture facts used here (docs/decisions.md): user 414 rated every movie of the
fixture, so nothing is left to recommend; user 603 has the fewest ratings (30);
after run_gds all 20 users have a community.
"""


def rated_movie_ids(driver, user_id: int) -> set[int]:
    """Every movie the user has rated, straight from the graph."""
    records = driver.execute_query(
        "MATCH (:User {userId: $id})-[:RATED]->(m) RETURN m.movieId AS id", id=user_id
    ).records
    return {record["id"] for record in records}


def test_recommendations_exclude_already_seen_movies(api_client, recommender_graph):
    """User 603 gets picks, none of which they have already rated."""
    response = api_client.get("/api/users/603/recommendations")
    assert response.status_code == 200
    body = response.json()
    assert body["userId"] == 603
    assert body["results"], "603 rated 30 of 50 movies, 20 candidates remain"
    recommended = {result["movieId"] for result in body["results"]}
    assert not recommended & rated_movie_ids(recommender_graph, 603)


def test_recommendations_scores_descend_and_carry_evidence(api_client):
    """Scores are sorted and every row explains itself (peers, genre matches)."""
    results = api_client.get("/api/users/603/recommendations").json()["results"]
    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)
    for result in results:
        assert result["peerCount"] >= 1
        assert 0 <= result["genreMatches"] <= 3


def test_recommendations_for_the_user_who_rated_everything(api_client):
    """User 414 rated all 50 movies: a known user, an empty list, not an error."""
    response = api_client.get("/api/users/414/recommendations")
    assert response.status_code == 200
    assert response.json() == {"userId": 414, "results": []}


def test_recommendations_unknown_user_is_404(api_client):
    """An id that is not in the graph names the user in a 404."""
    response = api_client.get("/api/users/999999/recommendations")
    assert response.status_code == 404
    assert "999999" in response.json()["detail"]


def test_discover_stays_inside_the_community_and_unseen(api_client, recommender_graph):
    """Every user's page holds only their community's movies, none already rated.

    After run_gds all 20 fixture users have a community, so this sweeps them all.
    """
    users = recommender_graph.execute_query(
        "MATCH (u:User) RETURN u.userId AS id, u.community AS community ORDER BY id"
    ).records
    assert len(users) == 20
    for user in users:
        body = api_client.get(f"/api/users/{user['id']}/discover").json()
        assert body["community"] == user["community"]
        for movie in body["results"]:
            assert movie["community"] == user["community"]
        recommended = {movie["movieId"] for movie in body["results"]}
        assert not recommended & rated_movie_ids(recommender_graph, user["id"])


def test_discover_orders_by_pagerank_and_paginates(api_client):
    """Results descend by pagerank and skip/limit pages never overlap."""
    full = api_client.get("/api/users/603/discover?limit=10").json()["results"]
    ranks = [movie["pagerank"] for movie in full]
    assert ranks == sorted(ranks, reverse=True)
    page1 = api_client.get("/api/users/603/discover?skip=0&limit=3").json()["results"]
    page2 = api_client.get("/api/users/603/discover?skip=3&limit=3").json()["results"]
    assert [m["movieId"] for m in page1] == [m["movieId"] for m in full[:3]]
    assert not {m["movieId"] for m in page1} & {m["movieId"] for m in page2}


def test_discover_unknown_user_is_404(api_client):
    """An id that is not in the graph names the user in a 404."""
    response = api_client.get("/api/users/999999/discover")
    assert response.status_code == 404
    assert "999999" in response.json()["detail"]
