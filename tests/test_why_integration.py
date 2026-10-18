"""The why endpoint on the fixture graph: the paths it returns are real.

Fixture facts (docs/decisions.md): user 603 has the fewest ratings and a
top recommendation with peer evidence; user 414 rated every movie. The LLM
is switched off (no key), so these tests check the verified part alone.
"""

import pytest

from app.services.paths import recommendation_evidence
from app.services.users import recommendations


@pytest.fixture(name="no_llm")
def fixture_no_llm(monkeypatch):
    """Run without an LLM: the paths must stand on their own."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)


@pytest.mark.usefixtures("no_llm")
def test_why_returns_the_paths_behind_a_real_recommendation(api_client, recommender_graph):
    """For 603's top pick, the endpoint's paths are the graph's evidence."""
    top = recommendations(recommender_graph, 603, 1)["results"][0]["movieId"]
    response = api_client.get(f"/api/users/603/recommendations/{top}/why")
    assert response.status_code == 200
    body = response.json()
    assert body["hasEvidence"] is True
    assert body["alreadyRated"] is None
    assert body["paths"] == recommendation_evidence(recommender_graph, 603, top)
    assert body["summary"] is None
    assert "No LLM configured" in body["note"]


@pytest.mark.usefixtures("no_llm")
def test_every_peer_in_the_paths_liked_the_movie(api_client, recommender_graph):
    """Spot-check the evidence: each peer's like of the movie exists in the graph."""
    top = recommendations(recommender_graph, 603, 1)["results"][0]["movieId"]
    body = api_client.get(f"/api/users/603/recommendations/{top}/why").json()
    for peer in body["paths"]["peers"]:
        rating = recommender_graph.execute_query(
            "MATCH (:User {userId: $peer})-[r:RATED]->(:Movie {movieId: $movie}) "
            "RETURN r.rating AS rating", peer=peer["peerId"], movie=top,
        ).records[0]["rating"]
        assert rating >= 4.0


@pytest.mark.usefixtures("no_llm")
def test_a_movie_rated_by_the_user_is_flagged(api_client):
    """User 414 rated every movie, so Toy Story comes back as already rated."""
    body = api_client.get("/api/users/414/recommendations/1/why").json()
    assert body["alreadyRated"] is not None
    assert body["title"] == "Toy Story"


@pytest.mark.usefixtures("no_llm")
def test_unknown_ids_are_404_on_the_real_graph(api_client):
    """Ids absent from the fixture answer 404."""
    assert api_client.get("/api/users/999999/recommendations/1/why").status_code == 404
    assert api_client.get("/api/users/603/recommendations/999999/why").status_code == 404
