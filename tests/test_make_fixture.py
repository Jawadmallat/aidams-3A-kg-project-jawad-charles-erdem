"""Unit tests for scripts/make_fixture.py (no database)."""

import pandas as pd

from scripts.make_fixture import build_fixture, top_ids, write_fixture


def make_frames():
    """Three users and three movies; user 3 and movie 30 are the least active."""
    ratings = pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 3],
        "movieId": [10, 20, 30, 10, 20, 10],
        "rating": [4.0, 3.0, 5.0, 2.0, 4.5, 3.5],
        "timestamp": [1, 2, 3, 4, 5, 6],
    })
    return {
        "movies": pd.DataFrame({"movieId": [10, 20, 30], "title": ["A", "B", "C"],
                                "genres": ["Drama"] * 3}),
        "links": pd.DataFrame({"movieId": [10, 20, 30], "imdbId": [1, 2, 3],
                               "tmdbId": [1.0, 2.0, 3.0]}),
        "ratings": ratings,
        "tags": pd.DataFrame({"userId": [1, 3], "movieId": [10, 30],
                              "tag": ["x", "y"], "timestamp": [7, 8]}),
    }


def test_top_ids_breaks_ties_by_smaller_id():
    """Equal activity is ordered by id so the fixture is deterministic."""
    frame = pd.DataFrame({"userId": [5, 5, 2, 2, 9]})
    assert top_ids(frame, "userId", 2) == [2, 5]


def test_build_fixture_keeps_only_pairs_inside_the_selection():
    """Ratings and tags survive only when both their user and movie were selected."""
    fixture = build_fixture(make_frames(), n_users=2, n_movies=2)
    assert set(fixture["movies"].movieId) == {10, 20}
    assert set(fixture["links"].movieId) == {10, 20}
    assert set(zip(fixture["ratings"].userId, fixture["ratings"].movieId)) == {
        (1, 10), (1, 20), (2, 10), (2, 20),
    }
    assert fixture["tags"].to_dict("records") == [
        {"userId": 1, "movieId": 10, "tag": "x", "timestamp": 7},
    ]


def test_write_fixture_produces_movielens_files(tmp_path):
    """The output has the four MovieLens CSVs plus a licence README."""
    write_fixture(build_fixture(make_frames(), 2, 2), tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {
        "movies.csv", "links.csv", "ratings.csv", "tags.csv", "README.md",
    }
    assert "licence" in (tmp_path / "README.md").read_text().lower()
