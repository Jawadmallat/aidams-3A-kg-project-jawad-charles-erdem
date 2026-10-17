"""Unit tests for the row builders of scripts/seed_data.py (no database)."""

import pandas as pd
import pytest

from scripts import seed_data


@pytest.fixture(name="frames")
def fixture_frames():
    """A tiny MovieLens-shaped dataset with the awkward cases in it."""
    return {
        "movies": pd.DataFrame({
            "movieId": [1, 2, 3],
            "title": ["Toy Story (1995)", "Matrix, The (1999)", "Babylon 5"],
            "genres": ["Animation|Comedy", "Action|Sci-Fi", "(no genres listed)"],
        }),
        "links": pd.DataFrame({
            "movieId": [1, 2, 3],
            "imdbId": [114709, 133093, 105946],
            "tmdbId": [862.0, 603.0, float("nan")],
        }),
        "ratings": pd.DataFrame({
            "userId": [10, 10, 11, 12, 12],
            "movieId": [1, 1, 2, 1, 3],
            "rating": [3.0, 4.5, 5.0, 2.0, 4.0],
            "timestamp": [100, 200, 300, 400, 500],
        }),
        "tags": pd.DataFrame({
            "userId": [10, 10, 13],
            "movieId": [1, 1, 2],
            "tag": ["Pixar", " pixar ", "   "],
            "timestamp": [150, 160, 170],
        }),
    }


def test_genre_rows_drop_placeholder(frames):
    """Genres are distinct, sorted and never include the placeholder."""
    genres = seed_data.build_genre_rows(frames["movies"])
    assert genres == ["Action", "Animation", "Comedy", "Sci-Fi"]


def test_movie_rows_are_cleaned_and_joined_with_links(frames):
    """Titles, years, genres and external ids come out in graph form."""
    built = seed_data.build_movie_rows(frames["movies"], frames["links"])
    rows = {row["movieId"]: row for row in built}
    assert rows[2] == {
        "movieId": 2, "title": "The Matrix", "year": 1999, "genres": ["Action", "Sci-Fi"],
        "imdbId": "tt0133093", "tmdbId": 603,
    }
    assert rows[3]["year"] is None and rows[3]["genres"] == [] and rows[3]["tmdbId"] is None


def test_user_rows_include_taggers_without_ratings(frames):
    """A user who only tagged still becomes a User node."""
    assert seed_data.build_user_rows(frames["ratings"], frames["tags"]) == [10, 11, 12, 13]


def test_rating_rows_keep_latest_rating_per_pair(frames):
    """A re-rated movie keeps the most recent rating, with native Python types."""
    rows = seed_data.build_rating_rows(frames["ratings"])
    assert len(rows) == 4
    assert {"userId": 10, "movieId": 1, "rating": 4.5, "timestamp": 200} in rows
    assert all(isinstance(row["rating"], float) and isinstance(row["userId"], int) for row in rows)


def test_tag_rows_are_normalised_and_deduplicated(frames):
    """Case variants collapse to one tag (earliest kept); blank tags are dropped."""
    assert seed_data.build_tag_rows(frames["tags"]) == [
        {"userId": 10, "movieId": 1, "tag": "pixar", "timestamp": 150},
    ]


def test_sample_users_keeps_whole_histories(frames):
    """Sampling selects users, keeping all of their ratings and the full catalogue."""
    sampled = seed_data.sample_users(frames, 0.5, random_state=1)
    kept = set(sampled["ratings"].userId)
    assert len(kept) == 2
    for user in kept:
        assert len(sampled["ratings"][sampled["ratings"].userId == user]) == len(
            frames["ratings"][frames["ratings"].userId == user]
        )
    assert sampled["movies"] is frames["movies"]
    assert seed_data.sample_users(frames, 0.5, random_state=1)["ratings"].equals(sampled["ratings"])


@pytest.mark.parametrize("fraction", [0, -0.1, 1.5])
def test_sample_users_rejects_bad_fraction(frames, fraction):
    """Fractions outside (0, 1] are refused."""
    with pytest.raises(ValueError):
        seed_data.sample_users(frames, fraction, random_state=1)


def test_batches_cover_every_row_once():
    """Batching splits rows into consecutive slices without loss."""
    assert list(seed_data.batches(list(range(7)), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
