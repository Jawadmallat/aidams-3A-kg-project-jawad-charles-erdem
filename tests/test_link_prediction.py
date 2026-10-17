"""Tests for scripts/link_prediction.py: unit tests on toy data, then the fixture."""

import numpy as np
import pandas as pd
import pytest

from scripts import link_prediction as lp
from scripts.movielens import LIKED_THRESHOLD


def toy_ratings():
    """Four users, five movies, timestamps 1..12 so the 80% cutoff is easy to reason about."""
    return pd.DataFrame({
        "userId":    [1, 1, 1, 2, 2, 2, 3, 3, 1, 2, 4, 4],
        "movieId":   [10, 20, 30, 10, 20, 40, 10, 30, 40, 30, 10, 50],
        "rating":    [5.0, 4.0, 2.0, 4.5, 4.0, 3.0, 4.0, 5.0, 4.5, 4.0, 5.0, 4.0],
        "timestamp": list(range(1, 13)),
    })


def toy_embeddings():
    """Two-dimensional embeddings: movie 10 aligned with user 1, movie 50 against."""
    return lp.Embeddings(
        users=np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        movies=np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.2, 0.8], [-1.0, 0.0]]),
        user_row={1: 0, 2: 1, 3: 2},
        movie_row={10: 0, 20: 1, 30: 2, 40: 3, 50: 4},
        movie_ids=np.array([10, 20, 30, 40, 50]),
    )


def test_temporal_split_puts_older_ratings_in_train():
    """Every training rating is older than every test rating."""
    train, test, cutoff = lp.temporal_split(toy_ratings())
    assert train.timestamp.max() < cutoff <= test.timestamp.min()
    assert len(train) + len(test) == len(toy_ratings())


def test_popularity_counts_training_likes_only():
    """Popularity is the number of likes per movie, zero for unliked movies."""
    train, _, _ = lp.temporal_split(toy_ratings())
    pop = lp.popularity(train, np.array([10, 20, 30, 50]))
    assert pop.tolist() == [3.0, 2.0, 1.0, 0.0]


def test_training_pairs_negatives_are_unseen_movies():
    """Negatives never include a movie the user rated in the training period."""
    train, _, _ = lp.temporal_split(toy_ratings())
    pairs = lp.training_pairs(train, toy_embeddings(), np.random.default_rng(0))
    seen = lp.seen_movies(train)
    negatives = pairs[pairs.label == 0]
    assert not negatives.empty
    assert all(movie not in seen[user] for user, movie in zip(negatives.userId, negatives.movieId))
    positives = pairs[pairs.label == 1]
    assert (train.merge(positives, on=["userId", "movieId"]).rating >= LIKED_THRESHOLD).all()


def test_precision_at_k_counts_hits_in_top_k():
    """Two of the three best-scored candidates are positives."""
    labels = np.array([1, 0, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.0])
    assert lp.precision_at_k(labels, scores, k=3) == pytest.approx(2 / 3)


def test_evaluate_rewards_a_perfect_scorer_and_skips_cold_start():
    """An oracle scorer gets AUC 1; user 4 (no training history) is not evaluated."""
    ratings = toy_ratings()
    train, test = ratings[ratings.timestamp <= 6], ratings[ratings.timestamp > 6]
    emb = toy_embeddings()
    liked = test[test.rating >= LIKED_THRESHOLD].groupby("userId").movieId.agg(set)

    def oracle(user, rows):
        return np.isin(emb.movie_ids[rows], list(liked.get(user, set()))).astype(float)

    per_user = lp.evaluate({"oracle": oracle}, train, test, emb)
    assert set(per_user.userId) == {1, 2}
    assert (per_user.auc == 1.0).all()


def test_summarise_and_paired_gain_report_intervals():
    """Summary has one row per model; the gain of a model over itself is exactly zero."""
    per_user = pd.DataFrame({
        "model": ["a", "a", "b", "b"], "userId": [1, 2, 1, 2],
        "auc": [0.6, 0.8, 0.7, 0.9], f"precision@{lp.TOP_K}": [0.1, 0.2, 0.1, 0.3],
    })
    summary = lp.summarise(per_user, n_boot=50)
    assert list(summary.index) == ["a", "b"] and summary.loc["a", "users"] == 2
    gain = lp.paired_gain(per_user, "b", "a", n_boot=50)
    assert gain.loc["auc", "mean gain"] == pytest.approx(0.1)
    assert lp.paired_gain(per_user, "a", "a", n_boot=50).loc["auc", "users improved"] == 0


def test_embeddings_only_cover_training_period_likes(seeded_graph, fixture_frames):
    """No user or movie gets an embedding from a like made after the cutoff."""
    ratings = fixture_frames["ratings"]
    train, _, cutoff = lp.temporal_split(ratings)
    emb = lp.compute_embeddings(seeded_graph, cutoff)
    train_likes = train[train.rating >= LIKED_THRESHOLD]
    assert set(emb.user_row) == set(train_likes.userId)
    assert set(emb.movie_row) == set(train_likes.movieId)
    assert emb.users.shape[1] == lp.EMBEDDING_DIM


def test_full_run_on_fixture_is_reproducible(seeded_graph):
    """The experiment runs end to end and gives identical results twice."""
    first = lp.run(seeded_graph)
    second = lp.run(seeded_graph)
    assert first["split"]["evaluated users"] >= 1
    metrics = first["per_user"][["auc", f"precision@{lp.TOP_K}"]]
    assert ((metrics >= 0) & (metrics <= 1)).all().all()
    pd.testing.assert_frame_equal(first["per_user"], second["per_user"])
