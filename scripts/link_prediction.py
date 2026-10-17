"""Link prediction: will a user rate a movie 4.0 or higher?

Temporal protocol (no information from the test period reaches the features):
    * ratings before the cutoff (80th percentile of timestamps) are the training period;
    * FastRP embeddings are computed by GDS on the training-period "liked" graph only;
    * popularity and the movies a user has already seen also come from that period;
    * labels are the movies each user liked after the cutoff.
The GDS properties written by run_gds.py (SIMILAR, community, pagerank) are
computed on all ratings and are deliberately not used here.

Compared rankers: popularity (training-period likes), a logistic regression on
the Hadamard product of user and movie embeddings, and the same plus popularity.
Metrics, per evaluated user then averaged with a bootstrap 95% interval:
AUC and precision@10 over the movies the user had not rated before the cutoff.

Usage:
    uv run --env-file .env python -m scripts.link_prediction
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from neo4j import Driver
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.db import get_driver
from scripts.movielens import LIKED_THRESHOLD

TRAIN_GRAPH = "train-likes"
CUTOFF_QUANTILE = 0.8
EMBEDDING_DIM = 64
NEGATIVES_PER_POSITIVE = 4
TOP_K = 10
SEED = 42

RATINGS_QUERY = """
MATCH (u:User)-[r:RATED]->(m:Movie)
RETURN u.userId AS userId, m.movieId AS movieId, r.rating AS rating, r.timestamp AS timestamp
"""

PROJECT_TRAIN = """
MATCH (u:User)-[r:RATED]->(m:Movie)
WHERE r.timestamp < $cutoff AND r.rating >= $liked
WITH u, m ORDER BY u.userId, m.movieId
WITH gds.graph.project($graph, u, m,
  {sourceNodeLabels: 'User', targetNodeLabels: 'Movie'},
  {undirectedRelationshipTypes: ['*']}) AS g
RETURN g.nodeCount AS nodes, g.relationshipCount AS relationships
"""

FASTRP = """
CALL gds.fastRP.stream($graph, {embeddingDimension: $dim, randomSeed: $seed, concurrency: 1})
YIELD nodeId, embedding
WITH gds.util.asNode(nodeId) AS node, embedding
RETURN node.userId AS userId, node.movieId AS movieId, embedding
"""


@dataclass
class Embeddings:
    """Row-aligned embedding matrices with id -> row lookups."""

    users: np.ndarray
    movies: np.ndarray
    user_row: dict[int, int]
    movie_row: dict[int, int]
    movie_ids: np.ndarray


Scorer = Callable[[int, np.ndarray], np.ndarray]


def load_ratings(driver: Driver) -> pd.DataFrame:
    """Read every RATED relationship from the graph."""
    records = driver.execute_query(RATINGS_QUERY).records
    return pd.DataFrame([record.data() for record in records])


def temporal_split(ratings: pd.DataFrame, quantile: float = CUTOFF_QUANTILE):
    """Split ratings at a timestamp quantile; return (train, test, cutoff)."""
    cutoff = int(ratings.timestamp.quantile(quantile))
    return ratings[ratings.timestamp < cutoff], ratings[ratings.timestamp >= cutoff], cutoff


def compute_embeddings(driver: Driver, cutoff: int, dim: int = EMBEDDING_DIM) -> Embeddings:
    """Project the training-period liked graph and embed it with FastRP."""
    driver.execute_query("CALL gds.graph.drop($g, false) YIELD graphName", g=TRAIN_GRAPH)
    try:
        driver.execute_query(PROJECT_TRAIN, graph=TRAIN_GRAPH, cutoff=cutoff, liked=LIKED_THRESHOLD)
        records = driver.execute_query(FASTRP, graph=TRAIN_GRAPH, dim=dim, seed=SEED).records
    finally:
        driver.execute_query("CALL gds.graph.drop($g, false) YIELD graphName", g=TRAIN_GRAPH)

    users = [(r["userId"], r["embedding"]) for r in records if r["userId"] is not None]
    movies = sorted((r["movieId"], r["embedding"]) for r in records if r["movieId"] is not None)
    return Embeddings(
        users=np.array([vector for _, vector in users]),
        movies=np.array([vector for _, vector in movies]),
        user_row={user: row for row, (user, _) in enumerate(users)},
        movie_row={movie: row for row, (movie, _) in enumerate(movies)},
        movie_ids=np.array([movie for movie, _ in movies]),
    )


def popularity(train: pd.DataFrame, movie_ids: np.ndarray) -> np.ndarray:
    """Training-period like count of each movie, aligned with movie_ids."""
    likes = train[train.rating >= LIKED_THRESHOLD].groupby("movieId").size()
    return likes.reindex(movie_ids, fill_value=0).to_numpy(dtype=float)


def seen_movies(train: pd.DataFrame) -> dict[int, set[int]]:
    """Movies each user rated during the training period."""
    return train.groupby("userId").movieId.agg(set).to_dict()


def training_pairs(train: pd.DataFrame, emb: Embeddings, rng: np.random.Generator) -> pd.DataFrame:
    """Training-period likes as positives, unrated movies sampled as negatives."""
    likes = train[(train.rating >= LIKED_THRESHOLD)
                  & train.userId.isin(emb.user_row) & train.movieId.isin(emb.movie_row)]
    seen = seen_movies(train)
    frames = [likes[["userId", "movieId"]].assign(label=1)]
    for user, n_likes in likes.groupby("userId").size().items():
        pool = np.setdiff1d(emb.movie_ids, list(seen[user]), assume_unique=True)
        size = min(len(pool), n_likes * NEGATIVES_PER_POSITIVE)
        negatives = rng.choice(pool, size=size, replace=False)
        frames.append(pd.DataFrame({"userId": user, "movieId": negatives, "label": 0}))
    return pd.concat(frames, ignore_index=True)


def pair_features(emb: Embeddings, users: np.ndarray, movie_rows: np.ndarray,
                  pop: np.ndarray | None) -> np.ndarray:
    """Hadamard product of user and movie embeddings, optionally with log popularity."""
    user_vectors = emb.users[[emb.user_row[u] for u in users]]
    features = user_vectors * emb.movies[movie_rows]
    if pop is not None:
        features = np.column_stack([features, np.log1p(pop[movie_rows])])
    return features


def fit_model(pairs: pd.DataFrame, emb: Embeddings, pop: np.ndarray | None):
    """Fit a scaled logistic regression on the training pairs."""
    rows = np.array([emb.movie_row[m] for m in pairs.movieId])
    features = pair_features(emb, pairs.userId.to_numpy(), rows, pop)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return model.fit(features, pairs.label.to_numpy())


def model_scorer(model, emb: Embeddings, pop: np.ndarray | None) -> Scorer:
    """Wrap a fitted model as scorer(user, candidate movie rows) -> scores."""
    def score(user: int, rows: np.ndarray) -> np.ndarray:
        users = np.full(len(rows), user)
        return model.predict_proba(pair_features(emb, users, rows, pop))[:, 1]
    return score


def popularity_scorer(pop: np.ndarray) -> Scorer:
    """Rank every user's candidates by training-period popularity."""
    return lambda user, rows: pop[rows]


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int = TOP_K) -> float:
    """Share of the k highest-scored candidates that are positives (stable on ties)."""
    top = np.argsort(-scores, kind="stable")[:k]
    return float(labels[top].mean())


def evaluate(scorers: dict[str, Scorer], train: pd.DataFrame, test: pd.DataFrame,
             emb: Embeddings) -> pd.DataFrame:
    """Per-user AUC and precision@k for each scorer, on users seen in both periods."""
    test_likes = test[test.rating >= LIKED_THRESHOLD].groupby("userId").movieId.agg(set)
    seen = seen_movies(train)
    results = []
    for user, liked in test_likes.items():
        if user not in emb.user_row or user not in seen:
            continue  # cold-start user: no training-period history to embed
        candidates = np.setdiff1d(emb.movie_ids, list(seen[user]), assume_unique=True)
        labels = np.isin(candidates, list(liked)).astype(int)
        if labels.sum() in (0, len(labels)):
            continue  # every test like is on a movie unknown before the cutoff
        rows = np.array([emb.movie_row[m] for m in candidates])
        for name, scorer in scorers.items():
            scores = scorer(user, rows)
            results.append({"model": name, "userId": user,
                            "auc": roc_auc_score(labels, scores),
                            f"precision@{TOP_K}": precision_at_k(labels, scores)})
    return pd.DataFrame(results)


def summarise(per_user: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Mean of each metric per model with a bootstrap (over users) 95% interval."""
    rng = np.random.default_rng(SEED)
    rows = []
    for name, group in per_user.groupby("model", sort=False):
        row = {"model": name, "users": len(group)}
        for metric in ("auc", f"precision@{TOP_K}"):
            values = group[metric].to_numpy()
            means = rng.choice(values, size=(n_boot, len(values))).mean(axis=1)
            low, high = np.percentile(means, [2.5, 97.5])
            row[metric] = f"{values.mean():.3f} [{low:.3f}, {high:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def paired_gain(per_user: pd.DataFrame, model: str, baseline: str,
                n_boot: int = 1000) -> pd.DataFrame:
    """Per-user difference model - baseline, with a bootstrap 95% interval.

    Pairing by user removes the between-user variance that makes the separate
    intervals overlap: an interval above zero means the model wins on these users.
    """
    rng = np.random.default_rng(SEED)
    wide = per_user.pivot(index="userId", columns="model")
    rows = {}
    for metric in ("auc", f"precision@{TOP_K}"):
        gains = (wide[(metric, model)] - wide[(metric, baseline)]).to_numpy()
        means = rng.choice(gains, size=(n_boot, len(gains))).mean(axis=1)
        low, high = np.percentile(means, [2.5, 97.5])
        rows[metric] = {
            "mean gain": round(gains.mean(), 3),
            "95% interval": f"[{low:.3f}, {high:.3f}]",
            "users improved": int((gains > 0).sum()),
            "users worse": int((gains < 0).sum()),
        }
    return pd.DataFrame(rows).T


def run(driver: Driver) -> dict:
    """Full experiment; returns the split facts, per-user results and summary."""
    ratings = load_ratings(driver)
    train, test, cutoff = temporal_split(ratings)
    emb = compute_embeddings(driver, cutoff)
    pop = popularity(train, emb.movie_ids)
    pairs = training_pairs(train, emb, np.random.default_rng(SEED))

    scorers = {
        "popularity (baseline)": popularity_scorer(pop),
        "FastRP": model_scorer(fit_model(pairs, emb, None), emb, None),
        "FastRP + popularity": model_scorer(fit_model(pairs, emb, pop), emb, pop),
    }
    per_user = evaluate(scorers, train, test, emb)
    test_users = set(test[test.rating >= LIKED_THRESHOLD].userId)
    return {
        "split": {
            "cutoff": pd.to_datetime(cutoff, unit="s").strftime("%Y-%m-%d"),
            "train ratings": len(train), "test ratings": len(test),
            "users with a test like": len(test_users),
            "cold-start users (no training history)": len(test_users - set(train.userId)),
            "evaluated users": per_user.userId.nunique(),
            "training pairs": len(pairs),
        },
        "per_user": per_user,
        "summary": summarise(per_user),
        "gain": paired_gain(per_user, "FastRP + popularity", "popularity (baseline)"),
    }


def main() -> int:
    """Entry point."""
    with get_driver() as driver:
        results = run(driver)
    for name, value in results["split"].items():
        print(f"{name:<40} {value}")
    print()
    print(results["summary"].to_string())
    print("\nFastRP + popularity minus popularity, per user:")
    print(results["gain"].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
