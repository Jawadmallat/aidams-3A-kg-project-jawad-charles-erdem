"""Load a MovieLens dataset into Neo4j.

Graph written (schema v1):
    (:User {userId})-[:RATED {rating, timestamp}]->(:Movie)
    (:User)-[:TAGGED {tag, timestamp}]->(:Movie)
    (:Movie {movieId, title, year, genres, imdbId, tmdbId})-[:IN_GENRE]->(:Genre {name})

Every write is a batched UNWIND + MERGE keyed on a uniqueness constraint, so
running the script twice leaves the graph unchanged (no duplicates).

Usage:
    uv run --env-file .env python -m scripts.seed_data
    uv run --env-file .env python -m scripts.seed_data --data-dir tests/fixtures/movielens
    uv run --env-file .env python -m scripts.seed_data --data-dir data/ml-25m \\
        --sample-users 0.05 --seed 42 --reset
"""

import argparse
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from neo4j import Driver

from scripts.db import get_driver
from scripts.movielens import imdb_id, normalize_tag, optional_int, parse_genres, parse_title
from scripts.schema import apply_schema

GENRES_QUERY = """
UNWIND $rows AS name
MERGE (:Genre {name: name})
"""

MOVIES_QUERY = """
UNWIND $rows AS row
MERGE (m:Movie {movieId: row.movieId})
SET m.title = row.title, m.year = row.year, m.genres = row.genres,
    m.imdbId = row.imdbId, m.tmdbId = row.tmdbId
WITH m, row
UNWIND row.genres AS genreName
MATCH (g:Genre {name: genreName})
MERGE (m)-[:IN_GENRE]->(g)
"""

USERS_QUERY = """
UNWIND $rows AS userId
MERGE (:User {userId: userId})
"""

RATINGS_QUERY = """
UNWIND $rows AS row
MATCH (u:User {userId: row.userId})
MATCH (m:Movie {movieId: row.movieId})
MERGE (u)-[r:RATED]->(m)
SET r.rating = row.rating, r.timestamp = row.timestamp
"""

TAGS_QUERY = """
UNWIND $rows AS row
MATCH (u:User {userId: row.userId})
MATCH (m:Movie {movieId: row.movieId})
MERGE (u)-[t:TAGGED {tag: row.tag}]->(m)
SET t.timestamp = row.timestamp
"""

COUNTS_QUERY = """
CALL () { MATCH (n:User) RETURN count(n) AS users }
CALL () { MATCH (n:Movie) RETURN count(n) AS movies }
CALL () { MATCH (n:Genre) RETURN count(n) AS genres }
CALL () { MATCH ()-[r:RATED]->() RETURN count(r) AS ratings }
CALL () { MATCH ()-[r:TAGGED]->() RETURN count(r) AS tags }
CALL () { MATCH ()-[r:IN_GENRE]->() RETURN count(r) AS inGenre }
RETURN users, movies, genres, ratings, tags, inGenre
"""


def read_dataset(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the four MovieLens CSV files of a dataset directory."""
    return {
        name: pd.read_csv(data_dir / f"{name}.csv")
        for name in ("movies", "ratings", "tags", "links")
    }


def sample_users(
    frames: dict[str, pd.DataFrame], fraction: float, random_state: int
) -> dict[str, pd.DataFrame]:
    """Keep a random fraction of users with all of their ratings and tags.

    Sampling users rather than individual ratings keeps each sampled user's
    full history, so neighbourhoods used by collaborative filtering stay intact.
    The movie catalogue is kept whole.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    users = pd.Series(frames["ratings"].userId.unique())
    kept = set(users.sample(frac=fraction, random_state=random_state))
    return {
        **frames,
        "ratings": frames["ratings"][frames["ratings"].userId.isin(kept)],
        "tags": frames["tags"][frames["tags"].userId.isin(kept)],
    }


def build_genre_rows(movies: pd.DataFrame) -> list[str]:
    """Return the sorted distinct genre names, without the placeholder."""
    return sorted({genre for raw in movies.genres for genre in parse_genres(raw)})


def build_movie_rows(movies: pd.DataFrame, links: pd.DataFrame) -> list[dict]:
    """Turn movies.csv joined with links.csv into Movie node properties."""
    merged = movies.merge(links, on="movieId", how="left")
    rows = []
    for record in merged.itertuples(index=False):
        title, year = parse_title(record.title)
        rows.append({
            "movieId": int(record.movieId),
            "title": title,
            "year": year,
            "genres": parse_genres(record.genres),
            "imdbId": imdb_id(record.imdbId),
            "tmdbId": optional_int(record.tmdbId),
        })
    return rows


def build_user_rows(ratings: pd.DataFrame, tags: pd.DataFrame) -> list[int]:
    """Return every user id that rated or tagged something."""
    return sorted({int(user) for user in pd.concat([ratings.userId, tags.userId]).unique()})


def build_rating_rows(ratings: pd.DataFrame) -> list[dict]:
    """Return one row per (user, movie) rating, last rating wins if repeated."""
    deduped = ratings.sort_values("timestamp").drop_duplicates(["userId", "movieId"], keep="last")
    return [
        {"userId": int(r.userId), "movieId": int(r.movieId),
         "rating": float(r.rating), "timestamp": int(r.timestamp)}
        for r in deduped.itertuples(index=False)
    ]


def build_tag_rows(tags: pd.DataFrame) -> list[dict]:
    """Normalise tags and keep the first application of each (user, movie, tag)."""
    frame = tags.assign(tag=tags.tag.map(normalize_tag)).dropna(subset=["tag"])
    deduped = frame.sort_values("timestamp").drop_duplicates(["userId", "movieId", "tag"])
    return [
        {"userId": int(t.userId), "movieId": int(t.movieId),
         "tag": t.tag, "timestamp": int(t.timestamp)}
        for t in deduped.itertuples(index=False)
    ]


def batches(rows: list, size: int) -> Iterator[list]:
    """Yield consecutive slices of at most `size` rows."""
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def write_batches(driver: Driver, label: str, query: str, rows: list, size: int) -> None:
    """Run a parameterised UNWIND query once per batch, each in its own transaction."""
    started = time.perf_counter()
    for batch in batches(rows, size):
        driver.execute_query(query, rows=batch)
    print(f"  {label:<8} {len(rows):>9,} rows in {time.perf_counter() - started:6.1f}s")


def reset_graph(driver: Driver) -> None:
    """Delete every node and relationship, in batches, keeping the schema.

    Relationships go first: a batch of nodes deleted with DETACH can carry
    hundreds of thousands of ratings (popular movies, very active users) and
    exceed the transaction memory limit, while a batch of relationships is
    bounded by its size.
    """
    with driver.session() as session:
        session.run(
            "MATCH ()-[r]->() CALL (r) { DELETE r } IN TRANSACTIONS OF 10000 ROWS"
        ).consume()
        session.run(
            "MATCH (n) CALL (n) { DELETE n } IN TRANSACTIONS OF 10000 ROWS"
        ).consume()


def graph_counts(driver: Driver) -> dict[str, int]:
    """Count nodes and relationships per label and type."""
    return driver.execute_query(COUNTS_QUERY).records[0].data()


def seed(
    driver: Driver, frames: dict[str, pd.DataFrame], batch_size: int = 5000
) -> dict[str, int]:
    """Apply the schema, then load genres, movies, users, ratings and tags."""
    apply_schema(driver)
    movies, ratings, tags = frames["movies"], frames["ratings"], frames["tags"]
    write_batches(driver, "genres", GENRES_QUERY, build_genre_rows(movies), batch_size)
    movie_rows = build_movie_rows(movies, frames["links"])
    write_batches(driver, "movies", MOVIES_QUERY, movie_rows, batch_size)
    write_batches(driver, "users", USERS_QUERY, build_user_rows(ratings, tags), batch_size)
    write_batches(driver, "ratings", RATINGS_QUERY, build_rating_rows(ratings), batch_size)
    write_batches(driver, "tags", TAGS_QUERY, build_tag_rows(tags), batch_size)
    return graph_counts(driver)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    default_dir = Path(os.environ.get("DATA_DIR", "data")) / "ml-latest-small"
    parser = argparse.ArgumentParser(description="Load a MovieLens dataset into Neo4j.")
    parser.add_argument("--data-dir", type=Path, default=default_dir,
                        help="directory holding movies/ratings/tags/links.csv")
    parser.add_argument("--sample-users", type=float, default=None, metavar="FRACTION",
                        help="keep only this fraction of users (e.g. 0.05 for ml-25m)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for --sample-users")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--reset", action="store_true",
                        help="delete all nodes and relationships before loading")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    frames = read_dataset(args.data_dir)
    if args.sample_users:
        frames = sample_users(frames, args.sample_users, args.seed)
    print(f"Loading {args.data_dir} into Neo4j")
    with get_driver() as driver:
        if args.reset:
            print("  resetting graph")
            reset_graph(driver)
        counts = seed(driver, frames, args.batch_size)
    print("Graph now holds: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
