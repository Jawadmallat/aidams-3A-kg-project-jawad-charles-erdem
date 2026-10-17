"""Build the small MovieLens test fixture committed under tests/fixtures/.

Takes the most active users and the most rated movies of ml-latest-small and keeps
only the ratings and tags between them. Choosing the densest corner of the data
means similarity, collaborative filtering and communities return real results
even on a graph small enough for unit tests and CI.

The output uses the MovieLens CSV format, so scripts/seed_data.py loads it as is:
    uv run python -m scripts.make_fixture
    uv run --env-file .env python -m scripts.seed_data --data-dir tests/fixtures/movielens
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from scripts.seed_data import read_dataset

DEFAULT_OUT = Path("tests/fixtures/movielens")

LICENCE_NOTE = """# MovieLens test fixture

A subset of [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/)
built by `scripts/make_fixture.py`: the {users} most active users, the {movies}
most rated movies, and the ratings and tags between them.

Regenerate it with:

    uv run python scripts/download_data.py
    uv run python -m scripts.make_fixture

## Licence

This subset is redistributed under the MovieLens usage licence: research use
only, no endorsement implied by the University of Minnesota or GroupLens, and
no commercial use without permission. Citation:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
> and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4:
> 19:1–19:19. https://doi.org/10.1145/2827872
"""


def top_ids(frame: pd.DataFrame, column: str, count: int) -> list[int]:
    """Return the `count` ids with the most rows, ties broken by smaller id."""
    sizes = frame.groupby(column).size().reset_index(name="n")
    ranked = sizes.sort_values(["n", column], ascending=[False, True])
    return ranked[column].head(count).tolist()


def build_fixture(
    frames: dict[str, pd.DataFrame], n_users: int, n_movies: int
) -> dict[str, pd.DataFrame]:
    """Restrict a dataset to its most active users and most rated movies."""
    ratings = frames["ratings"]
    users = set(top_ids(ratings, "userId", n_users))
    movies = set(top_ids(ratings, "movieId", n_movies))

    def keep(frame: pd.DataFrame) -> pd.DataFrame:
        mask = frame.userId.isin(users) & frame.movieId.isin(movies)
        return frame[mask].sort_values(["userId", "movieId", "timestamp"])

    return {
        "movies": frames["movies"][frames["movies"].movieId.isin(movies)].sort_values("movieId"),
        "links": frames["links"][frames["links"].movieId.isin(movies)].sort_values("movieId"),
        "ratings": keep(ratings),
        "tags": keep(frames["tags"]),
    }


def write_fixture(fixture: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Write the fixture CSVs and a README carrying the licence."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in fixture.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "README.md").write_text(LICENCE_NOTE.format(
        users=fixture["ratings"].userId.nunique(), movies=len(fixture["movies"]),
    ))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    default_source = Path(os.environ.get("DATA_DIR", "data")) / "ml-latest-small"
    parser = argparse.ArgumentParser(description="Build the MovieLens test fixture.")
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--movies", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    fixture = build_fixture(read_dataset(args.source), args.users, args.movies)
    write_fixture(fixture, args.out)
    sizes = ", ".join(f"{name}={len(frame)}" for name, frame in fixture.items())
    print(f"Fixture written to {args.out}: {sizes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
