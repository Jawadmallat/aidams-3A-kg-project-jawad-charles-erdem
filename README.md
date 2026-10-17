# Movie Recommender in Production

A graph-backed movie recommendation service built on Neo4j, Graph Data Science and FastAPI, using the MovieLens dataset. It includes an NL-to-Cypher endpoint and a "Why was this recommended?" endpoint whose explanations rest on actual graph paths.

AIDAMS 3A, Graph Databases project 4 (level 2).

## Team

| Member  | Role                                  |
|---------|---------------------------------------|
| Jawad   | A: data, graph modeling, GDS and ML   |
| Charles | TBD                                   |
| Erdem   | TBD                                   |

## Dataset

[MovieLens](https://grouplens.org/datasets/movielens/) by GroupLens Research, University of Minnesota.

| Dataset | Users | Movies | Ratings | Tag applications | Used for |
|---|---|---|---|---|---|
| `ml-latest-small` | 610 | 9,742 | 100,836 | 3,683 | development, the demo notebook, the ML evaluation |
| `ml-25m`, 5% of users (seed 42) | 8,127 | 62,423 | 1,265,960 | 39,166 | benchmarks (`--sample-users 0.05`) |
| Test fixture (`tests/fixtures/movielens`) | 20 | 50 | 860 | 282 | tests and CI, committed |

**Why MovieLens.** It is the reference dataset for recommender systems, so results can be compared with published baselines. It is a real user–movie bipartite graph with explicit ratings (0.5 to 5 stars) and timestamps, which supports both "liked" relationships and a split in time for the ML evaluation. It comes in two consistent sizes: the small set loads in seconds for development, and `ml-25m` makes performance measurements meaningful. `links.csv` adds IMDb and TMDb ids.

**Files used:** `movies.csv` (movieId, title with year, pipe-separated genres), `ratings.csv` (userId, movieId, rating, timestamp), `tags.csv` (userId, movieId, free-text tag, timestamp), `links.csv` (movieId, imdbId, tmdbId).

**Getting the data.** Datasets are downloaded into `data/` and never committed:

```bash
uv run python scripts/download_data.py                    # ml-latest-small
uv run python scripts/download_data.py --dataset ml-25m   # about 250 MB
```

**Licence.** MovieLens may be used for research and redistributed (including transformations, such as the test fixture) under the same licence; no commercial use without permission, and no endorsement by the University of Minnesota or GroupLens is implied. Citation: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>

The cleaning rules, the graph schema and the reasons behind them are in [docs/modeling.md](docs/modeling.md); the conventions the API relies on are in [docs/decisions.md](docs/decisions.md).

## Getting started

The project uses [uv](https://docs.astral.sh/uv/) for Python and dependency management (Python 3.12).

```bash
uv sync   # create .venv and install the locked dependencies
```

Installation, Docker, API examples, the graph schema, the architecture and the modeling rationale will be documented here as the project progresses.

## Appendix: our schema versus an AI-generated one

We asked ChatGPT (free tier, temporary chat) for a Neo4j schema from the project description alone, ran its key choices on our data, and compared. It agrees with our core model (ratings as relationships, Genre nodes, precomputed top-k similarity, the same uniqueness constraints). It differs where the data decides: its PageRank on the rating graph ranks movies almost exactly by number of ratings (rank correlation 0.968, against 0.145 for ours on `SIMILAR`); it keeps the year inside titles and ignores duplicate titles; and it models sparse tags as two extra node types. One of its ideas was adopted: a full-text index on tags. Full comparison, prompt and evidence: [docs/anti_genericity.md](docs/anti_genericity.md).

