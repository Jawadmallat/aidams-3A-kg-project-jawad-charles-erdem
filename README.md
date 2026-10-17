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

[MovieLens](https://grouplens.org/datasets/movielens/) by GroupLens Research, University of Minnesota:

- `ml-latest-small` (610 users, 9,742 movies, 100,836 ratings) for development and tests.
- A seeded sample of `ml-25m` for benchmarks.

The datasets are downloaded by script into `data/` and are never committed.

## Getting started

The project uses [uv](https://docs.astral.sh/uv/) for Python and dependency management (Python 3.12).

```bash
uv sync   # create .venv and install the locked dependencies
```

Installation, Docker, API examples, the graph schema, the architecture and the modeling rationale will be documented here as the project progresses.
