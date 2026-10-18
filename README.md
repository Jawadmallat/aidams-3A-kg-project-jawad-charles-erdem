# Movie Recommender in Production

[![CI](https://github.com/Jawadmallat/aidams-3A-kg-project-jawad-charles-erdem/actions/workflows/ci.yml/badge.svg)](https://github.com/Jawadmallat/aidams-3A-kg-project-jawad-charles-erdem/actions/workflows/ci.yml)

A graph-backed movie recommendation service built on Neo4j, Graph Data Science and FastAPI, using the MovieLens dataset. It includes an NL-to-Cypher endpoint and a "Why was this recommended?" endpoint whose explanations rest on actual graph paths.

AIDAMS 3A, Graph Databases project 4 (level 2).

## Team

| Member  | Role                                        |
|---------|---------------------------------------------|
| Jawad   | A: data, graph modeling, GDS and ML         |
| Erdem   | B: API, Cypher, performance, infrastructure |
| Charles | C: agentic component, documentation, quality |

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

## Quickstart (Docker Compose)

The whole stack - Neo4j 5.26 with the GDS and APOC plugins, the API, and Nginx as the only published entry point - runs through Compose:

```bash
cp .env.example .env            # set NEO4J_PASSWORD
docker compose up --build -d    # first boot downloads the Neo4j plugins (~2 min)

# load MovieLens inside the stack (safe to re-run)
docker compose exec api uv run python scripts/download_data.py
docker compose exec api uv run python -m scripts.seed_data
docker compose exec api uv run python -m scripts.run_gds

curl http://localhost:8080/api/movies/1/similar
```

Swagger UI: <http://localhost:8080/docs>. Only Nginx publishes a port (8080): Neo4j and the API are reachable solely on the internal network, and the proxy routes an allowlist (`/api/`, `/docs`, `/openapi.json`) - everything else 404s at the edge. For development against the containerized Neo4j, the overlay `docker-compose.dev.yml` publishes 7474/7687.

## API

| Endpoint | What it answers | Technique |
|---|---|---|
| `GET /api/movies/{id}/similar` | movies most like this one | precomputed GDS `SIMILAR` (Jaccard top-10); live co-rating Jaccard fallback for movies with fewer than 10 likes - the `source` field names the branch taken |
| `GET /api/users/{id}/recommendations` | personal picks | collaborative filtering: the 25 most like-minded users' likes, minus movies already seen, boosted 1 + 0.25 per shared top genre; `peerCount` and `genreMatches` carry the evidence |
| `GET /api/users/{id}/discover` | something outside the usual | unseen movies of the user's Louvain community, most central (PageRank) first |
| `GET /api/movies/trending` | popular and central | rating volume in a window ending at the newest rating in the data (the dataset ends in 2018; the `anchor` field makes that explicit), blended as `volume x (1 + pagerank)` |

```bash
curl 'http://localhost:8080/api/users/1/recommendations?limit=5'
curl 'http://localhost:8080/api/users/1/discover?skip=0&limit=10'
curl 'http://localhost:8080/api/movies/trending?days=365&limit=10'
```

Conventions: ids are integers; unknown user or movie answers 404 naming the id; a known user with nothing left to recommend answers 200 with an empty list; invalid parameters answer 422; a dead database answers 503. `skip`/`limit` pagination is capped (max 50) and deterministic (stable tiebreak on `movieId`), so pages never overlap.

The Cypher behind the endpoints uses scoped `CALL {}` subqueries, `EXISTS {}` / `COUNT {}` subqueries, list-comprehension predicates and multi-stage `WITH` aggregations deliberately - `app/services/*.py` keeps each query as one documented constant, and `app/services/paths.py` exposes the graph-path evidence the "why was this recommended" endpoint builds on.

## Performance

Measured on the ml-25m 5% sample (1,265,960 ratings): precomputed similar answers in ~3 ms p50 where the live fallback needs ~116 ms - the argument for precomputing `SIMILAR`; trending dropped from 180 ms to 30 ms p50 once `PROFILE` showed its anchor query scanning every rating (now 2 db hits through the `rated_timestamp` index). Full tables, the PROFILE plans before and after, and how to reproduce: [docs/performance.md](docs/performance.md).

## Public image

The API image is published to Docker Hub, [erdemsanver1/movie-recommender-api](https://hub.docker.com/r/erdemsanver1/movie-recommender-api), so the stack can run without building:

```bash
docker pull erdemsanver1/movie-recommender-api
```

Publishing is a manual, deliberate step (nothing in CI pushes images):

```bash
docker login
make docker-build docker-push DOCKER_USER=erdemsanver1
```

## Local development

The project uses [uv](https://docs.astral.sh/uv/) for Python and dependency management (Python 3.12).

```bash
uv sync      # create .venv and install the locked dependencies
make help    # every target: run, seed, gds, test, lint, benchmark, docker-*
make test    # pytest with coverage (integration tests need NEO4J_TEST_URI)
make lint    # pylint over app, scripts and tests; fails under 9.5
```

Credentials come from `.env` (never committed; copy `.env.example`). The two-container setup for local Neo4j, the graph contract and every data convention the queries follow are in [docs/decisions.md](docs/decisions.md); the modeling rationale is in [docs/modeling.md](docs/modeling.md).

## Appendix: our schema versus an AI-generated one

We asked ChatGPT (free tier, temporary chat) for a Neo4j schema from the project description alone, ran its key choices on our data, and compared. It agrees with our core model (ratings as relationships, Genre nodes, precomputed top-k similarity, the same uniqueness constraints). It differs where the data decides: its PageRank on the rating graph ranks movies almost exactly by number of ratings (rank correlation 0.968, against 0.145 for ours on `SIMILAR`); it keeps the year inside titles and ignores duplicate titles; and it models sparse tags as two extra node types. One of its ideas was adopted: a full-text index on tags. Full comparison, prompt and evidence: [docs/anti_genericity.md](docs/anti_genericity.md).

