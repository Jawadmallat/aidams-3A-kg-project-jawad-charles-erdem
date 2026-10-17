# Graph contract and data decisions

What the graph contains, how to load it, and the conventions every query should follow. Owned by Part A (data, modeling and ML). The reasoning behind each choice is in `demo.ipynb`, section 1.

| Status | Part |
|---|---|
| Available | Users, movies, genres, ratings, tags; constraints and indexes; test fixture |
| Available (`scripts/run_gds.py`) | `SIMILAR` relationships, `Movie.community`, `Movie.pagerank`, `User.community` |

## Local setup (until the Docker Compose stack exists)

Two Neo4j containers: one for development data, one disposable for the integration tests (they wipe it). Ports 7688 and 7689 avoid clashing with another local Neo4j on the defaults.

```bash
docker run -d --name kg-neo4j -p 7475:7474 -p 7688:7687 \
  -e NEO4J_AUTH=neo4j/<choose-a-password> \
  -e 'NEO4J_PLUGINS=["apoc","graph-data-science"]' \
  -e 'NEO4J_dbms_security_procedures_unrestricted=gds.*,apoc.*' \
  -e NEO4J_server_memory_heap_max__size=2G -e NEO4J_server_memory_pagecache_size=1G \
  -v kg-neo4j-data:/data neo4j:5.26

docker run -d --name kg-neo4j-test -p 7689:7687 \
  -e NEO4J_AUTH=neo4j/<same-password> \
  -e 'NEO4J_PLUGINS=["apoc","graph-data-science"]' \
  -e 'NEO4J_dbms_security_procedures_unrestricted=gds.*,apoc.*' \
  -e NEO4J_server_memory_heap_max__size=1G -e NEO4J_server_memory_pagecache_size=256M \
  neo4j:5.26
```

The first start downloads the plugins and takes a minute or two. Then create `.env` at the repository root (never committed):

```bash
NEO4J_URI=bolt://localhost:7688
NEO4J_USER=neo4j
NEO4J_PASSWORD=<the password above>
DATA_DIR=data
NEO4J_TEST_URI=bolt://localhost:7689
```

Install the dependencies and check everything works:

```bash
uv sync
uv run --env-file .env pytest -q      # all tests, including Neo4j integration tests
uv run pytest -q                      # without Neo4j: integration tests are skipped
```

Add new dependencies with `uv add <package>` (runtime) or `uv add --dev <package>` (tools, notebooks), so `pyproject.toml` and `uv.lock` stay the source of truth.

## Loading the data

```bash
uv run python scripts/download_data.py                        # ml-latest-small into data/
uv run --env-file .env python -m scripts.seed_data            # load it (safe to re-run)
uv run --env-file .env python -m scripts.seed_data --reset    # wipe, then load
uv run --env-file .env python -m scripts.seed_data --data-dir tests/fixtures/movielens --reset
uv run --env-file .env python -m scripts.run_gds              # SIMILAR, communities, PageRank (safe to re-run)
```

Run `run_gds` after every load: the seed script does not compute the GDS properties.

The scripts are run as modules (`python -m scripts.x`) and read these environment variables:

| Variable | Used by | Example |
|---|---|---|
| `NEO4J_URI` | all data scripts | `bolt://localhost:7687` |
| `NEO4J_USER` | all data scripts | `neo4j` |
| `NEO4J_PASSWORD` | all data scripts | (secret, `.env` only) |
| `DATA_DIR` | download and seed scripts | `data` |
| `NEO4J_TEST_URI` | integration tests only; **the tests wipe this database** | `bolt://localhost:7689` |

Neo4j must be **5.26** with the `apoc` and `graph-data-science` plugins and `dbms.security.procedures.unrestricted=gds.*,apoc.*`.

## Schema

```text
(:User)-[:RATED {rating, timestamp}]->(:Movie)
(:User)-[:TAGGED {tag, timestamp}]->(:Movie)
(:Movie)-[:IN_GENRE]->(:Genre)
(:Movie)-[:SIMILAR {score}]->(:Movie)
```

| Element | Property | Type | Notes |
|---|---|---|---|
| `User` | `userId` | integer | unique |
| | `community` | integer or absent | the community most of the user's liked movies belong to; absent when none of their liked movies has a community |
| `Movie` | `movieId` | integer | unique; the only key |
| | `title` | string | cleaned: no year, article in front (`The Matrix`); **not unique** |
| | `year` | integer or null | release year; null for 12 movies |
| | `genres` | list of strings | same values as the `IN_GENRE` links; empty for 34 movies |
| | `imdbId` | string | IMDb format, `tt0133093` |
| | `tmdbId` | integer or null | null for 8 movies |
| | `community` | integer or absent | Louvain on `SIMILAR`, ids 0 to n−1; absent for movies with fewer than 10 likes |
| | `pagerank` | float or absent | PageRank on `SIMILAR`; high when many movies list this one as similar; absent for movies with fewer than 10 likes |
| `Genre` | `name` | string | unique; 19 genres (`(no genres listed)` is not a genre) |
| `RATED` | `rating` | float | 0.5 to 5.0 in half stars |
| | `timestamp` | integer | Unix epoch seconds |
| `TAGGED` | `tag` | string | lower-cased, whitespace collapsed |
| | `timestamp` | integer | Unix epoch seconds |
| `SIMILAR` | `score` | float in (0, 1] | Jaccard similarity of the sets of users who liked each movie; top 10 per movie; only between movies with at least 10 likes |

## Constraints and indexes

Applied by `scripts/schema.py` (and by the seed script before loading).

| Name | Kind | On | Serves |
|---|---|---|---|
| `user_id_unique` | uniqueness constraint | `User.userId` | user lookups, `MERGE` during seeding |
| `movie_id_unique` | uniqueness constraint | `Movie.movieId` | movie lookups, `MERGE` during seeding |
| `genre_name_unique` | uniqueness constraint | `Genre.name` | genre lookups |
| `movie_title_fulltext` | full-text index | `Movie.title` | title search by words |
| `movie_year` | range index | `Movie.year` | year filters and ranges |
| `rated_timestamp` | range index | `RATED.timestamp` | time windows (trending, temporal split) |
| `tagged_tag` | range index | `TAGGED.tag` | "movies tagged X" |
| `tagged_tag_fulltext` | full-text index | `TAGGED.tag` | word search in tags ("dark" finds "dark comedy"), for `/api/ask` |
| `movie_community` | range index | `Movie.community` | movies of a community (discover) |
| `movie_pagerank` | range index | `Movie.pagerank` | ordering by PageRank (trending) |
| `user_community` | range index | `User.community` | users of a community |

Ask Part A for any other index an endpoint needs, with the query it serves, so it lands in `scripts/schema.py` rather than in ad hoc code.

## Conventions for queries

* **Ids are integers.** Pass `movieId` and `userId` as integers. A string parameter matches nothing and returns no error.
* **"Liked" means a rating of 4.0 or higher**, everywhere: recommendations, similarity, communities and the ML label. 48% of ratings qualify.
* **"Recent" is measured from the last rating in the data, not from today.** The small dataset ends in September 2018, so a window ending today is empty. Anchor windows on the most recent `RATED.timestamp`.
* **Titles are not unique** (five titles are shared by different films, for example two *Emma (1996)*). Find movies by title through the full-text index and expect several results:
  ```cypher
  CALL db.index.fulltext.queryNodes('movie_title_fulltext', $text) YIELD node, score
  RETURN node.movieId, node.title, node.year ORDER BY score DESC LIMIT 5
  ```
* **Genres exist twice on purpose.** Use the `genres` list to filter a movie you already have (`'Comedy' IN m.genres`); use `IN_GENRE` when the genre is part of the path (movies sharing a genre, explanation paths, GDS).
* **`IMAX` is a MovieLens genre** although it is a screening format. It is kept as MovieLens defines it.
* **Tags are a property of `TAGGED`**, not nodes, so each relationship records who tagged which movie with what.
* **GDS results only exist for movies with at least 10 likes** (1,182 of 9,742 in ml-latest-small, covering 74% of all likes). A movie without `SIMILAR` needs a fallback, for example co-rating Cypher (users who liked this movie also liked...). Treat a missing `pagerank` as 0.
* **`SIMILAR` is directed** (`a` lists `b` among its top 10), and the reverse link may not exist. Query it as `(m)-[:SIMILAR]->(other)` ordered by `score`.
* **About half of the users share one community** (the widely liked classics), so `discover` for them is close to popularity.

## Benchmark dataset

For performance measurements (PROFILE, latency before and after indexes), use the 5% user sample of `ml-25m`:

```bash
uv run python scripts/download_data.py --dataset ml-25m
uv run --env-file .env python -m scripts.seed_data --data-dir data/ml-25m --sample-users 0.05 --seed 42 --reset
uv run --env-file .env python -m scripts.run_gds
```

The seed makes the sample reproducible. Result: 8,127 users, 62,423 movies, 1,265,960 ratings, 39,166 tags, 52,660 `SIMILAR` relationships and 15 communities. Loading takes about 40 seconds and GDS about 3 minutes on a laptop. Use a database you do not run the tests against: the tests wipe theirs.

## Test fixture

`tests/fixtures/movielens/` holds a 20-user, 50-movie subset of ml-latest-small in MovieLens CSV format (regenerate with `python -m scripts.make_fixture`). Facts tests can rely on:

| Fact | Value |
|---|---|
| Users / movies / genres | 20 / 50 / 17 |
| Ratings / tags | 860 / 282 |
| User who rated every movie (no unseen movie left to recommend) | `414` |
| User with the fewest ratings | `603` (30 ratings) |
| Users with tags | `474`, `599`, `606` |
| Known movies | `1` Toy Story (1995), `2571` The Matrix (1999) |
| Movies with GDS results after `run_gds` | 29 (the other 21 have no `SIMILAR`: use them to test fallbacks) |
| Users with a community after `run_gds` | all 20 |
| Last rating | 2018-02-20 (`1519122884`) |

In pytest, the `seeded_graph` fixture from `tests/conftest.py` wipes the database at `NEO4J_TEST_URI`, loads the fixture once per session and returns the driver. Tests that use it are skipped when `NEO4J_TEST_URI` is not set.

```python
def test_toy_story_exists(seeded_graph):
    records = seeded_graph.execute_query(
        "MATCH (m:Movie {movieId: $id}) RETURN m.title AS title", id=1
    ).records
    assert records[0]["title"] == "Toy Story"
```
