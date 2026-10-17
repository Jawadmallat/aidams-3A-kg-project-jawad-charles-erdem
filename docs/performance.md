# Performance evidence

Latency of every endpoint on the benchmark dataset, and the PROFILE plans that
drove the two query rewrites. Owned by Part B (API, Cypher and performance).

## Setup

* **Dataset:** the reproducible 5% user sample of `ml-25m` (docs/decisions.md,
  "Benchmark dataset"): 8,127 users, 62,423 movies, **1,265,960 ratings**,
  52,660 `SIMILAR` relationships, 15 communities.
* **Server:** Neo4j 5.26 in Docker on a laptop (2 GB heap, 1 GB page cache).
* **Method:** `scripts/benchmark.py` builds the app in-process around the same
  driver the API uses and measures through the HTTP layer (validation, Cypher,
  serialization; no network). 50 requests per scenario after 5 warmups,
  cycling through 10 ids read from the graph itself.

Reproduce with:

```bash
uv run python scripts/download_data.py --dataset ml-25m
uv run --env-file .env python -m scripts.seed_data --data-dir data/ml-25m --sample-users 0.05 --seed 42 --reset
uv run --env-file .env python -m scripts.run_gds
uv run --env-file .env python -m scripts.benchmark --repeats 50 --warmup 5 --markdown
```

## Results (after the rewrites below)

| scenario | requests | p50 ms | p95 ms | mean ms |
|---|---|---|---|---|
| similar (gds) | 50 | 2.9 | 4.1 | 3.2 |
| similar (fallback) | 50 | 116.3 | 247.8 | 130.5 |
| recommendations | 50 | 119.1 | 654.6 | 187.8 |
| discover | 50 | 6.9 | 18.9 | 9.0 |
| trending page 1 | 50 | 30.2 | 40.8 | 31.6 |
| trending page 10 | 50 | 29.1 | 40.7 | 30.7 |

Reading the table:

* **similar (gds) at 3 ms vs similar (fallback) at 116 ms** is the argument for
  precomputing `SIMILAR`: the GDS path reads at most 10 relationships through
  the `movieId` constraint (PROFILE: `NodeUniqueIndexSeek`, 2 db hits, then an
  expand over 10 rows), while the fallback recomputes Jaccard live. The
  fallback only ever runs for movies with fewer than 10 likers, so its cost is
  bounded by a small audience.
* **recommendations** traverses 25 peers' full rating histories; its p95 is a
  user whose peers are very active. The `PEER_LIMIT` cap in
  `app/services/users.py` is what keeps this bounded as the dataset grows.
* **trending page 10 costs the same as page 1**: the window aggregation
  dominates and pagination only changes the slice of one ordering.

## Rewrite 1: the trending anchor (full scan -> 2 db hits)

Trending anchors its window on the newest `RATED.timestamp` (the data ends in
2018; docs/decisions.md). The obvious query planned as a **full relationship
scan touching every rating on every request**:

```cypher
MATCH ()-[r:RATED]->()
RETURN r.timestamp AS anchor ORDER BY r.timestamp DESC LIMIT 1
```

```text
Top                                        1 row
  Projection                               dbHits=1,265,960
    DirectedRelationshipTypeScan           dbHits=1,265,961   <- every rating
```

The `rated_timestamp` range index was not considered because nothing told the
planner the property is present. Adding `WHERE r.timestamp IS NOT NULL` lets
it serve `ORDER BY ... DESC LIMIT 1` straight from the index, read backwards:

```cypher
MATCH ()-[r:RATED]->()
WHERE r.timestamp IS NOT NULL
RETURN r.timestamp AS anchor ORDER BY r.timestamp DESC LIMIT 1
```

```text
Limit                                      1 row
  DirectedRelationshipIndexScan            dbHits=2           <- the index, backwards
    RANGE INDEX (anon_0)-[r:RATED(timestamp)]->(anon_1)
```

**Effect: trending p50 dropped from 180 ms to 30 ms** (the anchor dominated
the endpoint). The remaining cost is the window aggregation itself, which is
also index-backed: `DirectedRelationshipIndexSeekByRange` reads the 73,525
ratings inside the 365-day window instead of all 1,265,960.

## Rewrite 2: discover's ordering (computed alias -> composite seek)

Discover filters one community and orders by PageRank, which is why
`movie_community_pagerank` ON `(m.community, m.pagerank)` exists
(`scripts/schema.py`). The first version ordered by a **computed alias**:

```cypher
RETURN ..., round(coalesce(rec.pagerank, 0.0), 4) AS pagerank
ORDER BY pagerank DESC
```

No index can back an ordering on `round(coalesce(...))`, and the plan fell
back to the single-column community seek plus a full sort of the candidates.
Ordering by the **raw property** and filtering `rec.pagerank IS NOT NULL`
(which excludes nothing: `run_gds` writes `community` and `pagerank` together)
engages the composite index:

```text
Top        `rec.pagerank` DESC, movieId ASC LIMIT $limit + $skip   307 rows in
  ...
    NodeIndexSeek   RANGE INDEX rec:Movie(community, pagerank)
                    WHERE community = ...                          dbHits=453
```

The equality and the pagerank filter are answered by one seek. A small `Top`
remains by choice: the `movieId` tiebreak that keeps pagination deterministic
cannot come from the index, and sorting the ~300 candidate rows of a community
is trivial next to the anti-join.

**Known cost:** the `NOT EXISTS { (me)-[:RATED]->(rec) }` anti-join
(`Expand(Into)`, ~122k db hits for a very active user) dominates discover's
9 ms mean. Acceptable at this scale; the candidate set is bounded by the
community, never by the catalogue.

## Checklist over every endpoint plan

Checked on the PROFILE output of all six scenarios:

* every id lookup is a `NodeUniqueIndexSeek` on the `movieId`/`userId`
  uniqueness constraints — no label scans;
* **no `CartesianProduct` operator anywhere**;
* every query is parameterized (`$movieId`, `$liked`, `$anchor`, ...) — no
  string-built Cypher, plans stay cached;
* every query ends in `LIMIT` (with `SKIP` where paginated), with a
  deterministic `movieId` tiebreak so pages never overlap;
* unbounded expansions do not occur: genre and community sets are always
  entered from a seeded node (movie or user), per docs/decisions.md.
