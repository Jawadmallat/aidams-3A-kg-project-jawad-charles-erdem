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
| similar (gds) | 50 | 5.8 | 10.9 | 6.4 |
| similar (fallback) | 50 | 5.9 | 18.0 | 7.1 |
| recommendations | 50 | 138.8 | 626.3 | 193.3 |
| discover | 50 | 7.2 | 18.1 | 9.1 |
| trending page 1 | 50 | 30.5 | 42.1 | 32.1 |
| trending page 10 | 50 | 30.4 | 40.3 | 31.6 |

Reading the table:

* **Both similarity paths answer in single-digit milliseconds, and both lean on
  precomputation.** The GDS path reads at most 10 `SIMILAR` relationships
  through the `movieId` constraint (PROFILE: `NodeUniqueIndexSeek`, 2 db hits,
  then an expand over 10 rows). The fallback traverses the movie's small
  audience (under 10 likers by definition) and reads precomputed `likesCount`
  properties for its Jaccard denominators — before that rewrite (below) it
  measured **116 ms p50 / 248 ms p95**, which is what recomputing everything
  live costs and the clearest argument for precomputing `SIMILAR` itself.
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

## Rewrite 3: the fallback's Jaccard denominators (520,865 -> 2,781 db hits)

A review finding on PR #2 (Charles): the fallback's traversal is bounded by
the fallback movie's small audience, but its original Jaccard denominator ran
a `COUNT { (other)<-[:RATED]-() ... }` **per candidate**, and candidates are
whatever the movie's likers also liked — frequently very popular movies.
Measured for one ordinary fallback movie on the ml-25m sample: 242 candidates
whose counts iterated **111,847** rater relationships, **520,865 db hits** per
request — the whole explanation of the old 248 ms p95.

`run_gds` now stores every movie's `likesCount` (recomputed on each run, like
`community` and `pagerank`), and the fallback reads two properties instead of
counting raters:

```cypher
WITH other, shared, toFloat(shared) /
     (coalesce(m.likesCount, 0) + coalesce(other.likesCount, 0) - shared) AS score
```

Same movie, same results: **2,781 db hits** (187x fewer). The scenario went
from 116 ms p50 / 248 ms p95 to **5.9 ms p50 / 18 ms p95** — on par with the
GDS path. The counts are as fresh as the last `run_gds` run, exactly like the
`SIMILAR` relationships the fallback stands in for; an integration test checks
`likesCount` matches the live count after `run_gds`.

A second finding from the same review (Charles): the `0` defaults above break
when `likesCount` is absent, for example after a seed without `run_gds`. The
denominator goes negative, or reaches zero and fails the request with `/ by zero`.
Both `m` and `other` now fall back to the live count,
`coalesce(X.likesCount, COUNT { (X)<-[r:RATED]-(:User) WHERE r.rating >= $liked })`.
`coalesce` runs the `COUNT` only when the property is missing, so the fast path
is unchanged: still **2,781 db hits** for the same movie, against **584,482**
with the live branch forced for every movie (same scores). A stale count cannot
be detected this way and can still skew a score, or fail the request if the
denominator reaches zero. The rule: rerun `run_gds` after a seed or new ratings.

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
