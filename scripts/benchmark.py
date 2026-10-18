"""Measure the latency of every API endpoint against the connected database.

The app is built in-process around the same driver the API uses and exercised
through its HTTP layer (TestClient), so a measurement covers validation,
Cypher execution and serialization without network noise or a separate server.
Run it against the ml-25m 5% sample (docs/decisions.md, "Benchmark dataset")
so the numbers mean something:

    uv run --env-file .env python -m scripts.benchmark --repeats 50 --warmup 5
    uv run --env-file .env python -m scripts.benchmark --markdown

Scenario ids are drawn from the graph itself: movies with SIMILAR (the GDS
path), rated movies without SIMILAR (the fallback path), and users with
enough ratings to have peers.
"""

import argparse
import statistics
import sys
import time

from fastapi.testclient import TestClient
from neo4j import Driver

from app.main import create_app
from scripts.db import get_driver

GDS_MOVIES_QUERY = """
MATCH (m:Movie) WHERE (m)-[:SIMILAR]->()
WITH m ORDER BY m.movieId LIMIT $count
RETURN collect(m.movieId) AS ids
"""

FALLBACK_MOVIES_QUERY = """
MATCH (m:Movie)
WHERE NOT (m)-[:SIMILAR]->()
  AND COUNT { (m)<-[r:RATED]-() WHERE r.rating >= 4.0 } >= 2
WITH m ORDER BY m.movieId LIMIT $count
RETURN collect(m.movieId) AS ids
"""

ACTIVE_USERS_QUERY = """
MATCH (u:User)
WHERE COUNT { (u)-[:RATED]->() } >= 20
WITH u ORDER BY u.userId LIMIT $count
RETURN collect(u.userId) AS ids
"""


def pick_ids(driver: Driver, query: str, count: int) -> list[int]:
    """Deterministic scenario ids straight from the connected graph."""
    return driver.execute_query(query, count=count).records[0]["ids"]


def build_scenarios(driver: Driver, count: int) -> dict[str, list[str]]:
    """Scenario name -> list of URLs to cycle through."""
    gds_movies = pick_ids(driver, GDS_MOVIES_QUERY, count)
    fallback_movies = pick_ids(driver, FALLBACK_MOVIES_QUERY, count)
    users = pick_ids(driver, ACTIVE_USERS_QUERY, count)
    return {
        "similar (gds)": [f"/api/movies/{m}/similar" for m in gds_movies],
        "similar (fallback)": [f"/api/movies/{m}/similar" for m in fallback_movies],
        "recommendations": [f"/api/users/{u}/recommendations" for u in users],
        "discover": [f"/api/users/{u}/discover" for u in users],
        "trending page 1": ["/api/movies/trending?skip=0&limit=10"],
        "trending page 10": ["/api/movies/trending?skip=90&limit=10"],
    }


def measure(client: TestClient, urls: list[str], repeats: int, warmup: int) -> list[float]:
    """Request latencies in milliseconds, cycling through the scenario's URLs."""
    for i in range(warmup):
        client.get(urls[i % len(urls)])
    samples = []
    for i in range(repeats):
        url = urls[i % len(urls)]
        started = time.perf_counter()
        response = client.get(url)
        elapsed = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            raise RuntimeError(f"{url} answered {response.status_code}")
        samples.append(elapsed)
    return samples


def percentile(samples: list[float], fraction: float) -> float:
    """The value below which `fraction` of the sorted samples fall."""
    ordered = sorted(samples)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def summarize(results: dict[str, list[float]], markdown: bool) -> str:
    """One table row per scenario: requests, p50, p95, mean."""
    header = f"{'scenario':<20} {'requests':>8} {'p50 ms':>8} {'p95 ms':>8} {'mean ms':>8}"
    separator = "-" * len(header)
    if markdown:
        header = "| scenario | requests | p50 ms | p95 ms | mean ms |"
        separator = "|---|---|---|---|---|"
    lines = [header, separator]
    for name, samples in results.items():
        row = (name, len(samples), percentile(samples, 0.50),
               percentile(samples, 0.95), statistics.mean(samples))
        if markdown:
            lines.append(f"| {row[0]} | {row[1]} | {row[2]:.1f} | {row[3]:.1f} | {row[4]:.1f} |")
        else:
            lines.append(f"{row[0]:<20} {row[1]:>8} {row[2]:>8.1f} {row[3]:>8.1f} {row[4]:>8.1f}")
    return "\n".join(lines)


def run(driver: Driver, repeats: int, warmup: int, ids_per_scenario: int) -> dict[str, list[float]]:
    """Measure every scenario against the given driver's database."""
    scenarios = build_scenarios(driver, ids_per_scenario)
    client = TestClient(create_app(driver=driver))
    return {
        name: measure(client, urls, repeats, warmup)
        for name, urls in scenarios.items()
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repeats", type=int, default=50, help="requests per scenario")
    parser.add_argument("--warmup", type=int, default=5, help="unmeasured requests first")
    parser.add_argument("--ids", type=int, default=10, help="distinct ids cycled per scenario")
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    with get_driver() as driver:
        counts = driver.execute_query(
            "MATCH (u:User) WITH count(u) AS users MATCH ()-[r:RATED]->() "
            "RETURN users, count(r) AS ratings"
        ).records[0]
        print(f"database: {counts['users']:,} users, {counts['ratings']:,} ratings")
        results = run(driver, args.repeats, args.warmup, args.ids)
    print(summarize(results, args.markdown))
    return 0


if __name__ == "__main__":
    sys.exit(main())
