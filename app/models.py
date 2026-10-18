"""Pydantic response models of the API.

Field names match the Neo4j property names (movieId, pagerank, ...) so service
rows map to models without renaming, and responses read like the graph contract
in docs/decisions.md.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class MovieSummary(BaseModel):
    """The movie fields every endpoint returns."""

    movieId: int
    title: str
    year: int | None
    genres: list[str]


class SimilarMovie(MovieSummary):
    """A movie similar to the requested one."""

    score: float


class Recommendation(MovieSummary):
    """A recommended movie with the evidence behind its score."""

    score: float
    peerCount: int
    genreMatches: int


class RecommendationsResponse(BaseModel):
    """Answer of GET /api/users/{id}/recommendations.

    An empty `results` means the user is known but has nothing left to be
    recommended (rated everything, or no like-minded users yet).
    """

    userId: int
    results: list[Recommendation]


class DiscoverMovie(MovieSummary):
    """A community pick: central (PageRank) in the user's Louvain community."""

    community: int
    pagerank: float


class DiscoverResponse(BaseModel):
    """Answer of GET /api/users/{id}/discover.

    `community` is the user's Louvain community; null when none of their
    liked movies has one, in which case `results` is empty.
    """

    userId: int
    community: int | None
    results: list[DiscoverMovie]


class TrendingMovie(MovieSummary):
    """A trending movie with the two signals its score blends."""

    recentRatings: int
    pagerank: float
    score: float


class TrendingResponse(BaseModel):
    """Answer of GET /api/movies/trending.

    `anchor` is the newest RATED timestamp in the data; the window of
    `windowDays` days ends there, not today (the dataset ends in 2018).
    Null anchor means an empty graph.
    """

    anchor: int | None
    windowDays: int
    results: list[TrendingMovie]


class SimilarResponse(BaseModel):
    """Answer of GET /api/movies/{id}/similar.

    `source` tells which branch produced the results: "gds" reads the
    precomputed SIMILAR relationships, "co_rating" is the live Jaccard
    fallback for movies with fewer than 10 likes (docs/decisions.md).
    """

    movieId: int
    title: str
    source: Literal["gds", "co_rating"]
    results: list[SimilarMovie]


class CypherDraft(BaseModel):
    """What the LLM must answer for /api/ask: a query, never the answer itself.

    An empty `cypher` means the model judged the question unanswerable from
    the graph; `explanation` then says why.
    """

    cypher: str
    params: dict[str, Any] = Field(default_factory=dict)
    explanation: str


class AskRequest(BaseModel):
    """Body of POST /api/ask."""

    question: str = Field(min_length=3, max_length=500, examples=[
        "Which 5 movies have the most ratings of 4 stars or more?"
    ])


class GuardrailCheck(BaseModel):
    """One validation step the generated query went through."""

    name: str
    passed: bool
    detail: str


class AskResponse(BaseModel):
    """Answer of POST /api/ask.

    `rows` are facts read from Neo4j by the query in `cypher`. Everything
    else the model wrote (`cypher`, `explanation`) is AI-generated: the query
    passed every check in `checks`, but it may still not be what the question
    meant, which is why it is returned for the reader to inspect. `answered`
    is false when the model declined (no query ran, `cypher` is null).
    """

    question: str
    answered: bool
    cypher: str | None
    params: dict[str, Any]
    explanation: str
    checks: list[GuardrailCheck]
    rows: list[dict[str, Any]]
    rowCount: int
    truncated: bool
    attempts: int
    model: str
