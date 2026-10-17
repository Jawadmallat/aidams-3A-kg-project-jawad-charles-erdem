"""Pydantic response models of the API.

Field names match the Neo4j property names (movieId, pagerank, ...) so service
rows map to models without renaming, and responses read like the graph contract
in docs/decisions.md.
"""

from typing import Literal

from pydantic import BaseModel


class MovieSummary(BaseModel):
    """The movie fields every endpoint returns."""

    movieId: int
    title: str
    year: int | None
    genres: list[str]


class SimilarMovie(MovieSummary):
    """A movie similar to the requested one."""

    score: float


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
