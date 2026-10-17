"""Movie endpoints: /api/movies/{id}/similar now, /api/movies/trending later."""

from fastapi import APIRouter, Depends, HTTPException, Query
from neo4j import Driver

from app.db import get_driver
from app.models import SimilarResponse
from app.services import movies as service

DEFAULT_LIMIT = 10
MAX_LIMIT = 50

router = APIRouter(prefix="/api/movies", tags=["movies"])


@router.get("/{movie_id}/similar", response_model=SimilarResponse)
def similar(
    movie_id: int,
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    driver: Driver = Depends(get_driver),
) -> SimilarResponse:
    """Movies most similar to this one (precomputed GDS, co-rating fallback)."""
    found = service.similar_movies(driver, movie_id, limit)
    if found is None:
        raise HTTPException(status_code=404, detail=f"movie {movie_id} not found")
    return SimilarResponse(**found)
