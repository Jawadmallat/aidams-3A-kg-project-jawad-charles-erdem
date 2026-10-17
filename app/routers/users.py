"""User endpoints: /api/users/{id}/recommendations now, /discover later."""

from fastapi import APIRouter, Depends, HTTPException, Query
from neo4j import Driver

from app.db import get_driver
from app.models import RecommendationsResponse
from app.services import users as service

DEFAULT_LIMIT = 10
MAX_LIMIT = 50

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("/{user_id}/recommendations", response_model=RecommendationsResponse)
def recommendations(
    user_id: int,
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    driver: Driver = Depends(get_driver),
) -> RecommendationsResponse:
    """Personal picks: what the users most like-minded to this one liked."""
    found = service.recommendations(driver, user_id, limit)
    if found is None:
        raise HTTPException(status_code=404, detail=f"user {user_id} not found")
    return RecommendationsResponse(**found)
