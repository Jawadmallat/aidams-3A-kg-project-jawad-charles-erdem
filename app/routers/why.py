"""GET /api/users/{id}/recommendations/{movie_id}/why: the graph paths behind a pick."""

from fastapi import APIRouter, Depends, HTTPException
from neo4j import Driver

from app.db import get_driver
from app.models import WhyResponse
from app.services import why as service

router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("/{user_id}/recommendations/{movie_id}/why", response_model=WhyResponse)
def why(user_id: int, movie_id: int, driver: Driver = Depends(get_driver)) -> WhyResponse:
    """Why this movie for this user: verified graph paths and an AI summary of them.

    Works without an LLM (the summary is then null); with no path between
    the user and the movie it says so instead of inventing a reason.
    """
    try:
        return WhyResponse(**service.explain_recommendation(driver, user_id, movie_id))
    except service.UnknownUser as error:
        raise HTTPException(status_code=404, detail=f"user {user_id} not found") from error
    except service.UnknownMovie as error:
        raise HTTPException(status_code=404, detail=f"movie {movie_id} not found") from error
