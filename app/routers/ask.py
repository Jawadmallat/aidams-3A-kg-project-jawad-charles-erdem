"""POST /api/ask: a question in English, answered by validated read-only Cypher."""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from neo4j import Driver

from app import llm
from app.db import get_driver
from app.models import AskRequest, AskResponse
from app.services import ask as service

router = APIRouter(prefix="/api", tags=["ask"])

ERROR_RESPONSES = {
    400: {"description": "The generated query was rejected; the body lists every check"},
    502: {"description": "The LLM provider failed or answered off-schema"},
    503: {"description": "No LLM configured (LLM_API_KEY / LLM_MODEL), or database down"},
    504: {"description": "The generated query exceeded the time limit"},
}


@router.post("/ask", response_model=AskResponse, responses=ERROR_RESPONSES)
def ask(request: AskRequest, driver: Driver = Depends(get_driver)):
    """Ask the graph a question in natural language.

    An LLM writes the Cypher; it runs only after the guardrails, an EXPLAIN
    dry run and inside a read transaction. The response returns the query
    and every check with the rows, so the AI-written part can be inspected.
    """
    try:
        return service.answer(driver, request.question)
    except llm.LLMNotConfigured:
        return JSONResponse(status_code=503, content={
            "detail": "LLM not configured: set LLM_API_KEY and LLM_MODEL (see .env.example)"
        })
    except llm.LLMError as error:
        return JSONResponse(status_code=502, content={"detail": str(error)})
    except service.QueryRejected as error:
        return JSONResponse(status_code=400, content=service.rejection_body(error))
    except service.QueryTimeout as error:
        return JSONResponse(status_code=504, content={"detail": str(error)})
