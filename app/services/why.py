"""Why was this movie recommended? Graph paths first, AI wording second.

The explanation is the evidence from app/services/paths.py: like-minded
users and genre paths, read from Neo4j. The LLM may only reword that
evidence into a short summary, and the summary is checked against it:
every number and genre it states must appear in the paths. With no
evidence the endpoint says so and the LLM is not called, so it cannot
invent a reason.
"""

import json
import re

from neo4j import Driver

from app import llm
from app.models import WhySummary
from app.services.paths import recommendation_evidence

PAIR_QUERY = """
MATCH (u:User {userId: $userId})
OPTIONAL MATCH (m:Movie {movieId: $movieId})
OPTIONAL MATCH (u)-[r:RATED]->(m)
RETURN m.title AS title, r.rating AS rating,
       COLLECT { MATCH (g:Genre) RETURN g.name } AS allGenres
"""

SYSTEM_PROMPT = """You explain a movie recommendation to the user it was made for.
Use ONLY the facts in the JSON evidence. Do not describe the movie, its plot,
cast or reputation, and do not add numbers or genres that are not in the
evidence. Write two or three plain sentences addressed to the user ("you").
Copy numbers exactly as given. Refer to other users as "user <peerId>".
Use plain words, not the JSON field names: sharedCount is the number of
movies you and that user both liked; likedMoviesInGenre is how many movies
of that genre you liked yourself."""

NO_EVIDENCE_NOTE = (
    "No graph path links this user to this movie: no like-minded user liked "
    "it and it shares no genre with the user's liked movies. There is no "
    "evidence-based reason to recommend it."
)


UNVERIFIED_NOTE = (
    "UNVERIFIED: the AI summary states {claims}, which the graph paths do not "
    "contain. Trust the paths, not the summary."
)

ALREADY_RATED_NOTE = (
    "This user already rated this movie ({rating}), so /recommendations would "
    "not suggest it; the paths show what links them anyway."
)


class UnknownUser(Exception):
    """The user id is not in the graph."""


class UnknownMovie(Exception):
    """The movie id is not in the graph."""


def check_summary(summary: str, evidence: dict, user_id: int, movie: dict) -> dict:
    """Flag numbers and genres the summary states but the evidence does not.

    Numbers are compared as integers (written-out numbers such as "five"
    are not checked); genres are matched as whole words against all the
    genres of the graph, so "Comedy" is only allowed if a genre path has it.
    """
    allowed_numbers = set(re.findall(r"\d+", json.dumps(evidence)))
    allowed_numbers |= set(re.findall(r"\d+", movie["title"]))
    allowed_numbers |= {str(user_id), str(len(evidence["peers"])),
                        str(len(evidence["sharedGenres"]))}
    unsupported = sorted(set(re.findall(r"\d+", summary)) - allowed_numbers, key=int)
    shared_genres = {path["genre"].lower() for path in evidence["sharedGenres"]}
    for genre in movie["allGenres"]:
        mentioned = re.search(rf"(?<![\w-]){re.escape(genre)}(?![\w-])", summary, re.IGNORECASE)
        if mentioned and genre.lower() not in shared_genres:
            unsupported.append(genre)
    return {"verified": not unsupported, "unsupported": unsupported}


def summarize(evidence: dict, title: str) -> tuple[str | None, str | None]:
    """Ask the LLM to reword the evidence; (summary, note), one of them None."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"recommendedMovie": title, **evidence})},
    ]
    try:
        return llm.complete_json(messages, WhySummary).summary, None
    except llm.LLMNotConfigured:
        return None, "No LLM configured: the paths are the explanation."
    except llm.LLMError as error:
        return None, f"AI summary unavailable ({error}); the paths are the explanation."


def explain_recommendation(driver: Driver, user_id: int, movie_id: int) -> dict:
    """The verified paths for (user, movie), plus a checked AI summary of them.

    Raises UnknownUser / UnknownMovie for ids not in the graph.
    """
    records = driver.execute_query(PAIR_QUERY, userId=user_id, movieId=movie_id).records
    if not records:
        raise UnknownUser(user_id)
    movie = records[0].data()
    if movie["title"] is None:
        raise UnknownMovie(movie_id)

    evidence = recommendation_evidence(driver, user_id, movie_id)
    has_evidence = bool(evidence["peers"] or evidence["sharedGenres"])
    summary, note, summary_check = None, NO_EVIDENCE_NOTE, None
    if has_evidence:
        summary, note = summarize(evidence, movie["title"])
        if summary is not None:
            summary_check = check_summary(summary, evidence, user_id, movie)
            if not summary_check["verified"]:
                note = UNVERIFIED_NOTE.format(claims=", ".join(summary_check["unsupported"]))
    if movie["rating"] is not None and note is None:
        note = ALREADY_RATED_NOTE.format(rating=movie["rating"])
    return {
        "userId": user_id, "movieId": movie_id, "title": movie["title"],
        "alreadyRated": movie["rating"], "hasEvidence": has_evidence, "paths": evidence,
        "summary": summary, "summarySource": "llm" if summary is not None else None,
        "summaryCheck": summary_check, "note": note,
    }
