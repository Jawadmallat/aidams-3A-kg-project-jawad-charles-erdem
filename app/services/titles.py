"""Resolve the movie titles a question names to movieIds, on the server.

The evaluation showed the LLM cannot be trusted to search titles itself: the
full-text index matches ANY word ("the OR matrix"), and the model often kept
every hit instead of the best one. So the model now only names the movies;
this module picks the match, deterministically and visibly:

1. full-text candidates for the title (Lucene syntax escaped), optionally
   filtered on the year;
2. only candidates whose title contains every word of the requested title
   are kept, so "Toy Story 2" can never silently become "Toy Story";
3. an exact title match wins, then the full-text score, then the lowest id.

A tie on the last two criteria is flagged `ambiguous` (two films called
"Emma" in 1996). No candidate left means the movie is not in the graph.
"""

import re
import unicodedata

import neo4j
from neo4j import Driver

from app.models import MovieRef

# How many full-text hits are considered before the word filter.
CANDIDATES = 10

RESOLVE_QUERY = """
CALL db.index.fulltext.queryNodes('movie_title_fulltext', $text) YIELD node, score
WHERE $year IS NULL OR node.year = $year
RETURN node.movieId AS movieId, node.title AS title, node.year AS year, score
ORDER BY score DESC, movieId
LIMIT $candidates
"""

_LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')


def fold(text: str) -> str:
    """Lower-cased and without accents: MovieLens writes "Amelie", users "Amélie"."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def lucene_query(title: str) -> str:
    """The title as a plain Lucene query: folded, special characters escaped.

    Lower-casing also disarms the AND / OR / NOT operators; the index
    analyzer lower-cases titles anyway (but keeps accents, hence fold).
    """
    return _LUCENE_SPECIAL.sub(r"\\\1", fold(title))


def words(title: str) -> set[str]:
    """The folded words of a title."""
    return set(re.findall(r"\w+", fold(title)))


def pick(ref: MovieRef, candidates: list[dict]) -> dict:
    """Choose the match for `ref` among full-text candidates (pure function)."""
    wanted = words(ref.title)
    kept = [c for c in candidates if wanted <= words(c["title"])]
    unresolved = {
        "param": ref.param, "title": ref.title, "year": ref.year, "movieId": None,
        "matchedTitle": None, "matchedYear": None, "score": None, "ambiguous": False,
    }
    if not kept:
        return unresolved

    def rank(candidate: dict) -> tuple:
        exact = fold(candidate["title"]) == fold(ref.title)
        return (not exact, -candidate["score"], candidate["movieId"])

    kept.sort(key=rank)
    best = kept[0]
    ambiguous = len(kept) > 1 and rank(kept[1])[:2] == rank(best)[:2]
    return {
        **unresolved, "movieId": best["movieId"], "matchedTitle": best["title"],
        "matchedYear": best["year"], "score": round(best["score"], 3), "ambiguous": ambiguous,
    }


def resolve_movies(driver: Driver, refs: list[MovieRef]) -> list[dict]:
    """Resolve every named movie; unresolved ones have a null movieId."""
    if not refs:
        return []
    resolved = []
    with driver.session(default_access_mode=neo4j.READ_ACCESS) as session:
        for ref in refs:
            query = lucene_query(ref.title)
            candidates = [] if not query.strip() else session.execute_read(
                lambda tx, q=query, y=ref.year: tx.run(
                    RESOLVE_QUERY, text=q, year=y, candidates=CANDIDATES
                ).data()
            )
            resolved.append(pick(ref, candidates))
    return resolved
