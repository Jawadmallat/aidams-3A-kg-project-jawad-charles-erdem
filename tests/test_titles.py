"""Server-side title resolution: the choice rules, then the fixture graph.

`pick` is pure, so the rules (every word present, exact title first, then
score, ties flagged) are tested on hand-made candidates; the last tests run
the full-text query on the fixture (skipped without NEO4J_TEST_URI).
"""

import pytest

from app.models import MovieRef
from app.services.titles import fold, lucene_query, pick, resolve_movies, words


def candidate(movie_id: int, title: str, score: float, year: int | None = 1995) -> dict:
    """A full-text hit as RESOLVE_QUERY returns it."""
    return {"movieId": movie_id, "title": title, "year": year, "score": score}


def test_fold_lowers_and_strips_accents():
    """Users type accents MovieLens leaves out, and the other way round."""
    assert fold("Amélie") == "amelie"
    assert fold("LÉON") == "leon"


def test_lucene_query_escapes_syntax_and_disarms_operators():
    """Titles with Lucene syntax become plain word queries."""
    assert lucene_query("Star Wars: Episode IV (1977)!") == r"star wars\: episode iv \(1977\)\!"
    assert lucene_query("Mission: Impossible - Rogue Nation") == \
        r"mission\: impossible \- rogue nation"
    assert lucene_query("NOT AND OR") == "not and or"


def test_words_split_on_punctuation():
    """Words ignore punctuation and case."""
    assert words("The Godfather: Part II") == {"the", "godfather", "part", "ii"}


def test_an_exact_title_beats_a_higher_score():
    """'Heat' is Heat, even if 'Dead Heat' scores higher."""
    ref = MovieRef(param="m", title="Heat")
    best = pick(ref, [candidate(1, "Dead Heat", 5.0), candidate(6, "Heat", 4.0)])
    assert (best["movieId"], best["matchedTitle"], best["ambiguous"]) == (6, "Heat", False)


def test_without_an_exact_title_the_best_score_wins():
    """A partial title takes the best-scoring candidate that has all its words."""
    ref = MovieRef(param="m", title="Matrix")
    best = pick(ref, [candidate(6365, "The Matrix Reloaded", 4.3),
                      candidate(2571, "The Matrix", 4.9)])
    assert best["movieId"] == 2571
    assert best["score"] == 4.9


def test_every_word_must_be_in_the_matched_title():
    """'Toy Story 5' must not quietly become 'Toy Story'."""
    ref = MovieRef(param="m", title="Toy Story 5")
    result = pick(ref, [candidate(1, "Toy Story", 6.0), candidate(3114, "Toy Story 2", 5.0)])
    assert result["movieId"] is None
    assert result["matchedTitle"] is None


def test_a_tie_is_flagged_ambiguous_and_resolved_to_the_lowest_id():
    """Two films called Emma in 1996: pick one deterministically, say it is a tie."""
    ref = MovieRef(param="m", title="Emma", year=1996)
    result = pick(ref, [candidate(26958, "Emma", 4.89, 1996), candidate(838, "Emma", 4.89, 1996)])
    assert (result["movieId"], result["ambiguous"]) == (838, True)


def test_no_candidate_is_unresolved_and_keeps_the_request():
    """The response still says what was asked for."""
    result = pick(MovieRef(param="m1", title="Nope", year=2001), [])
    assert result == {"param": "m1", "title": "Nope", "year": 2001, "movieId": None,
                      "matchedTitle": None, "matchedYear": None, "score": None,
                      "ambiguous": False}


def test_no_reference_needs_no_database():
    """Questions naming no movie never open a session."""
    assert not resolve_movies(driver=None, refs=[])


def test_resolution_on_the_fixture(seeded_graph):
    """Titles, accents-free spelling and unknown movies on the real index."""
    refs = [MovieRef(param="a", title="toy story"), MovieRef(param="b", title="Matrix"),
            MovieRef(param="c", title="Toy Story", year=2020),
            MovieRef(param="d", title="Movie That Does Not Exist")]
    resolved = {movie["param"]: movie["movieId"] for movie in resolve_movies(seeded_graph, refs)}
    assert resolved == {"a": 1, "b": 2571, "c": None, "d": None}


@pytest.mark.parametrize("title", ["!!!", "   ", "(", '"'])
def test_titles_made_of_syntax_do_not_break_the_index(seeded_graph, title):
    """Punctuation-only titles resolve to nothing instead of a Lucene error."""
    result = resolve_movies(seeded_graph, [MovieRef(param="x", title=title)])
    assert result[0]["movieId"] is None
