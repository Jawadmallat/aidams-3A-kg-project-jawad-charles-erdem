"""The why endpoint without a database: evidence handling and the fact check.

The driver is a stub answering the pair query, the evidence function is
replaced by fixed paths, and the LLM by a recorder, so the rules are pinned
down: no evidence means no LLM call; no key still returns the paths; the
summary is checked against the paths.
"""

import pytest
from fastapi.testclient import TestClient

from app import llm
from app.main import create_app
from app.models import WhySummary
from app.services import why

ALL_GENRES = ["Action", "Adventure", "Comedy", "Drama", "Sci-Fi", "War"]
EVIDENCE = {
    "peers": [
        {"peerId": 414, "sharedCount": 130, "sharedTitles": ["Go", "Alien"]},
        {"peerId": 380, "sharedCount": 87, "sharedTitles": ["Superman"]},
    ],
    "sharedGenres": [{"genre": "Comedy", "likedMoviesInGenre": 70}],
}
real_complete_json = llm.complete_json
MOVIE = {"title": "Terminator 2: Judgment Day", "rating": None, "allGenres": ALL_GENRES}


class Record:
    """Stands in for a neo4j Record."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def data(self) -> dict:
        """The record as a dict."""
        return self._data


class PairDriver:
    """Answers the pair query with fixed rows; [] means unknown user."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def execute_query(self, *_args, **_kwargs):
        """Return the scripted records."""
        return type("Result", (), {"records": [Record(row) for row in self.rows]})


class RecordingLLM:
    """Answers a fixed summary and counts calls."""

    def __init__(self, summary: str) -> None:
        self.summary = summary
        self.calls = 0

    def __call__(self, messages, schema):
        assert schema is WhySummary
        self.calls += 1
        return WhySummary(summary=self.summary)


@pytest.fixture(name="setup")
def fixture_setup(monkeypatch):
    """Script the movie row, the evidence and the LLM; return a client and the LLM."""
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "m")

    def build(evidence=None, movie=None, summary="You and user 414 both liked 130 movies."):
        recorder = RecordingLLM(summary)
        monkeypatch.setattr(llm, "complete_json", recorder)
        monkeypatch.setattr(why, "recommendation_evidence",
                            lambda *_: evidence if evidence is not None else EVIDENCE)
        rows = [] if movie == "no user" else [movie or MOVIE]
        client = TestClient(create_app(driver=PairDriver(rows)))
        return client, recorder

    return build


def test_paths_and_checked_summary_are_returned_separately(setup):
    """Verified paths and the AI summary come back side by side."""
    client, recorder = setup()
    body = client.get("/api/users/1/recommendations/589/why").json()
    assert body["paths"] == EVIDENCE
    assert body["hasEvidence"] is True
    assert body["summary"] == "You and user 414 both liked 130 movies."
    assert body["summarySource"] == "llm"
    assert body["summaryCheck"] == {"verified": True, "unsupported": []}
    assert body["note"] is None
    assert recorder.calls == 1


def test_no_evidence_says_so_and_never_calls_the_llm(setup):
    """Empty paths: no summary, an explicit note, and zero LLM calls."""
    client, recorder = setup(evidence={"peers": [], "sharedGenres": []})
    body = client.get("/api/users/1/recommendations/128/why").json()
    assert body["hasEvidence"] is False
    assert body["summary"] is None
    assert body["summarySource"] is None
    assert body["note"] == why.NO_EVIDENCE_NOTE
    assert recorder.calls == 0


def test_without_a_key_the_paths_still_answer(setup, monkeypatch):
    """No LLM: 200 with the paths, a null summary and a note (not a 503)."""
    client, _ = setup()
    monkeypatch.setattr(llm, "complete_json", real_complete_json)
    monkeypatch.delenv("LLM_API_KEY")
    response = client.get("/api/users/1/recommendations/589/why")
    assert response.status_code == 200
    body = response.json()
    assert body["paths"] == EVIDENCE
    assert body["summary"] is None
    assert "No LLM configured" in body["note"]


def test_an_llm_failure_keeps_the_paths(setup, monkeypatch):
    """A provider error drops only the summary."""
    client, _ = setup()

    def failing(*_):
        raise llm.LLMError("LLM provider error: RateLimitError")
    monkeypatch.setattr(llm, "complete_json", failing)
    body = client.get("/api/users/1/recommendations/589/why").json()
    assert body["paths"] == EVIDENCE
    assert body["summary"] is None
    assert "RateLimitError" in body["note"]


def test_an_already_rated_movie_is_flagged(setup):
    """The user's own rating is reported, with a note."""
    client, _ = setup(movie={**MOVIE, "rating": 4.5})
    body = client.get("/api/users/1/recommendations/589/why").json()
    assert body["alreadyRated"] == 4.5
    assert "already rated" in body["note"]


@pytest.mark.parametrize(
    ("movie", "detail"),
    [("no user", "user 1 not found"), ({**MOVIE, "title": None}, "movie 589 not found")],
)
def test_unknown_ids_are_404(setup, movie, detail):
    """Unknown user or movie answers 404 naming the id."""
    client, _ = setup(movie=movie)
    response = client.get("/api/users/1/recommendations/589/why")
    assert response.status_code == 404
    assert response.json() == {"detail": detail}


def test_non_integer_ids_are_422(setup):
    """Ids are validated before any query."""
    client, _ = setup()
    assert client.get("/api/users/x/recommendations/1/why").status_code == 422
    assert client.get("/api/users/1/recommendations/y/why").status_code == 422


@pytest.mark.parametrize(
    ("summary", "unsupported"),
    [
        ("You and user 414 both liked 130 movies; you liked 70 comedies.", []),
        ("Users 414 and 380 share your taste; 2 peers in total.", []),
        ("Terminator 2 suits you: user 1 likes Comedy.", []),
        ("You and user 414 both liked 250 movies.", ["250"]),
        ("Over 95% of fans of Action movies loved it.", ["95", "Action"]),
        ("A Sci-Fi classic for Drama lovers.", ["Drama", "Sci-Fi"]),
        ("Star Wars fans will like it, user 414 says.", []),
    ],
)
def test_check_summary_flags_numbers_and_genres_not_in_the_paths(summary, unsupported):
    """Every number and genre stated must come from the evidence."""
    result = why.check_summary(summary, EVIDENCE, 1, MOVIE)
    assert result == {"verified": not unsupported, "unsupported": unsupported}


def test_an_unverified_summary_is_returned_flagged(setup):
    """A summary inventing a number is kept but marked unverified."""
    client, _ = setup(summary="You and user 414 both liked 999 movies.")
    body = client.get("/api/users/1/recommendations/589/why").json()
    assert body["summaryCheck"] == {"verified": False, "unsupported": ["999"]}
    assert body["summary"] == "You and user 414 both liked 999 movies."
    assert body["note"].startswith("UNVERIFIED")
    assert "999" in body["note"]


def test_the_llm_only_sees_the_evidence(setup, monkeypatch):
    """The prompt carries the evidence JSON and the title, nothing else."""
    seen = []

    def capture(messages, _schema):
        seen.extend(messages)
        return WhySummary(summary="ok")
    client, _ = setup()
    monkeypatch.setattr(llm, "complete_json", capture)
    client.get("/api/users/1/recommendations/589/why")
    assert seen[0]["role"] == "system"
    assert "ONLY the facts" in seen[0]["content"]
    assert '"peerId": 414' in seen[1]["content"]
    assert "Terminator 2: Judgment Day" in seen[1]["content"]
