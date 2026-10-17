"""Pure parsing helpers that turn raw MovieLens CSV values into graph properties.

Kept free of I/O and Neo4j so every cleaning rule can be unit tested on its own.
"""

import math
import re

NO_GENRES = "(no genres listed)"

# Articles MovieLens moves to the end of a title ("Matrix, The"). Case-sensitive on
# purpose: "Good, the Bad and the Ugly, The" must only move the final "The".
ARTICLES = (
    "The", "A", "An",
    "Le", "La", "Les", "L'", "El", "Il", "Das", "Der", "Die", "Los", "Las",
)

# Trailing "(1999)" or, for series, "(2006–2007)" with an en dash or a hyphen.
_YEAR_RE = re.compile(r"\s*\((\d{4})(?:[–-]\d{4})?\)\s*$")

# "<main>, <article>" optionally followed by an alternate title in parentheses,
# e.g. "City of Lost Children, The (Cité des enfants perdus, La)".
_ARTICLE_RE = re.compile(
    r"^(?P<main>.+?), (?P<article>" + "|".join(re.escape(a) for a in ARTICLES) + r")"
    r"(?P<alternate> \(.+\))?$"
)


def _collapse_spaces(text: str) -> str:
    """Strip a string and collapse inner runs of whitespace to single spaces."""
    return " ".join(text.split())


def parse_title(raw: str) -> tuple[str, int | None]:
    """Split a MovieLens title into a display title and a release year.

    "Matrix, The (1999)" -> ("The Matrix", 1999). The year is None when the title
    carries none; for a range such as "(2006–2007)" the first year is kept.
    """
    text = _collapse_spaces(raw)
    year = None
    match = _YEAR_RE.search(text)
    if match:
        year = int(match.group(1))
        text = text[: match.start()].strip()

    match = _ARTICLE_RE.match(text)
    if match:
        article = match.group("article")
        separator = "" if article.endswith("'") else " "
        text = f"{article}{separator}{match.group('main')}{match.group('alternate') or ''}"

    return (text or _collapse_spaces(raw)), year


def parse_genres(raw: str) -> list[str]:
    """Split a pipe-separated genre string, dropping the "no genres" placeholder.

    Order is preserved and duplicates are removed.
    """
    genres = []
    for genre in raw.split("|"):
        genre = genre.strip()
        if genre and genre != NO_GENRES and genre not in genres:
            genres.append(genre)
    return genres


def normalize_tag(raw: str) -> str | None:
    """Lower-case a free-text tag and collapse its whitespace; None if it is empty."""
    tag = _collapse_spaces(str(raw)).lower()
    return tag or None


def optional_int(value) -> int | None:
    """Convert a CSV cell to int, mapping empty cells and NaN to None."""
    if value is None or value == "":
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return int(value)


def imdb_id(value) -> str | None:
    """Format a numeric IMDb id the way IMDb does: 114709 -> "tt0114709"."""
    number = optional_int(value)
    return None if number is None else f"tt{number:07d}"
