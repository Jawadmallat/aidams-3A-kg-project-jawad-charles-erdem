"""Unit tests for the MovieLens parsing helpers, using real titles from the dataset."""

import pytest

from scripts.movielens import imdb_id, normalize_tag, optional_int, parse_genres, parse_title


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Toy Story (1995)", ("Toy Story", 1995)),
        ("Matrix, The (1999)", ("The Matrix", 1999)),
        ("Haine, La (1995)", ("La Haine", 1995)),
        ("Avventura, L' (1960)", ("L'Avventura", 1960)),
        (
            "City of Lost Children, The (Cité des enfants perdus, La) (1995)",
            ("The City of Lost Children (Cité des enfants perdus, La)", 1995),
        ),
        (
            "Good, the Bad and the Ugly, The (Buono, il brutto, il cattivo, Il) (1966)",
            ("The Good, the Bad and the Ugly (Buono, il brutto, il cattivo, Il)", 1966),
        ),
        (
            "Man on the Train (Homme du train, L') (2002)",
            ("Man on the Train (Homme du train, L')", 2002),
        ),
        ("Like Father, Like Son (1987)", ("Like Father, Like Son", 1987)),
        ("Death Note: Desu nôto (2006–2007)", ("Death Note: Desu nôto", 2006)),
        ("(500) Days of Summer (2009)", ("(500) Days of Summer", 2009)),
        ("Babylon 5", ("Babylon 5", None)),
        ("Paterson ", ("Paterson", None)),
        ("Nocturnal Animals  (2016) ", ("Nocturnal Animals", 2016)),
    ],
)
def test_parse_title(raw, expected):
    """Titles lose their year, regain their leading article and are trimmed."""
    assert parse_title(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Adventure|Animation|Children", ["Adventure", "Animation", "Children"]),
        ("Drama", ["Drama"]),
        ("(no genres listed)", []),
        ("Drama|Drama| Comedy ", ["Drama", "Comedy"]),
        ("IMAX|Sci-Fi", ["IMAX", "Sci-Fi"]),
    ],
)
def test_parse_genres(raw, expected):
    """The placeholder is dropped, duplicates removed, order kept."""
    assert parse_genres(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Dark Comedy", "dark comedy"),
        ("  twist   ending ", "twist ending"),
        ("Pixar", "pixar"),
        ("   ", None),
        (1984, "1984"),
    ],
)
def test_normalize_tag(raw, expected):
    """Tags are lower-cased and whitespace-normalised; blank tags become None."""
    assert normalize_tag(raw) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(862.0, 862), ("862", 862), (float("nan"), None), (None, None), ("", None)],
)
def test_optional_int(value, expected):
    """Missing CSV cells (NaN, empty, None) map to None."""
    assert optional_int(value) == expected


def test_imdb_id_is_zero_padded():
    """IMDb ids get the tt prefix and seven digits."""
    assert imdb_id(114709) == "tt0114709"
    assert imdb_id(float("nan")) is None
