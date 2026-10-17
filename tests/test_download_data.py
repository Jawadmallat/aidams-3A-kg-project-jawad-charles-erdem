"""Unit tests for scripts/download_data.py (no network access)."""

import zipfile
from pathlib import Path

import pytest

from scripts import download_data


def make_archive(path: Path, name: str, files: tuple[str, ...]) -> None:
    """Build a fake MovieLens zip containing the given CSV files."""
    with zipfile.ZipFile(path, "w") as zipped:
        for filename in files:
            zipped.writestr(f"{name}/{filename}", "header\n")


def fake_download_from(source: Path):
    """Return a download() replacement that copies a local archive."""
    calls = []

    def fake_download(url: str, destination: Path) -> None:
        calls.append(url)
        destination.write_bytes(source.read_bytes())

    return fake_download, calls


def test_dataset_url_points_to_grouplens():
    """The URL follows the GroupLens naming scheme."""
    assert download_data.dataset_url("ml-25m") == (
        "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
    )


def test_is_extracted_requires_every_csv(tmp_path):
    """A directory missing one CSV does not count as extracted."""
    for filename in download_data.REQUIRED_FILES[:-1]:
        (tmp_path / filename).write_text("x")
    assert not download_data.is_extracted(tmp_path)
    (tmp_path / download_data.REQUIRED_FILES[-1]).write_text("x")
    assert download_data.is_extracted(tmp_path)


def test_fetch_extracts_then_skips_on_rerun(tmp_path, monkeypatch):
    """First run downloads and extracts; second run does not download again."""
    source = tmp_path / "source.zip"
    make_archive(source, "ml-latest-small", download_data.REQUIRED_FILES)
    fake_download, calls = fake_download_from(source)
    monkeypatch.setattr(download_data, "download", fake_download)
    data_dir = tmp_path / "data"

    target = download_data.fetch_dataset("ml-latest-small", data_dir)
    download_data.fetch_dataset("ml-latest-small", data_dir)

    assert download_data.is_extracted(target)
    assert len(calls) == 1
    assert not (data_dir / "ml-latest-small.zip").exists()


def test_fetch_rejects_incomplete_archive(tmp_path, monkeypatch):
    """An archive without the expected CSVs raises instead of half-succeeding."""
    source = tmp_path / "source.zip"
    make_archive(source, "ml-latest-small", ("movies.csv",))
    fake_download, _ = fake_download_from(source)
    monkeypatch.setattr(download_data, "download", fake_download)

    with pytest.raises(RuntimeError):
        download_data.fetch_dataset("ml-latest-small", tmp_path / "data")
