"""Download and extract MovieLens datasets into the local data directory.

The datasets are never committed: they are fetched on demand from GroupLens.
Re-running the script is safe; an already extracted dataset is skipped.

Usage:
    uv run python scripts/download_data.py                 # ml-latest-small
    uv run python scripts/download_data.py --dataset ml-25m
"""

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://files.grouplens.org/datasets/movielens"
DATASETS = ("ml-latest-small", "ml-25m")
REQUIRED_FILES = ("movies.csv", "ratings.csv", "tags.csv", "links.csv")


def dataset_url(name: str) -> str:
    """Return the GroupLens download URL of a dataset."""
    return f"{BASE_URL}/{name}.zip"


def is_extracted(target: Path) -> bool:
    """Tell whether every CSV file the seed script needs is already present."""
    return all((target / filename).is_file() for filename in REQUIRED_FILES)


def download(url: str, destination: Path) -> None:
    """Stream a URL to a file, writing to a temporary name first."""
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out)
    partial.rename(destination)


def fetch_dataset(name: str, data_dir: Path) -> Path:
    """Download and extract one dataset, returning the directory holding its CSVs."""
    target = data_dir / name
    if is_extracted(target):
        print(f"{name}: already present in {target}, skipping")
        return target

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / f"{name}.zip"
    if not archive.is_file():
        print(f"{name}: downloading {dataset_url(name)}")
        download(dataset_url(name), archive)

    print(f"{name}: extracting {archive}")
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(data_dir)
    archive.unlink()

    if not is_extracted(target):
        raise RuntimeError(f"{name}: archive did not contain {', '.join(REQUIRED_FILES)}")
    return target


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=DATASETS, default="ml-latest-small")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", "data")))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    path = fetch_dataset(args.dataset, args.data_dir)
    print(f"{args.dataset}: ready in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
