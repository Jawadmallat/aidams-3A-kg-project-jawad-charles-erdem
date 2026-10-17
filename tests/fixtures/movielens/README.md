# MovieLens test fixture

A subset of [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/)
built by `scripts/make_fixture.py`: the 20 most active users, the 50
most rated movies, and the ratings and tags between them.

Regenerate it with:

    uv run python scripts/download_data.py
    uv run python -m scripts.make_fixture

## Licence

This subset is redistributed under the MovieLens usage licence: research use
only, no endorsement implied by the University of Minnesota or GroupLens, and
no commercial use without permission. Citation:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
> and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4:
> 19:1–19:19. https://doi.org/10.1145/2827872
