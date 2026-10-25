"""The public surface matches the committed snapshot.

A change to a public name, signature, default or torch operator name fails here until the
snapshot is regenerated with `python -m tests.public_api` in the same change, so every
public-surface change is visible in review.
"""

import json

import pytest

from tests.public_api import SNAPSHOT, surface


@pytest.mark.smoke
def test_public_surface_matches_the_snapshot() -> None:
    expected = set(json.loads(SNAPSHOT.read_text()))
    actual = set(surface())
    assert (sorted(actual - expected), sorted(expected - actual)) == ([], []), (
        "public surface changed; regenerate with `python -m tests.public_api` if intended"
    )
