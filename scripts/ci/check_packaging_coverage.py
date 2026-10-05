#!/usr/bin/env python3
"""Fail unless every `tileops._FAMILIES` member has a collected ``packaging`` case.

    python scripts/ci/check_packaging_coverage.py

Collection only, over the selection the installed-wheel run tests: no kernel is built and
no GPU is needed.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _Families:
    """Records the family each collected ``packaging`` case names."""

    def __init__(self) -> None:
        self.covered: dict[str, int] = {}
        self.errors: list[str] = []

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        import tileops

        for item in session.items:
            for mark in item.iter_markers("packaging"):
                family = mark.kwargs.get("family")
                if family in tileops._FAMILIES:
                    self.covered[family] = self.covered.get(family, 0) + 1
                else:
                    self.errors.append(f"{item.nodeid}: {family!r} is not a tileops family")


def main() -> int:
    os.chdir(REPO)
    recorder = _Families()
    pytest.main(
        ["--collect-only", "-q", "-p", "no:cacheprovider", "-m", "packaging", "tests/ops"],
        plugins=[recorder],
    )

    import tileops

    errors = recorder.errors + [
        f"family {family!r} has no packaging case"
        for family in tileops._FAMILIES
        if family not in recorder.covered
    ]
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    for family in tileops._FAMILIES:
        print(f"{family}: {recorder.covered[family]} packaging case(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
