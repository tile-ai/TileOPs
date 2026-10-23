#!/usr/bin/env python3
"""Check and print the H200 CUDA Graph qualification-to-test mapping without a GPU."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tests.cuda_graph_contract import QUALIFIED_OPS  # noqa: E402
from tileops.manifest import load_manifest  # noqa: E402


class _Evidence:
    def __init__(self) -> None:
        self.covered: dict[str, list[str]] = {}
        self.errors: list[str] = []

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        manifest = load_manifest()
        for item in session.items:
            marks = list(item.iter_markers("cuda_graph"))
            if len(marks) != 1:
                self.errors.append(f"{item.nodeid}: expected one cuda_graph(op=...) marker")
                continue
            op = marks[0].kwargs.get("op")
            if op not in QUALIFIED_OPS or manifest.get(op, {}).get("status") != "implemented":
                self.errors.append(f"{item.nodeid}: {op!r} is not a qualified implemented op")
                continue
            if item.get_closest_marker("sm90") is None or item.get_closest_marker("xfail"):
                self.errors.append(f"{item.nodeid}: graph evidence must be sm90 and not xfail")
                continue
            self.covered.setdefault(op, []).append(item.nodeid)


def main() -> int:
    os.chdir(REPO)
    recorder = _Evidence()
    status = pytest.main(
        ["--collect-only", "-q", "-p", "no:cacheprovider", "-m", "cuda_graph", "tests/ops"],
        plugins=[recorder],
    )
    if status != pytest.ExitCode.OK:
        print(f"CUDA Graph collection failed: pytest exit code {int(status)}", file=sys.stderr)
        return int(status)
    errors = recorder.errors + [
        f"{op}: no CUDA Graph evidence" for op in sorted(QUALIFIED_OPS - recorder.covered.keys())
    ]
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    for op, nodes in sorted(recorder.covered.items()):
        print(f"{op}:\n  " + "\n  ".join(nodes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
