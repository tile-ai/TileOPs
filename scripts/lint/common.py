"""What every lint here shares: the repository root, the files a run reads, the
allow-marker spelling and the report.

A lint that takes files reads the ones named on its command line, or its default trees when
none is named. Each finding is one ``path:line: message`` line on stdout, and a run with any
finding exits 1. A lint that honours an allow marker spells it
``lint: <rule> allow: <reason>``; a marker without a reason exempts nothing.
"""

import re
import sys
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def use_source_tree() -> None:
    """Import ``tileops`` from ``src/``: a hook runs where the package is not installed."""
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))


def allow_marker(rule: str) -> re.Pattern[str]:
    """The ``lint: <rule> allow: <reason>`` marker, with a reason, for *rule*."""
    return re.compile(rf"\blint:\s*{re.escape(rule)}\s+allow:\s*\w", re.IGNORECASE)


def files(
    named: Iterable[Path | str], trees: Iterable[str], suffixes: Iterable[str] | None
) -> list[Path]:
    """The *named* files, or every file under *trees* when none is named.

    *trees* are relative to the repository root. Only files with one of *suffixes* are kept;
    None keeps every file.
    """

    def kept(path: Path) -> bool:
        return suffixes is None or path.suffix in suffixes

    named = [Path(p) for p in named]
    suffixes = None if suffixes is None else tuple(suffixes)
    if named:
        return [p for p in named if kept(p)]
    return sorted(
        p
        for tree in trees
        for p in (REPO_ROOT / tree).rglob("*")
        if kept(p) and p.is_file() and "__pycache__" not in p.parts
    )


def relative(path: Path) -> str:
    """*path* relative to the repository root when it is under it, as git prints it."""
    resolved = path.resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return resolved.relative_to(REPO_ROOT).as_posix()
    return path.as_posix()


def report(findings: Iterable[str]) -> int:
    """Print each finding on its own line and return the exit code: 1 if any, else 0."""
    found = False
    for finding in findings:
        print(finding)
        found = True
    return 1 if found else 0
