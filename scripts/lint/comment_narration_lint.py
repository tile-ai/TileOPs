#!/usr/bin/env python3
"""Lint comments and docstrings for narration of how the code came to be.

A comment states what the code promises and what its input must satisfy. How it got there
(what the author attempted or which version came before) is history: it is true of the pull
request, not of the code, and it goes stale with the next edit. ``_NARRATION`` contains only
high-confidence forms of that history; words such as "measured", "step" and "phase" also
have stable domain meanings and are not findings on their own, while a sentence that starts
"measured on/at" records the run that produced a result rather than a code contract.

Only ``#`` comments and docstrings are read, so an identifier such as ``phase`` in barrier
code or ``measured_ms`` is never a finding. A line that must state history carries
``lint: narration allow: <reason>`` on that same line.

Usage: ``comment_narration_lint.py [--base REF] [FILE ...]``. With ``--base``, only the
lines added since the merge base of REF and HEAD are read (``lint_diff.py``); the hook
passes ``--base auto`` to find the TileOPs upstream, overridable with
``TILEOPS_LINT_BASE``. Without it, every line is. With no files, scans
``src/tileops/``, ``tests/``, ``benchmarks/`` and ``workloads/``. Exits 1 on a finding and
2 when the base has no merge base with HEAD.
"""

import argparse
import re
import sys
from pathlib import Path

from comment_lines import allowed_lines, prose_lines
from lint_diff import BASE_ENV, NoMergeBaseError, added_lines, relative, resolve_base

REPO_ROOT = Path(__file__).resolve().parents[2]

_NARRATION = re.compile(
    r"\b(?:we|i)\s+(?:have\s+)?(?:tried|measured)\b"
    r"|^\s*(?:\#\s*|['\"]{3}\s*)?tried\b"
    r"|\b(?:this|that|it|we)\s+used to\b"
    r"|(?:^\s*(?:\#\s*|['\"]{3}\s*)?|(?<=[.;:])\s*)measured\s+(?:on|at)\b"
    r"|\b(?:as before|earlier version|previous version|after measuring)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _narration(text: str) -> list[str]:
    return [match.group(0) for match in _NARRATION.finditer(text)]


def findings(path: Path, added: set[int] | None = None) -> list[str]:
    """Narration in *path*'s comments and docstrings, on the lines in *added* (None: all)."""
    source = path.read_text(encoding="utf-8")
    allowed = allowed_lines(source, "narration")
    out = []
    for prose in prose_lines(source):
        if prose.line in allowed or (added is not None and prose.line not in added):
            continue
        for phrase in _narration(prose.text):
            out.append(
                f"{path}:{prose.line}: {prose.kind} narrates history ({phrase!r}) — state what "
                "the code promises and what its input must satisfy"
            )
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="read only the lines added since the merge base")
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)
    trees = ("src/tileops", "tests", "benchmarks", "workloads")
    paths = [p for p in args.files if p.suffix == ".py"] or sorted(
        p for tree in trees for p in (REPO_ROOT / tree).rglob("*.py")
    )
    added: dict[str, set[int] | None] = {}
    if args.base:
        try:
            base = resolve_base(args.base, REPO_ROOT)
            if base is None:
                print(
                    "comment_narration_lint: skipping diff-aware check: no fetched "
                    f"tile-ai/TileOPs main remote; set {BASE_ENV} to a base ref",
                    file=sys.stderr,
                )
                return 0
            added = added_lines(paths, base, REPO_ROOT)
        except NoMergeBaseError as exc:
            print(f"comment_narration_lint: {exc}", file=sys.stderr)
            return 2
    errors = []
    for path in paths:
        lines = added.get(relative(path, REPO_ROOT)) if args.base else None
        if args.base and lines == set():
            continue
        errors += findings(path, lines)
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
