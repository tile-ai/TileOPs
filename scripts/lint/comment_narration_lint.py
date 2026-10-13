#!/usr/bin/env python3
"""Lint comments and docstrings for narration of how the code came to be.

A comment states what the code promises and what its input must satisfy. How it got there
(what was attempted, which version came before, which step or phase of the work a line
belongs to, which library it copies) is history: it is true of the pull request, not of the code,
and it goes stale with the next edit. ``_NARRATION`` lists the phrases that mark it,
matched case-insensitively as whole words.

Only ``#`` comments and docstrings are read, so an identifier such as ``phase`` in barrier
code or ``measured_ms`` is never a finding. ``_PASSIVE_USE`` leaves one form alone
because it is not history: the passive of "use" followed by "to", after "is", "are" or
"be", which names a purpose rather than a past habit.

Usage: ``comment_narration_lint.py [--base REF] [FILE ...]``. With ``--base``, only the
lines added since the merge base of REF and HEAD are read (``lint_diff.py``); the hook
passes ``--base origin/main``. Without it, every line is. With no files, scans
``src/tileops/``, ``tests/``, ``benchmarks/`` and ``workloads/``. Exits 1 on a finding and
2 when the base has no merge base with HEAD.
"""

import argparse
import re
import sys
from pathlib import Path

from comment_lines import prose_lines
from lint_diff import NoMergeBaseError, added_lines, relative

REPO_ROOT = Path(__file__).resolve().parents[2]

_NARRATION = re.compile(
    r"\b(?:tried|measured|step [0-9]|used to|as before|earlier version|previous version"
    r"|phase [0-9]|as (?:vllm|flashinfer|deepgemm|torch) does)\b",
    re.IGNORECASE,
)
_PASSIVE_USE = re.compile(r"\b(?:is|are|be)\s+$", re.IGNORECASE)


def _narration(text: str) -> list[str]:
    hits = []
    for match in _NARRATION.finditer(text):
        phrase = match.group(0)
        if phrase.lower() == "used to" and _PASSIVE_USE.search(text[: match.start()]):
            continue
        hits.append(phrase)
    return hits


def findings(path: Path, added: set[int] | None = None) -> list[str]:
    """Narration in *path*'s comments and docstrings, on the lines in *added* (None: all)."""
    out = []
    for prose in prose_lines(path.read_text(encoding="utf-8")):
        if added is not None and prose.line not in added:
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
            added = added_lines(paths, args.base, REPO_ROOT)
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
