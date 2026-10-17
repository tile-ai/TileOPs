#!/usr/bin/env python3
"""Lint comments and docstrings for narration of how the code came to be.

A comment states what the code promises and what its input must satisfy. How it got there
(what the author attempted or which version came before) is history: it is true of the pull
request, not of the code, and it goes stale with the next edit. ``_NARRATION`` contains only
high-confidence forms of that history; words such as "measured", "step" and "phase" also
have stable domain meanings and are not findings on their own. Where a threshold or config
was measured ("Measured on H200: ...") is the reason ``.claude/rules/code-style.md`` asks to
sit beside a tuned value, so it is a contract, not history.

Only ``#`` comments and docstrings are read, so an identifier such as ``phase`` in barrier
code or ``measured_ms`` is never a finding. A line that must state history carries
``lint: narration allow: <reason>`` on that same line.

Usage: ``comment_narration_lint.py [FILE ...]``. With no files, scans ``src/tileops/``,
``tests/``, ``benchmarks/`` and ``workloads/``. Exits 1 on a finding.
"""

import argparse
import re
import sys
from pathlib import Path

from comment_lines import allowed_lines, prose_lines
from common import files, relative, report

SCANNED = ("src/tileops", "tests", "benchmarks", "workloads")

_NARRATION = re.compile(
    r"\b(?:we|i)\s+(?:have\s+)?(?:tried|measured)\b"
    r"|^\s*(?:\#\s*|['\"]{3}\s*)?tried\b"
    r"|\b(?:this|that|it|we)\s+used to\b"
    r"|\b(?:as before|earlier version|previous version|after measuring)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _narration(text: str) -> list[str]:
    return [match.group(0) for match in _NARRATION.finditer(text)]


def findings(path: Path) -> list[str]:
    """Narration in *path*'s comments and docstrings."""
    source = path.read_text(encoding="utf-8")
    allowed = allowed_lines(source, "narration")
    out = []
    for prose in prose_lines(source):
        if prose.line in allowed:
            continue
        for phrase in _narration(prose.text):
            out.append(
                f"{relative(path)}:{prose.line}: {prose.kind} narrates history ({phrase!r}) — state what "
                "the code promises and what its input must satisfy, or add "
                '"# lint: narration allow: <reason>" on the line'
            )
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)
    return report(e for p in files(args.files, SCANNED, (".py",)) for e in findings(p))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
