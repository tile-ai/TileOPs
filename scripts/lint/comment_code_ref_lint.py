#!/usr/bin/env python3
"""Lint comments and docstrings for citations of code locations.

A comment that points at a line of a file (``<file>.py:<n>``, ``<file>.py#L<n>``,
``line <n> of <file>.py``) goes stale with the first edit above that line, and nothing
reports it. Which code enforces a stated invariant belongs in the pull request's report; the
comment names the invariant, and a symbol (``_check_tiles``, ``:func:`check_tiles```)
when it must point somewhere, since a symbol survives an edit. A line that must cite a
location carries ``lint: code-ref allow: <reason>`` on that same line.

A location names its file: a source file name followed by ``:<n>``, ``:L<n>``, ``#L<n>``
or ``line <n>``, or ``line <n>`` / ``lines <n>`` followed by ``of`` or ``in`` and a file
name. A line number with no file ("line 128", "lines 3-4") is not decidable as a citation,
since "a 64-byte line 128" is a size and "L2" a cache level, and is not a finding.

Usage: ``comment_code_ref_lint.py [FILE ...]``. With no arguments, scans ``src/tileops/``,
``tests/``, ``benchmarks/``, ``workloads/`` and ``scripts/``. Exits 1 on a finding.
"""

import argparse
import re
import sys
from pathlib import Path

from comment_lines import allowed_lines, prose_lines
from common import files, relative, report

SCANNED = ("src/tileops", "tests", "benchmarks", "workloads", "scripts")

_FILE = r"[\w./-]*\w\.(?:py|pyi|cu|cuh|cc|cpp|cxx|c|h|hpp|inl|yaml|yml|toml|md|json|sh)"
_LOCATIONS = (
    # A source file name, then the line in it.
    re.compile(_FILE + r"(?::L?\d+|#L\d+|,?\s+lines?\s+\d+)", re.IGNORECASE),
    # A line, then the source file it is in.
    re.compile(r"\blines?\s+\d+(?:\s*[-–]\s*\d+)?,?\s+(?:of|in)\s+" + _FILE + r"\b", re.IGNORECASE),
)


def findings(path: Path) -> list[str]:
    """Every code-location citation in *path*'s comments and docstrings."""
    source = path.read_text(encoding="utf-8")
    allowed = allowed_lines(source, "code-ref")
    out = []
    for prose in prose_lines(source):
        if prose.line in allowed:
            continue
        for pattern in _LOCATIONS:
            match = pattern.search(prose.text)
            if match:
                out.append(
                    f"{relative(path)}:{prose.line}: {prose.kind} cites a code location "
                    f"({match.group(0)!r}) — name the symbol, and map invariants to "
                    'their enforcers in the pull request, or add "# lint: code-ref allow: '
                    '<reason>" on the line'
                )
                break
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)
    return report(e for p in files(args.files, SCANNED, (".py",)) for e in findings(p))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
