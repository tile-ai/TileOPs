#!/usr/bin/env python3
"""Lint comments and docstrings for citations of code locations.

A comment that points at a line elsewhere (``<file>.py:<n>``, ``line <n>``,
``<file>.py#L<n>``) goes stale with the first edit above that line, and nothing reports
it. Which code enforces a stated invariant belongs in the pull request's report; the
comment names the invariant, and a symbol (``_check_tiles``, ``:func:`check_tiles```)
when it must point somewhere, since a symbol survives an edit.

A location is a source file name followed by ``:<n>``, ``#L<n>``, ``L<n>`` or
``line <n>``, or ``line <n>`` / ``lines <n>`` on its own. ``L2`` alone is a cache level,
and "cache line 128" is a size; neither is a location.

Usage: ``comment_code_ref_lint.py [FILE ...]``. With no arguments, scans ``src/tileops/``,
``tests/``, ``benchmarks/``, ``workloads/`` and ``scripts/``. Exits 1 on a finding.
"""

import argparse
import re
import sys
from pathlib import Path

from comment_lines import prose_lines

REPO_ROOT = Path(__file__).resolve().parents[2]

_LOCATIONS = (
    # A source file name, then the line in it.
    re.compile(
        r"[\w./-]*\w\.(?:py|pyi|cu|cuh|cc|cpp|cxx|c|h|hpp|inl|yaml|yml|toml|md|json|sh)"
        r"(?::\d+|[:#]?\s*L\d+\b|,?\s+lines?\s+\d+)",
        re.IGNORECASE,
    ),
    re.compile(r"(?<!cache )\blines?\s+\d+\b", re.IGNORECASE),
)


def findings(path: Path) -> list[str]:
    """Every code-location citation in *path*'s comments and docstrings."""
    out = []
    for prose in prose_lines(path.read_text(encoding="utf-8")):
        for pattern in _LOCATIONS:
            match = pattern.search(prose.text)
            if match:
                out.append(
                    f"{path}:{prose.line}: {prose.kind} cites a code location "
                    f"({match.group(0)!r}) — name the symbol, and map invariants to "
                    "their enforcers in the pull request"
                )
                break
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)
    trees = ("src/tileops", "tests", "benchmarks", "workloads", "scripts")
    paths = [p for p in args.files if p.suffix == ".py"] or sorted(
        p for tree in trees for p in (REPO_ROOT / tree).rglob("*.py")
    )
    errors = [e for p in paths for e in findings(p)]
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
