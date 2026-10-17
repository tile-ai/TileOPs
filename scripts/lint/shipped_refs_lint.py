#!/usr/bin/env python3
"""Flag review-process references in shipped source.

Flags, on any line of a UTF-8 text file: ``#`` followed by 3+ digits (an issue or PR number),
``AC-<n>``, ``round-<n> review`` and ``Follow-up: #<n>``. A ``#`` token of 3, 4, 6 or 8 hex
characters with at least one letter is a CSS color and passes; an all-digit token does not.

Default scan: ``src/tileops/``, ``tests/``, ``benchmarks/``, ``scripts/``, ``workloads/``.
"""

import argparse
import re
import sys
from pathlib import Path

from common import files, relative, report

# The whole hex token, so the color test reads all of it.
_HASH_TOKEN = re.compile(r"(?<![0-9A-Za-z])#([0-9a-fA-F]+)\b")
_CSS_HEX_LENGTHS = frozenset({3, 4, 6, 8})
_HEX_LETTER = re.compile(r"[a-fA-F]")

_PLAIN_PATTERNS = (
    ("AC label", re.compile(r"\bAC-[0-9]+\b")),
    ("review-round reference", re.compile(r"\bround-[0-9]+ review\b")),
    ("follow-up marker", re.compile(r"[Ff]ollow-up:\s*#[0-9]+")),
)

SCANNED = ("src/tileops", "tests", "benchmarks", "scripts", "workloads")


def _hash_number_violations(line: str) -> list[str]:
    """Return the issue/PR-number tokens on a line, excluding CSS hex colors."""
    violations = []
    for match in _HASH_TOKEN.finditer(line):
        token = match.group(1)
        # A hex letter is what marks a color; its length does not.
        if len(token) in _CSS_HEX_LENGTHS and _HEX_LETTER.search(token):
            continue
        if re.fullmatch(r"[0-9]{3,}", token):
            violations.append(match.group(0))
    return violations


def lint_text(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, message)`` pairs for every violation in ``text``."""
    findings = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for token in _hash_number_violations(line):
            findings.append((lineno, f"issue/PR-number reference {token!r}"))
        for label, pattern in _PLAIN_PATTERNS:
            for match in pattern.finditer(line):
                findings.append((lineno, f"{label} {match.group(0)!r}"))
    return findings


def findings(path: Path) -> list[str]:
    """Every violation in the file at *path*; none for a binary or unreadable file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []  # binary or unreadable: not shipped text source
    return [f"{relative(path)}:{lineno}: {message}" for lineno, message in lint_text(text)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)
    return report(e for p in files(args.files, SCANNED, None) for e in findings(p))


if __name__ == "__main__":
    sys.exit(main())
