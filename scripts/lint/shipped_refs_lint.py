#!/usr/bin/env python3
"""Lint shipped source for review-process references.

Shipped source (code, docstrings, comments) must not reference issue/PR
numbers, acceptance-criteria labels, review rounds, or follow-up markers;
those belong to the review process, not the artifact. Flags, per line:

- ``#`` followed by 3+ digits (an issue/PR-number reference), unless the
  token reads as a CSS hex color: ``#`` followed by exactly 3, 4, 6, or 8
  hex characters ending at a word boundary (``#fff``, ``#191a16``,
  ``#112233``).
- ``AC-<n>`` labels.
- ``round-<n> review`` phrases.
- ``Follow-up: #<n>`` markers.

Usage: ``shipped_refs_lint.py [FILE ...]``. With no arguments, scans the
default shipped-source trees (``tileops/``, ``tests/``, ``benchmarks/``,
``scripts/``) excluding ``tileops/manifest/``. Exits 1 when any violation
is found.
"""

import re
import sys

from _common import run

# A hash-number token, captured with its full hex extent so a CSS hex color
# (#fff, #1234, #191a16, #11223344) can be recognized and exempted.
_HASH_TOKEN = re.compile(r"(?<![0-9A-Za-z])#([0-9a-fA-F]+)\b")
_CSS_HEX_LENGTHS = frozenset({3, 4, 6, 8})

_PLAIN_PATTERNS = (
    ("AC label", re.compile(r"\bAC-[0-9]+\b")),
    ("review-round reference", re.compile(r"\bround-[0-9]+ review\b")),
    ("follow-up marker", re.compile(r"[Ff]ollow-up:\s*#[0-9]+")),
)



def _hash_number_violations(line: str) -> list[str]:
    """Return the issue/PR-number tokens on a line, excluding CSS hex colors."""
    violations = []
    for match in _HASH_TOKEN.finditer(line):
        token = match.group(1)
        if len(token) in _CSS_HEX_LENGTHS:
            continue  # reads as a CSS hex color
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


def main(argv: list[str] | None = None) -> int:
    return run(lint_text, __doc__, argv)


if __name__ == "__main__":
    sys.exit(main())
