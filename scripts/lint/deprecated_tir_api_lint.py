#!/usr/bin/env python3
"""Lint shipped source for deprecated TIR spellings.

Both forms below still parse, so nothing fails until a toolchain bump drops
them; a check is what keeps them from drifting back in. Flags, per line:

- The superseded parameter type — the one named after a buffer rather than a
  tensor. The supported spelling is ``T.Tensor(shape, dtype)``; the signature
  is identical, so correcting a site is a rename.
- ``T.reinterpret`` called with a string literal first, the deprecated
  dtype-first order. The supported spelling puts the value first:
  ``T.reinterpret(value, dtype)``.

Usage: ``deprecated_tir_api_lint.py [FILE ...]``. With no arguments, scans the
default shipped-source trees (``tileops/``, ``tests/``, ``benchmarks/``,
``scripts/``) excluding ``tileops/manifest/``. Exits 1 when any violation
is found.
"""

import argparse
import re
import sys
from pathlib import Path

# Assembled from parts so this file does not trip its own check.
_DEPRECATED_BUFFER_TYPE = "T." + "Buffer"

_PATTERNS = (
    (
        re.compile(rf"\b{re.escape(_DEPRECATED_BUFFER_TYPE)}\b"),
        f"deprecated {_DEPRECATED_BUFFER_TYPE}; use T.Tensor(shape, dtype)",
    ),
    (
        # A quote as the first argument means the dtype was passed first.
        re.compile(r"\bT\.reinterpret\(\s*[\"']"),
        "deprecated dtype-first T.reinterpret; use T.reinterpret(value, dtype)",
    ),
)

DEFAULT_ROOTS = ("tileops", "tests", "benchmarks", "scripts")
DEFAULT_EXCLUDE = "tileops/manifest"


def lint_text(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, message)`` pairs for every violation in ``text``."""
    findings = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern, message in _PATTERNS:
            if pattern.search(line):
                findings.append((lineno, message))
    return findings


def _default_files() -> list[Path]:
    files = []
    for root in DEFAULT_ROOTS:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for path in sorted(root_path.rglob("*")):
            if not path.is_file():
                continue
            if path.as_posix().startswith(DEFAULT_EXCLUDE + "/"):
                continue
            files.append(path)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)

    files = args.files or _default_files()
    exit_code = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable: not shipped text source
        for lineno, message in lint_text(text):
            print(f"{path}:{lineno}: {message}")
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
