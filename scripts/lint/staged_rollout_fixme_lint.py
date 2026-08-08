#!/usr/bin/env python3
"""Lint ``FIXME(staged-rollout)`` markers for the required block format.

Every marker must open a comment block of the shape:

    # FIXME(staged-rollout): <one-line summary>
    #
    # Broken invariant: <what contract is currently violated>
    # Why: <which process constraint requires this temporary state>
    # Cleanup: <concrete condition that triggers removal of this marker>

Sections must appear in that order inside the contiguous comment block, and
the ``Cleanup:`` text must name an invariant to restore — never a PR number.

Usage: ``staged_rollout_fixme_lint.py [FILE ...]``. With no arguments, scans
the default shipped-source trees (``tileops/``, ``tests/``, ``benchmarks/``,
``scripts/``). Exits 1 when any marker violates the format.
"""

import argparse
import re
import sys
from pathlib import Path

_MARKER = re.compile(r"FIXME\(staged-rollout\)(?P<colon>:?)\s*(?P<summary>.*)")
_COMMENT = re.compile(r"^\s*#(?P<body>.*)$")
_SECTIONS = ("Broken invariant:", "Why:", "Cleanup:")
_PR_NUMBER = re.compile(r"#[0-9]+")

DEFAULT_ROOTS = ("tileops", "tests", "benchmarks", "scripts")


def _comment_block(lines: list[str], start: int) -> list[str]:
    """Collect comment bodies from ``start`` through the end of the block."""
    block = []
    for line in lines[start:]:
        match = _COMMENT.match(line)
        if match is None:
            break
        block.append(match.group("body").strip())
    return block


def lint_text(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, message)`` pairs for every malformed marker."""
    findings = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        comment = _COMMENT.match(line)
        if comment is None:
            continue  # the convention block lives in comments only
        marker = _MARKER.search(comment.group("body"))
        if marker is None:
            continue
        lineno = index + 1
        if marker.group("colon") != ":" or not marker.group("summary").strip():
            findings.append((lineno, "marker must read 'FIXME(staged-rollout): <summary>'"))
            continue

        block = _comment_block(lines, index)
        cursor = 1  # skip the marker line itself
        for section in _SECTIONS:
            found_at = None
            for offset in range(cursor, len(block)):
                if block[offset].startswith(section):
                    found_at = offset
                    break
            if found_at is None:
                findings.append(
                    (lineno, f"block is missing '{section}' (sections must appear in order)")
                )
                break
            if not block[found_at][len(section):].strip():
                findings.append((lineno, f"'{section}' line has no text"))
                break
            cursor = found_at + 1
        else:
            cleanup_start = cursor - 1
            cleanup_text = " ".join(block[cleanup_start:])
            if _PR_NUMBER.search(cleanup_text):
                findings.append(
                    (lineno, "'Cleanup:' must name the invariant to restore, not a PR number")
                )
    return findings


def _default_files() -> list[Path]:
    files = []
    for root in DEFAULT_ROOTS:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        files.extend(p for p in sorted(root_path.rglob("*")) if p.is_file())
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
