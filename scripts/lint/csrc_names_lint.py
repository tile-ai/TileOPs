#!/usr/bin/env python3
"""Lint the names the C++/CUDA headers under ``src/tileops/csrc/`` define.

A kernel calls a csrc function as ``tileops::<name>``, so a reader can tell it from a TileLang
API, which is ``tl::<name>``, and a TileLang release cannot add a name that collides with it.
A header therefore declares everything inside ``namespace tileops { ... }``: nothing else at
file scope but preprocessor lines, no ``namespace tl`` anywhere, and no ``tileops_`` or
``__tl_`` prefix, which the namespace makes redundant. A macro has no namespace, so its name
carries the ``TILEOPS_`` prefix instead.

Usage: ``csrc_names_lint.py [FILE ...]``. With no arguments, scans ``src/tileops/csrc/``.
Exits 1 on a finding.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCANNED = "src/tileops/csrc"
SUFFIXES = (".h", ".cuh", ".hpp")

_TOKEN = re.compile(
    r"""
    (?P<line_comment>//[^\n]*)
  | (?P<block_comment>/\*.*?\*/)
  | (?P<string>"(?:\\.|[^"\\\n])*")
  | (?P<char>'(?:\\.|[^'\\\n])*')
  | (?P<directive>^[ \t]*\#(?:\\\n|[^\n])*)
    """,
    re.DOTALL | re.MULTILINE | re.VERBOSE,
)
_DEFINE = re.compile(r"#\s*define\s+(\w+)")
_PREFIXED = re.compile(r"\b(?:tileops_|__tl_)\w*")
_NAMESPACE_TL = re.compile(r"\bnamespace\s+tl\b")
_OPEN_TILEOPS = re.compile(r"namespace\s+tileops\s*\{")


def _blank(text: str) -> str:
    """Spaces in place of *text*, keeping its newlines so line numbers survive."""
    return re.sub(r"[^\n]", " ", text)


def findings(path: Path) -> list[str]:
    """Return one ``path:line: message`` per violation in the header at *path*."""
    source = path.read_text(encoding="utf-8")
    out = []

    def report(offset: int, message: str) -> None:
        out.append(f"{path}:{source.count(chr(10), 0, offset) + 1}: {message}")

    code_parts = []
    last = 0
    for match in _TOKEN.finditer(source):
        code_parts.append(source[last : match.start()])
        if match.lastgroup == "directive":
            define = _DEFINE.match(match.group().strip())
            if define and not define.group(1).startswith("TILEOPS_"):
                report(match.start(), f"macro {define.group(1)} lacks the TILEOPS_ prefix")
        code_parts.append(_blank(match.group()))
        last = match.end()
    code_parts.append(source[last:])
    code = "".join(code_parts)

    # A macro body expands into code, so it is held to the same names as code.
    directives = "".join(
        m.group() if m.lastgroup == "directive" else _blank(m.group())
        for m in _TOKEN.finditer(source)
    )
    for text in (code, directives):
        for match in _PREFIXED.finditer(text):
            report(match.start(), f"{match.group()} carries a prefix namespace tileops replaces")
        for match in _NAMESPACE_TL.finditer(text):
            report(match.start(), "namespace tl belongs to TileLang; declare in namespace tileops")

    # File scope holds nothing but `namespace tileops { ... }` blocks.
    pos, depth = 0, 0
    while pos < len(code):
        if depth == 0:
            opened = _OPEN_TILEOPS.match(code, pos)
            if opened:
                pos, depth = opened.end(), 1
                continue
            if not code[pos].isspace():
                report(pos, "declaration outside namespace tileops")
                end = code.find(";", pos)
                brace = code.find("{", pos)
                if brace != -1 and (end == -1 or brace < end):
                    pos, depth = brace + 1, 1
                else:
                    pos = len(code) if end == -1 else end + 1
                continue
        elif code[pos] == "{":
            depth += 1
        elif code[pos] == "}":
            depth -= 1
        pos += 1
    return out


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or sorted(
        p for p in (REPO_ROOT / SCANNED).rglob("*") if p.suffix in SUFFIXES
    )
    problems = [line for path in paths for line in findings(path)]
    for line in problems:
        print(line)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
