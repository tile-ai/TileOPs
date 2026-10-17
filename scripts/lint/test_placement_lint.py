#!/usr/bin/env python3
"""Lint ``tests/ops/`` for tests that construct a kernel instead of calling its op.

A test under ``tests/ops/`` guards what an op call promises, and the op's dispatch is the
only path a caller has to a kernel (``docs/design/testing.md`` § Tests). A call whose callee
is named ``...Kernel`` constructs a kernel outside that path, wherever in the file it sits:
a test, a helper, or module scope. A kernel class named without being called (a base class,
an ``isinstance`` check, a value handed to the op) constructs nothing and is not a finding.

A kernel test belongs where ``testing.md`` puts it. An exceptional line carries
``lint: test-placement allow: <reason>`` on that same line.

Usage: ``test_placement_lint.py [FILE ...]``. With no files, scans ``tests/ops/``. Exits 1
on a finding.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

from comment_lines import allowed_lines
from common import files, relative, report

SCANNED = ("tests/ops",)

_KERNEL = re.compile(r"\w*Kernel")


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def findings(path: Path) -> list[str]:
    """Kernel constructions in *path*."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    allowed = allowed_lines(source, "test-placement")
    owner: dict[int, str] = {}
    for function in ast.walk(tree):
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(function):
                owner.setdefault(id(node), function.name)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = _name(node.func)
        if callee and _KERNEL.fullmatch(callee) and node.lineno not in allowed:
            where = owner.get(id(node), "<module>")
            out.append(
                (
                    node.lineno,
                    f"{relative(path)}:{node.lineno}: {where} constructs {callee} directly — a test under "
                    "tests/ops/ reaches the kernel through the op, or add "
                    '"# lint: test-placement allow: <reason>" on the line',
                )
            )
    return [message for _, message in sorted(out)]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)
    return report(e for p in files(args.files, SCANNED, (".py",)) for e in findings(p))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
