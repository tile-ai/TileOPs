#!/usr/bin/env python3
"""Lint ``tests/ops/`` for tests that construct a kernel instead of calling its op.

A test under ``tests/ops/`` guards what an op call promises, and the op's dispatch is the
only path a caller has to a kernel (``docs/design/testing.md`` § Tests). A call whose callee
is named ``...Kernel`` constructs a kernel outside that path, wherever in the file it sits:
a test, a helper, or module scope. A kernel class named without being called (a base class,
an ``isinstance`` check, a value handed to the op) constructs nothing and is not a finding.

A kernel test belongs where ``testing.md`` puts it. An exceptional line carries
``lint: test-placement allow: <reason>`` on that same line.

Usage: ``test_placement_lint.py [--base REF] [FILE ...]``. With ``--base``, only a
construction on a line added since the merge base of REF and HEAD is reported
(``lint_diff.py``); the hook passes ``--base auto`` to find the TileOPs upstream,
overridable with ``TILEOPS_LINT_BASE``. Without it, every construction is. With no files,
scans ``tests/ops/``. Exits 1 on a finding and 2 when the base has no merge base with HEAD.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

from comment_lines import allowed_lines
from lint_diff import BASE_ENV, NoMergeBaseError, added_lines, relative, resolve_base

REPO_ROOT = Path(__file__).resolve().parents[2]

_KERNEL = re.compile(r"\w*Kernel")


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def findings(path: Path, added: set[int] | None = None) -> list[str]:
    """Kernel constructions in *path* on the lines in *added* (None: all)."""
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
        if (
            callee
            and _KERNEL.fullmatch(callee)
            and node.lineno not in allowed
            and (added is None or node.lineno in added)
        ):
            where = owner.get(id(node), "<module>")
            out.append(
                (
                    node.lineno,
                    f"{path}:{node.lineno}: {where} constructs {callee} directly — a test under "
                    "tests/ops/ reaches the kernel through the op, or add "
                    '"# lint: test-placement allow: <reason>" on the line',
                )
            )
    return [message for _, message in sorted(out)]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="report only the lines added since the merge base")
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args(argv)
    paths = [p for p in args.files if p.suffix == ".py"] or sorted(
        (REPO_ROOT / "tests" / "ops").rglob("*.py")
    )
    added: dict[str, set[int] | None] = {}
    if args.base:
        try:
            base = resolve_base(args.base, REPO_ROOT)
            if base is None:
                print(
                    "test_placement_lint: skipping diff-aware check: no fetched "
                    f"tile-ai/TileOPs main remote; set {BASE_ENV} to a base ref",
                    file=sys.stderr,
                )
                return 0
            added = added_lines(paths, base, REPO_ROOT)
        except NoMergeBaseError as exc:
            print(f"test_placement_lint: {exc}", file=sys.stderr)
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
