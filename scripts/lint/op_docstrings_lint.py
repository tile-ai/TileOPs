#!/usr/bin/env python3
"""Lint op classes for the docstrings the API reference is built from.

The docs site gives every rendered member an entry, so a missing docstring
publishes a heading and a signature with nothing under them. For each class under
``src/tileops/ops/`` that defines either method, this checks that ``__init__`` and
``forward`` have a docstring, that neither carries reStructuredText the site
renders as literal text, and that construction parameters sit on ``__init__``
rather than in the class docstring's ``Args:``. See docs/development.md.

Usage: ``op_docstrings_lint.py [FILE ...]``; with no arguments, scans the tree.
"""

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OPS_DIR = REPO_ROOT / "src" / "tileops" / "ops"

RST = re.compile(r"\.\. (?:math|note|warning)::|:(?:func|meth|class|mod):`|>>> ")
CLASS_ARGS = re.compile(r"^\s*Args:\s*$", re.M)


def check_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as exc:  # ruff reports these; do not double up
        return [f"{path}: cannot parse ({exc})"]

    try:
        rel = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        rel = path

    problems: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        methods = {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}
        if not ({"__init__", "forward"} & methods.keys()):
            continue

        for name in ("__init__", "forward"):
            method = methods.get(name)
            if method is None:
                continue
            doc = ast.get_docstring(method)
            if doc is None:
                problems.append(
                    f"{rel}:{method.lineno}: {node.name}.{name} has no docstring; "
                    f"the API page renders an empty entry for it"
                )
            elif RST.search(doc):
                problems.append(
                    f"{rel}:{method.lineno}: {node.name}.{name} carries "
                    f"reStructuredText; docstrings render as Markdown"
                )

        if "__init__" in methods and CLASS_ARGS.search(ast.get_docstring(node) or ""):
            problems.append(
                f"{rel}:{node.lineno}: {node.name} documents parameters in its class "
                f"docstring; they belong on __init__"
            )
    return problems


def main(argv: list[str]) -> int:
    files = [Path(a) for a in argv] or sorted(OPS_DIR.rglob("*.py"))
    problems = [
        problem
        for path in files
        if path.suffix == ".py" and path.exists()
        for problem in check_file(path)
    ]
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
