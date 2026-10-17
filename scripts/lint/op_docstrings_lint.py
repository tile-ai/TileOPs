#!/usr/bin/env python3
"""Flag op docstrings the API reference cannot render.

For each class under ``src/tileops/ops/``: each ``__init__`` and ``forward`` it defines has
a docstring without the reStructuredText the site shows literally (``.. math::``,
``:func:`` roles, ``>>>``), and a class that defines ``__init__`` has no ``Args:`` in its
own docstring, since parameters belong on ``__init__`` (docs/development.md).

Default scan: ``src/tileops/ops/``.
"""

import ast
import re
import sys
from pathlib import Path

from common import files, relative, report

SCANNED = ("src/tileops/ops",)

RST = re.compile(r"\.\. (?:math|note|warning)::|:(?:func|meth|class|mod):`|>>> ")
CLASS_ARGS = re.compile(r"^\s*Args:\s*$", re.M)


def check_file(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as exc:  # ruff reports these; do not double up
        return [f"{path}: cannot parse ({exc})"]

    rel = relative(path)

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
    paths = files(argv, SCANNED, (".py",))
    return report(e for p in paths if p.exists() for e in check_file(p))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
