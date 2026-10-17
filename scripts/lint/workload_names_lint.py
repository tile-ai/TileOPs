#!/usr/bin/env python3
"""Lint benchmark files for parametrize cases that carry no id.

A case id is the workload's name wherever it is read later, and pytest's
positional fallback (``shape0-dtype0``) is also positional in the perf history:
inserting a case renumbers the ones after it, so their history silently starts
over.

Flags a ``pytest.param(...)`` without ``id=`` and a bare tuple in a parametrize
value list, one finding per case.

Usage: ``workload_names_lint.py [FILE ...]``. With no arguments, scans
``benchmarks/ops/``. Exits 1 on an unnamed case.
"""

import ast
import sys
from pathlib import Path

from common import files, relative, report

SCANNED = ("benchmarks/ops",)


def _case_lists(tree: ast.Module):
    """The value lists pytest turns into cases: the second argument of a
    ``parametrize(...)`` call, and the ``("a, b", [...])`` entry a fixture base
    expands into one."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "parametrize"
            and len(node.args) > 1
            and isinstance(node.args[1], ast.List)
        ):
            yield node.args[1]
        elif (
            isinstance(node, ast.Tuple)
            and len(node.elts) == 2
            and isinstance(node.elts[0], ast.Constant)
            and isinstance(node.elts[0].value, str)
            and isinstance(node.elts[1], ast.List)
        ):
            yield node.elts[1]


def unnamed_cases(source: str) -> list[int]:
    """Line numbers of cases with no id.

    A `pytest.param(...)` without `id=` counts wherever it is written, including
    inside a helper that returns the list: scoping this to literal lists would
    miss every file that builds its cases in a function. A bare tuple counts
    only inside a parametrize value list, where it is a case rather than a
    tuple.
    """
    tree = ast.parse(source)
    lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "param"
        and not any(kw.arg == "id" for kw in node.keywords)
    ]
    for case_list in _case_lists(tree):
        lines += [e.lineno for e in case_list.elts if isinstance(e, (ast.Tuple, ast.List))]
    return sorted(lines)


def findings(path: Path) -> list[str]:
    """One finding per case in *path* that has no id; none outside ``benchmarks/ops/``."""
    rel = relative(path)
    if not rel.startswith("benchmarks/ops/"):
        return []
    return [
        f"{rel}:{line}: parametrize case has no id — give it one naming the scenario it stands for"
        for line in unnamed_cases(path.read_text(encoding="utf-8"))
    ]


def main(argv: list[str]) -> int:
    return report(e for p in files(argv, SCANNED, (".py",)) for e in findings(p))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
