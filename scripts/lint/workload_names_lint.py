#!/usr/bin/env python3
"""Flag a benchmark parametrize case without an id.

pytest's positional id (``shape0-dtype0``) shifts when a case is inserted, and the perf
history keyed by it starts over. Flags a ``pytest.param(...)`` without ``id=`` anywhere and
a bare tuple or list in a parametrize value list.

Default scan: ``benchmarks/ops/``.
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

    A ``pytest.param(...)`` counts wherever it is written, so a helper that builds the
    list is read too; a bare tuple counts only inside a parametrize value list.
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
