#!/usr/bin/env python3
"""Lint benchmark files for parametrize cases that carry no id.

A case id is the workload's name wherever it is read later, and pytest's
positional fallback (``shape0-dtype0``) is also positional in the perf history:
inserting a case renumbers the ones after it, so their history silently starts
over.

Flags a ``pytest.param(...)`` without ``id=`` and a bare tuple in a parametrize
value list. Files that predate the rule are exempt by name; a file whose cases
have all been named must leave the list, which is what keeps the list from
outliving the work.

Usage: ``workload_names_lint.py [FILE ...]``. With no arguments, scans
``benchmarks/ops/``. Exits 1 on an unnamed case in a file that is not exempt,
and on an exempt file that no longer needs to be.
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmarks" / "ops"

# Files written before the rule. Name their cases and delete the line.
EXEMPT = {
    "benchmarks/ops/bench_binary_elementwise.py",
    "benchmarks/ops/bench_moe_shared_fused_moe.py",
}


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


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv] or sorted(BENCH_DIR.rglob("*.py"))
    failed = False
    for path in paths:
        try:
            rel = path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel = path.as_posix()
        if not rel.startswith("benchmarks/ops/"):
            continue
        found = unnamed_cases(path.read_text(encoding="utf-8"))
        if rel in EXEMPT:
            if not found:
                failed = True
                print(
                    f"{rel}: every case is named now — drop it from EXEMPT in "
                    f"{Path(__file__).name}.",
                    file=sys.stderr,
                )
            continue
        if found:
            failed = True
            where = ", ".join(str(n) for n in found[:5])
            print(
                f"{rel}: {len(found)} parametrize cases with no id; first at "
                f"line {where}. Give each an id naming the scenario it stands "
                f"for.",
                file=sys.stderr,
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
