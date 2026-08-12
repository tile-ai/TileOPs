"""Structural gates for the benchmark stage's boundary.

Coverage is deliberately literal: these scan `benchmarks/ops` for a static
`tests` import and for a definition named exactly `gen_inputs`. They do not prove input construction is absent — a
module-level draw helper, or one injected into a workload as a callable, would
pass. Both forms existed here and were removed rather than gated for, because
naming every way to build a tensor is a losing game.

See docs/design/trust-model.md §Benchmark.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIRS = ("benchmarks/ops",)


def _benchmark_files() -> list[Path]:
    return [
        path
        for rel in BENCHMARK_DIRS
        for path in sorted((REPO_ROOT / rel).rglob("*.py"))
        if path.name != "__init__.py"
    ]


def _scan(finder) -> dict[str, list[str]]:
    offenders = {}
    for path in _benchmark_files():
        hits = finder(ast.parse(path.read_text(), filename=str(path)))
        if hits:
            offenders[str(path.relative_to(REPO_ROOT))] = hits
    return offenders


def _imports_tests(tree: ast.AST) -> list[str]:
    def is_tests(name: str) -> bool:
        return name == "tests" or name.startswith("tests.")

    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and is_tests(node.module or ""):
            hits.append(f"from {node.module} (line {node.lineno})")
        elif isinstance(node, ast.Import):
            hits += [
                f"import {a.name} (line {node.lineno})"
                for a in node.names if is_tests(a.name)
            ]
    return hits


def _defines_gen_inputs(tree: ast.AST) -> list[str]:
    return [
        f"line {n.lineno}"
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "gen_inputs"
    ]


@pytest.mark.smoke
def test_benchmarks_do_not_import_tests_package() -> None:
    """A benchmark that imports tests/ breaks on any test-side refactor, and no
    PR gate runs the nightly benchmarks that would catch it."""
    assert _scan(_imports_tests) == {}


@pytest.mark.smoke
def test_benchmarks_do_not_author_gen_inputs() -> None:
    """Import the op's workload from workloads/; if it has none, add it there."""
    assert _scan(_defines_gen_inputs) == {}
