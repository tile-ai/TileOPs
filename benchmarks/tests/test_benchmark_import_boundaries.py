"""Structural gates for the benchmark stage's boundary.

See docs/design/trust-model.md §Benchmark.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIRS = (
    REPO_ROOT / "benchmarks" / "ops",
    REPO_ROOT / "benchmarks" / "kernels",
)


def _production_benchmark_files() -> list[Path]:
    files: list[Path] = []
    for bench_dir in BENCHMARK_DIRS:
        files.extend(
            path for path in bench_dir.rglob("*.py")
            if path.name != "__init__.py"
        )
    return sorted(files)


def _test_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "tests" or (node.module or "").startswith("tests."):
                imports.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Import):
            imports.extend(
                f"import {alias.name}" for alias in node.names
                if alias.name == "tests" or alias.name.startswith("tests.")
            )
    return imports


def _gen_inputs_definitions(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        f"line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "gen_inputs"
    ]


@pytest.mark.smoke
def test_production_benchmarks_do_not_import_tests_package() -> None:
    offenders = {
        str(path.relative_to(REPO_ROOT)): imports
        for path in _production_benchmark_files()
        if (imports := _test_imports(path))
    }

    assert offenders == {}


@pytest.mark.smoke
def test_production_benchmarks_do_not_author_gen_inputs() -> None:
    """A benchmark imports the op's workload from ``workloads/``.

    If the op has none, add it there in its own PR rather than growing a second
    copy of the inputs here.
    """
    offenders = {
        str(path.relative_to(REPO_ROOT)): sites
        for path in _production_benchmark_files()
        if (sites := _gen_inputs_definitions(path))
    }

    assert offenders == {}
