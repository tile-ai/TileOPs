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

# Names that mark an oracle or a baseline rather than input construction.
ORACLE_PREFIXES = ("ref_",)
ORACLE_SUFFIXES = ("_ref", "_baseline", "_ref_program")


def _production_benchmark_files() -> list[Path]:
    files: list[Path] = []
    for bench_dir in BENCHMARK_DIRS:
        files.extend(
            path for path in bench_dir.rglob("*.py")
            if path.name != "__init__.py"
        )
    return sorted(files)


def _looks_like_oracle(name: str) -> bool:
    return name.startswith(ORACLE_PREFIXES) or name.endswith(ORACLE_SUFFIXES)


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


def _workload_oracle_imports(path: Path) -> list[str]:
    """Oracle-shaped symbols pulled out of the shared workloads layer.

    The layer is gated to hold inputs only, so this is defence in depth: it
    catches a symbol that slipped in, including one renamed at the import site.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if module != "workloads" and not module.startswith("workloads."):
            continue
        hits += [
            f"from {module} import {alias.name} (line {node.lineno})"
            for alias in node.names
            if _looks_like_oracle(alias.name)
        ]
    return hits


def _bound_names(node: ast.AST) -> list[tuple[str, int]]:
    """Names a body binds to a callable, including async and alias forms."""
    bound = []
    for child in getattr(node, "body", []):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.append((child.name, child.lineno))
        elif isinstance(child, ast.Assign):
            bound += [
                (t.id, child.lineno) for t in child.targets if isinstance(t, ast.Name)
            ]
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            bound.append((child.target.id, child.lineno))
    return bound


def _definitions_named(path: Path, name: str) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = [
        f"module-level {name} (line {lineno})"
        for bound, lineno in _bound_names(tree) if bound == name
    ]
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            hits += [
                f"{node.name}.{name} (line {lineno})"
                for bound, lineno in _bound_names(node) if bound == name
            ]
    return hits


@pytest.mark.smoke
def test_production_benchmarks_do_not_import_tests_package() -> None:
    offenders = {
        str(path.relative_to(REPO_ROOT)): imports
        for path in _production_benchmark_files()
        if (imports := _test_imports(path))
    }

    assert offenders == {}


@pytest.mark.smoke
def test_production_benchmarks_do_not_import_workload_oracles() -> None:
    """Workload classes are the intended import; oracles out of that layer are not."""
    offenders = {
        str(path.relative_to(REPO_ROOT)): imports
        for path in _production_benchmark_files()
        if (imports := _workload_oracle_imports(path))
    }

    assert offenders == {}


@pytest.mark.smoke
def test_production_benchmarks_do_not_author_gen_inputs() -> None:
    """Input generation belongs to the shared layer both stages import.

    A benchmark that needs a workload imports it from ``workloads/``. If the op
    has none, that is a test-stage MUST PROVIDE gap: add it in a workloads-only
    PR rather than growing a second copy here.
    """
    offenders = {
        str(path.relative_to(REPO_ROOT)): sites
        for path in _production_benchmark_files()
        if (sites := _definitions_named(path, "gen_inputs"))
    }

    assert offenders == {}


@pytest.mark.smoke
def test_production_benchmarks_do_not_define_ref_program() -> None:
    """``ref_program`` names the test stage's correctness oracle.

    A benchmark's PyTorch implementation is a timing baseline, not an oracle:
    name it ``torch_baseline`` so the two never read as the same artifact.
    """
    offenders = {
        str(path.relative_to(REPO_ROOT)): sites
        for path in _production_benchmark_files()
        if (sites := _definitions_named(path, "ref_program"))
    }

    assert offenders == {}
