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


def _defined_methods(path: Path, name: str) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        f"line {node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
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
    """Input generation belongs to the shared layer both stages import.

    A benchmark that needs a workload imports it from ``workloads/``. If the op
    has none, that is a test-stage MUST PROVIDE gap: add it in a workloads-only
    PR rather than growing a second copy here.
    """
    offenders = {
        str(path.relative_to(REPO_ROOT)): sites
        for path in _production_benchmark_files()
        if (sites := _defined_methods(path, "gen_inputs"))
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
        if (sites := _defined_methods(path, "ref_program"))
    }

    assert offenders == {}
