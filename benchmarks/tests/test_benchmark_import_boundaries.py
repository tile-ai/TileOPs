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


@pytest.mark.smoke
def test_production_benchmarks_do_not_import_tests_package() -> None:
    offenders = {
        str(path.relative_to(REPO_ROOT)): imports
        for path in _production_benchmark_files()
        if (imports := _test_imports(path))
    }

    assert offenders == {}
