"""Structural gate for the test stage's MUST PROVIDE obligation.

Input generation lives in ``workloads/<family>.py`` so both the test stage and
the benchmark stage can import it. A workload authored inside ``tests/`` is
unreachable from ``benchmarks/``, which may neither import ``tests`` nor write
``workloads/`` — the benchmark's only remaining option is a second copy.

See docs/design/trust-model.md §Test.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"


def _test_files() -> list[Path]:
    return sorted(
        path for path in TESTS_DIR.rglob("*.py")
        if path.name != "__init__.py" and path.name != Path(__file__).name
    )


def _inline_workloads(path: Path) -> list[str]:
    """Classes that build their own inputs instead of composing a workload."""
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if any(
            isinstance(m, ast.FunctionDef) and m.name == "gen_inputs"
            for m in node.body
        ):
            offenders.append(f"{node.name} (line {node.lineno})")
    return offenders


@pytest.mark.smoke
def test_tests_do_not_author_gen_inputs() -> None:
    offenders = {
        str(path.relative_to(REPO_ROOT)): names
        for path in _test_files()
        if (names := _inline_workloads(path))
    }

    assert offenders == {}
