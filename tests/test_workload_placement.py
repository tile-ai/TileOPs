"""Structural gates for the workloads layer's contract.

Input generation lives in ``workloads/<family>.py`` so both the test stage and
the benchmark stage can import it. A workload authored inside ``tests/`` is
unreachable from ``benchmarks/``, which may neither import ``tests`` nor write
``workloads/`` — the benchmark's only remaining option is a second copy.

The shared layer carries inputs and nothing else. An oracle or a baseline
placed there becomes a surface both stages read, which is the coupling the
stage boundary exists to prevent.

See docs/design/trust-model.md §Test and §Workloads Layer.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
WORKLOADS_DIR = REPO_ROOT / "workloads"

# Names that mark a correctness oracle or a timing baseline rather than inputs.
FORBIDDEN_IN_WORKLOADS = ("ref_program", "torch_baseline", "check")


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*.py")
        if path.name != "__init__.py" and path.resolve() != Path(__file__).resolve()
    )


def _classes_defining(path: Path, names: tuple[str, ...]) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        hits = sorted(
            m.name for m in node.body
            if isinstance(m, ast.FunctionDef) and m.name in names
        )
        if hits:
            offenders.append(f"{node.name} defines {hits} (line {node.lineno})")
    return offenders


@pytest.mark.smoke
def test_tests_do_not_author_gen_inputs() -> None:
    offenders = {
        str(path.relative_to(REPO_ROOT)): names
        for path in _python_files(TESTS_DIR)
        if (names := _classes_defining(path, ("gen_inputs",)))
    }

    assert offenders == {}


@pytest.mark.smoke
def test_workloads_carry_inputs_only() -> None:
    offenders = {
        str(path.relative_to(REPO_ROOT)): names
        for path in _python_files(WORKLOADS_DIR)
        if (names := _classes_defining(path, FORBIDDEN_IN_WORKLOADS))
    }

    assert offenders == {}
