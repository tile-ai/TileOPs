"""Structural gates for the workloads layer's contract.

Input generation lives in ``workloads/<family>.py`` so both the test stage and
the benchmark stage can import it. A workload authored inside ``tests/`` is
unreachable from ``benchmarks/``, which may neither import ``tests`` nor write
``workloads/`` — the benchmark's only remaining option is a second copy.

The shared layer carries inputs and nothing else. An oracle, a tolerance or a
baseline placed there becomes a surface both stages read, which is the coupling
the stage boundary exists to prevent.

See docs/design/trust-model.md §Test and §Workloads Layer.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"
WORKLOADS_DIR = REPO_ROOT / "workloads"

# Exact names that mark a correctness oracle, a tolerance, or roofline maths.
FORBIDDEN_IN_WORKLOADS = frozenset(
    {"ref_program", "check", "calculate_flops", "calculate_memory"}
)
# Any timing baseline, whatever the vendor: torch_baseline, flashinfer_baseline...
BASELINE_SUFFIX = "_baseline"


def _python_files(root: Path) -> list[Path]:
    here = Path(__file__).resolve()
    return sorted(
        path for path in root.rglob("*.py")
        if path.name != "__init__.py" and path.resolve() != here
    )


def _is_forbidden(name: str, forbidden: frozenset[str], baselines: bool) -> bool:
    return name in forbidden or (baselines and name.endswith(BASELINE_SUFFIX))


def _defined_names(node: ast.AST) -> list[tuple[str, int]]:
    """Every name a body binds to a callable, however it is spelled.

    Covers ``def``, ``async def``, and the alias forms ``name = fn`` and
    ``name: T = fn`` that would otherwise slip past a def-only scan.
    """
    bound = []
    for child in getattr(node, "body", []):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.append((child.name, child.lineno))
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    bound.append((target.id, child.lineno))
        elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
            bound.append((child.target.id, child.lineno))
    return bound


def _offending_definitions(
    path: Path, forbidden: frozenset[str], *, baselines: bool = False
) -> list[str]:
    """Forbidden names bound at module level or inside any class in ``path``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    hits = [
        f"module-level {name} (line {lineno})"
        for name, lineno in _defined_names(tree)
        if _is_forbidden(name, forbidden, baselines)
    ]
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        hits += [
            f"{node.name}.{name} (line {lineno})"
            for name, lineno in _defined_names(node)
            if _is_forbidden(name, forbidden, baselines)
        ]
    return hits


@pytest.mark.smoke
def test_tests_do_not_author_gen_inputs() -> None:
    offenders = {
        str(path.relative_to(REPO_ROOT)): names
        for path in _python_files(TESTS_DIR)
        if (names := _offending_definitions(path, frozenset({"gen_inputs"})))
    }

    assert offenders == {}


@pytest.mark.smoke
def test_workloads_carry_inputs_only() -> None:
    offenders = {
        str(path.relative_to(REPO_ROOT)): names
        for path in _python_files(WORKLOADS_DIR)
        if (names := _offending_definitions(path, FORBIDDEN_IN_WORKLOADS, baselines=True))
    }

    assert offenders == {}
