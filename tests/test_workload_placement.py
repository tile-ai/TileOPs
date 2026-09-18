"""Structural gates for the workloads layer.

Input construction and the op's reference computation live in
``workloads/<family>.py`` so both the test stage and the benchmark stage read one
definition. Tolerances, checks and roofline numbers do not: those are decisions,
and a decision placed there reaches the other stage.

See docs/design/trust-model.md §Test and §Workloads Layer.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

# Tolerance and roofline names — decisions, not definitions.
NOT_IN_WORKLOADS = ("check", "calculate_flops", "calculate_memory")


def _methods_named(root: Path, wanted) -> dict[str, list[str]]:
    """Map each file under ``root`` to the wanted methods any class defines."""
    offenders = {}
    for path in sorted(root.rglob("*.py")):
        if path.name == "__init__.py" or path.resolve() == SELF:
            continue
        hits = [
            f"{node.name}.{m.name} (line {m.lineno})"
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
            if isinstance(node, ast.ClassDef)
            for m in node.body
            if isinstance(m, ast.FunctionDef) and wanted(m.name)
        ]
        if hits:
            offenders[str(path.relative_to(REPO_ROOT))] = hits
    return offenders


@pytest.mark.smoke
def test_tests_do_not_author_gen_inputs() -> None:
    assert _methods_named(REPO_ROOT / "tests", lambda n: n == "gen_inputs") == {}


@pytest.mark.smoke
def test_workloads_carry_no_decisions() -> None:
    assert _methods_named(REPO_ROOT / "workloads", NOT_IN_WORKLOADS.__contains__) == {}


def _seeds_global_rng(node: ast.AST) -> bool:
    """Whether *node* is a call that seeds the global RNG.

    ``torch.manual_seed(...)`` and the imported ``manual_seed(...)`` do.
    ``generator.manual_seed(...)`` does not: it seeds a generator the workload
    owns, which is what the rule asks for.
    """
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id == "manual_seed"
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "manual_seed"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    )


def _global_seed_calls(root: Path) -> dict[str, list[str]]:
    """Map each file under *root* to the calls that seed the global RNG."""
    offenders = {}
    for path in sorted(root.rglob("*.py")):
        hits = [
            f"line {node.lineno}"
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
            if _seeds_global_rng(node)
        ]
        if hits:
            offenders[str(path.relative_to(REPO_ROOT))] = hits
    return offenders


@pytest.mark.smoke
def test_workloads_do_not_seed_the_global_rng() -> None:
    """A workload that reseeds the global RNG moves every later draw in the session.

    The conftests seed it once per test; an input that must not move with the
    stream takes ``WorkloadBase.rng()`` instead.
    """
    assert _global_seed_calls(REPO_ROOT / "workloads") == {}
