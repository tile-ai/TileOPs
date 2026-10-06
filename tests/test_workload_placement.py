"""Structural gates for the workloads layer.

Input construction and the op's reference computation live in
``workloads/<family>.py`` or its family package so tests and benchmarks read one
definition, including numerical policy. Pytest checks, timing and roofline
calculations remain consumer responsibilities.

See docs/design/layer-boundaries.md §Test and §Workloads Layer.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

# Consumer execution and accounting do not belong in workloads.
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
def test_workloads_carry_no_consumer_execution() -> None:
    assert _methods_named(REPO_ROOT / "workloads", NOT_IN_WORKLOADS.__contains__) == {}


@pytest.mark.smoke
def test_workloads_do_not_import_the_benchmark_layer() -> None:
    """The benchmark layer consumes the workload contract; the dependency never runs back."""
    offenders = {}
    for path in sorted((REPO_ROOT / "workloads").rglob("*.py")):
        hits = [
            f"line {node.lineno}"
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
            if isinstance(node, ast.ImportFrom)
            and (node.module or "").split(".")[0] == "benchmarks"
            or isinstance(node, ast.Import)
            and any(alias.name.split(".")[0] == "benchmarks" for alias in node.names)
        ]
        if hits:
            offenders[str(path.relative_to(REPO_ROOT))] = hits
    assert offenders == {}


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


@pytest.mark.smoke
def test_consumers_do_not_override_workload_contract():
    for directory in ("tests/ops", "benchmarks/ops"):
        assert (
            _methods_named(REPO_ROOT / directory, {"ref_program", "verification"}.__contains__)
            == {}
        )
    offenders = []
    for path in [
        *(REPO_ROOT / "benchmarks/ops").glob("*.py"),
        *(REPO_ROOT / "tests/ops").glob("*.py"),
    ]:
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"Exact", "Partial", "Custom"}
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders


@pytest.mark.smoke
def test_consumers_do_not_define_private_numerical_comparators():
    offenders = []
    for directory in ("tests/ops", "benchmarks/ops"):
        for path in (REPO_ROOT / directory).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if (
                    isinstance(node, ast.FunctionDef)
                    and not node.name.startswith("test_")
                    and ("compare" in node.name or "tolerance" in node.name)
                ):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} {node.name}")
    assert not offenders, offenders
