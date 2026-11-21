"""Structural gates for the workloads layer.

Input construction and the op's reference computation live in
``workloads/<family>.py`` or its family package so tests and benchmarks read one
definition, including numerical policy. Pytest checks, timing and roofline
calculations remain consumer responsibilities.

See docs/design/layer-boundaries.md §Test and §Workloads layer.
"""

import ast
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
CONSUMERS = ("tests/ops", "benchmarks/ops")


def _offenders(directories: Iterable[str], hits_of: Callable[[ast.AST], str | None]) -> dict:
    """Map each file under *directories* to the descriptions *hits_of* gives its nodes."""
    offenders = {}
    for directory in directories:
        for path in sorted((REPO_ROOT / directory).rglob("*.py")):
            if path.resolve() == SELF:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            hits = [hit for node in ast.walk(tree) if (hit := hits_of(node)) is not None]
            if hits:
                offenders[str(path.relative_to(REPO_ROOT))] = hits
    return offenders


def _methods_named(*names: str) -> Callable[[ast.AST], str | None]:
    """A node test matching a method of one of *names* that any class defines."""

    def hits_of(node: ast.AST) -> str | None:
        if isinstance(node, ast.ClassDef):
            methods = [m for m in node.body if isinstance(m, ast.FunctionDef) and m.name in names]
            if methods:
                return ", ".join(f"{node.name}.{m.name} (line {m.lineno})" for m in methods)
        return None

    return hits_of


def test_tests_do_not_author_gen_inputs() -> None:
    assert _offenders(["tests"], _methods_named("gen_inputs")) == {}


def test_workloads_carry_no_consumer_execution() -> None:
    names = ("check", "calculate_flops", "calculate_memory")
    assert _offenders(["workloads"], _methods_named(*names)) == {}


def test_workloads_do_not_import_the_benchmark_layer() -> None:
    """The benchmark layer consumes the workload contract; the dependency never runs back."""

    def imports_benchmarks(node: ast.AST) -> str | None:
        if isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        elif isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        else:
            return None
        return (
            f"line {node.lineno}" if any(m.split(".")[0] == "benchmarks" for m in modules) else None
        )

    assert _offenders(["workloads"], imports_benchmarks) == {}


def test_workloads_do_not_seed_the_global_rng() -> None:
    """A workload that reseeds the global RNG moves every later draw in the session.

    The conftests seed it once per test; an input that must not move with the
    stream takes ``WorkloadBase.rng()`` instead. ``generator.manual_seed(...)``
    seeds a generator the workload owns, which is what the rule asks for.
    """

    def seeds_global_rng(node: ast.AST) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        seeds = (isinstance(func, ast.Name) and func.id == "manual_seed") or (
            isinstance(func, ast.Attribute)
            and func.attr == "manual_seed"
            and isinstance(func.value, ast.Name)
            and func.value.id == "torch"
        )
        return f"line {node.lineno}" if seeds else None

    assert _offenders(["workloads"], seeds_global_rng) == {}


def test_consumers_do_not_override_workload_contract():
    assert _offenders(CONSUMERS, _methods_named("ref_program", "verification")) == {}

    def builds_evidence(node: ast.AST) -> str | None:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"Exact", "Partial", "Custom"}
        ):
            return f"line {node.lineno}"
        return None

    assert _offenders(CONSUMERS, builds_evidence) == {}


def test_consumers_do_not_define_private_numerical_comparators():
    def defines_comparator(node: ast.AST) -> str | None:
        if (
            isinstance(node, ast.FunctionDef)
            and not node.name.startswith("test_")
            and ("compare" in node.name or "tolerance" in node.name)
        ):
            return f"line {node.lineno} {node.name}"
        return None

    assert _offenders(CONSUMERS, defines_comparator) == {}
