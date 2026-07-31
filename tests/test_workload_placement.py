"""Structural gates for the workloads layer.

Input construction lives in ``workloads/<family>.py`` so both the test stage and
the benchmark stage can import it, and that layer carries inputs only — an
oracle or a baseline placed there becomes a surface both stages share.

See docs/design/trust-model.md §Test and §Workloads Layer.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()

# Oracle, tolerance and roofline names, plus any vendor's timing baseline.
NOT_IN_WORKLOADS = ("ref_program", "check", "calculate_flops", "calculate_memory")


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
def test_workloads_carry_inputs_only() -> None:
    forbidden = lambda n: n in NOT_IN_WORKLOADS or n.endswith("_baseline")  # noqa: E731
    assert _methods_named(REPO_ROOT / "workloads", forbidden) == {}
