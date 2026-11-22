"""The import directions between the package, the experimental ``models/`` tree and the repo.

``src/tileops`` ships and depends on no model or checkpoint package; ``models/`` reaches no private
``tileops`` module; a model's reference imports no ``tileops``; references, op tests and
benchmarks do not depend on a model. Whether a model computes only through ops is a review
question, not an import one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.structural]

_ROOT = Path(__file__).resolve().parents[1]
_REPO_ONLY = ("models", "workloads", "tests", "benchmarks")
_CHECKPOINT = ("safetensors", "transformers", "huggingface_hub")


def _dotted(node: ast.AST) -> str | None:
    """``a.b.c`` for an attribute chain rooted at a name, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _imports(root: Path) -> list[tuple[str, str]]:
    """``(location, module)`` for every absolute import in the Python files under ``root``, and
    for every ``tileops.<...>`` attribute chain, which reaches a module as an import does."""
    found = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [f"{node.module}.{alias.name}" for alias in node.names]
            elif isinstance(node, ast.Attribute) and (chain := _dotted(node)):
                names = [chain] if chain.startswith("tileops.") else []
            else:
                continue
            found += [(f"{path.relative_to(root)}:{node.lineno}", name) for name in names]
    return found


def _package_violations(root: Path) -> list[str]:
    """Imports under ``root`` (the shipped package) of a repo-only tree or a checkpoint package."""
    banned = (*_REPO_ONLY, *_CHECKPOINT)
    return [f"{at}: {name}" for at, name in _imports(root) if name.split(".")[0] in banned]


def _model_violations(root: Path) -> list[str]:
    """Imports under ``root`` (a model tree) of a private ``tileops`` module or of the trees that
    test it: a model that reached its reference would be checked against itself."""
    found = []
    for at, name in _imports(root):
        top, *rest = name.split(".")
        private = top == "tileops" and any(p.startswith("_") for p in rest)
        if private or top in ("workloads", "tests", "benchmarks"):
            found.append(f"{at}: {name}")
    return found


def _reference_violations(root: Path) -> list[str]:
    """Imports under ``root`` (model references) of ``tileops``: a reference that ran a TileOPs
    kernel would check the model against the code under test."""
    return [f"{at}: {name}" for at, name in _imports(root) if name.split(".")[0] == "tileops"]


def _consumer_violations(root: Path) -> list[str]:
    """Imports under ``root`` (references, op tests or benchmarks) of a model."""
    return [f"{at}: {name}" for at, name in _imports(root) if name.split(".")[0] == "models"]


@pytest.mark.parametrize(
    ("check", "tree"),
    [
        pytest.param(_package_violations, "src/tileops", id="package"),
        pytest.param(_model_violations, "models", id="models"),
        pytest.param(_reference_violations, "workloads/models", id="model-references"),
        pytest.param(_consumer_violations, "workloads", id="workloads"),
        pytest.param(_consumer_violations, "tests/ops", id="op-tests"),
        pytest.param(_consumer_violations, "benchmarks", id="benchmarks"),
    ],
)
def test_the_repository_keeps_the_import_directions(check, tree):
    assert (_ROOT / tree).is_dir(), f"{tree} is not a directory; the check would pass on nothing"
    assert check(_ROOT / tree) == []


@pytest.mark.parametrize(
    ("check", "source"),
    [
        pytest.param(_package_violations, "from models.qwen3_5 import config", id="package-model"),
        pytest.param(_package_violations, "import safetensors", id="package-checkpoint"),
        pytest.param(_package_violations, "from workloads.device import x", id="package-repo"),
        pytest.param(_model_violations, "from tileops._ops.gemm import G", id="model-private-op"),
        pytest.param(_model_violations, "from tileops import _utils", id="model-private-name"),
        pytest.param(_model_violations, "from workloads.models import ref", id="model-reference"),
        pytest.param(_model_violations, "import tileops\ntileops._ops.G()", id="model-attribute"),
        pytest.param(_consumer_violations, "import models.deepseek_v4", id="consumer-model"),
        pytest.param(_reference_violations, "from tileops.norm import N", id="reference-tileops"),
    ],
)
def test_a_forbidden_import_is_reported(tmp_path, check, source):
    (tmp_path / "m.py").write_text(source + "\n")
    assert check(tmp_path) != []


@pytest.mark.parametrize(
    ("check", "source"),
    [
        pytest.param(_package_violations, "from tileops.gemm import GEMMFwdOp", id="package"),
        pytest.param(_model_violations, "from tileops.gemm import GEMMFwdOp", id="model"),
        pytest.param(_consumer_violations, "from workloads.device import x", id="consumer"),
    ],
)
def test_an_allowed_import_is_not_reported(tmp_path, check, source):
    (tmp_path / "m.py").write_text(source + "\n")
    assert check(tmp_path) == []
