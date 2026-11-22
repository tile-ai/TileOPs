"""The import directions between the package, the experimental ``models/`` tree and the repo.

``src/tileops`` ships and depends on no model or checkpoint package; ``models/`` composes the
public ops only; op tests and benchmarks do not depend on a model.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.structural]

_ROOT = Path(__file__).resolve().parents[1]
_REPO_ONLY = ("models", "workloads", "tests", "benchmarks")
_CHECKPOINT = ("safetensors", "transformers", "huggingface_hub")


def _imports(root: Path) -> list[tuple[str, str]]:
    """``(location, module)`` for every absolute import in the Python files under ``root``."""
    found = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                continue
            found += [(f"{path.relative_to(root)}:{node.lineno}", name) for name in names]
    return found


def _package_violations(root: Path) -> list[str]:
    """Imports under ``root`` (the shipped package) of a repo-only tree or a checkpoint package."""
    banned = (*_REPO_ONLY, *_CHECKPOINT)
    return [f"{at}: {name}" for at, name in _imports(root) if name.split(".")[0] in banned]


def _model_violations(root: Path) -> list[str]:
    """Imports under ``root`` (a model tree) of a private ``tileops`` module."""
    return [
        f"{at}: {name}"
        for at, name in _imports(root)
        if name.split(".")[0] == "tileops" and any(p.startswith("_") for p in name.split(".")[1:])
    ]


def _consumer_violations(root: Path) -> list[str]:
    """Imports under ``root`` (op tests or benchmarks) of a model."""
    return [f"{at}: {name}" for at, name in _imports(root) if name.split(".")[0] == "models"]


@pytest.mark.parametrize(
    ("check", "tree"),
    [
        pytest.param(_package_violations, "src/tileops", id="package"),
        pytest.param(_model_violations, "models", id="models"),
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
        pytest.param(_consumer_violations, "import models.deepseek_v4", id="consumer-model"),
    ],
)
def test_a_forbidden_import_is_reported(tmp_path, check, source):
    (tmp_path / "m.py").write_text(source + "\n")
    assert len(check(tmp_path)) == 1


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
