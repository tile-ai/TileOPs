"""``tileops.models`` composes models from the public op interface only."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.structural]

_MODELS = Path(__file__).resolve().parents[1] / "src" / "tileops" / "models"


def _private_imports(root: Path) -> list[str]:
    """Each import under ``root`` of a private ``tileops`` module or of a repo-only package."""
    found = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                continue
            for name in names:
                parts = name.split(".")
                repo_only = parts[0] in ("workloads", "tests", "benchmarks")
                private = parts[0] == "tileops" and any(p.startswith("_") for p in parts[1:])
                if repo_only or private:
                    found.append(f"{path.relative_to(root)}:{node.lineno}: {name}")
    return found


def test_models_import_no_private_module():
    """A model reaches kernels only through the ops a caller of the package can construct."""
    assert _private_imports(_MODELS) == []


@pytest.mark.parametrize(
    "source",
    [
        "from tileops._ops.gemm import GEMMFwdOp",
        "import tileops._kernels.gemm",
        "from tileops import _utils",
        "from workloads.models import reference",
    ],
)
def test_a_private_import_is_reported(tmp_path, source):
    (tmp_path / "model.py").write_text(source + "\n")
    assert len(_private_imports(tmp_path)) == 1


def test_a_public_op_import_is_not_reported(tmp_path):
    (tmp_path / "model.py").write_text("from tileops.gemm import GEMMFwdOp\n")
    assert _private_imports(tmp_path) == []
