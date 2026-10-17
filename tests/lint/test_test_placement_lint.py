"""Unit tests for ``scripts/lint/test_placement_lint.py``."""

from pathlib import Path

import pytest
import test_placement_lint as lint

pytestmark = pytest.mark.smoke


def _findings(tmp_path: Path, source: str) -> list[str]:
    path = tmp_path / "test_probe.py"
    path.write_text(source, encoding="utf-8")
    return lint.findings(path)


def test_a_kernel_construction_is_reported_with_its_function(tmp_path: Path) -> None:
    (found,) = _findings(tmp_path, "def test_tiles():\n    kernel = GemmKernel(64, 64)\n")

    assert found.endswith(
        "test_probe.py:2: test_tiles constructs GemmKernel directly — a test under "
        "tests/ops/ reaches the kernel through the op, or add "
        '"# lint: test-placement allow: <reason>" on the line'
    )


CONSTRUCTIONS = [
    pytest.param(
        "def _reference(x):\n    return GemmKernel(64)(x)\n\n\n"
        "def test_ref():\n    _reference(1)\n",
        "_reference",
        id="inside-a-helper",
    ),
    pytest.param(
        "import tileops.kernels.gemm as gemm\n\n\ndef test_ref():\n    gemm.GemmKernel(64)\n",
        "test_ref",
        id="through-a-module-attribute",
    ),
    pytest.param("KERNEL = GemmKernel(64)\n", "<module>", id="at-module-scope"),
    pytest.param(
        "def test_ref():\n    run(lambda: GemmKernel(64))\n", "test_ref", id="inside-a-lambda"
    ),
]


@pytest.mark.parametrize(("source", "where"), CONSTRUCTIONS)
def test_a_kernel_construction_anywhere_in_the_file_is_reported(
    tmp_path: Path, source: str, where: str
) -> None:
    (found,) = _findings(tmp_path, source)

    assert f" {where} constructs GemmKernel directly" in found


# A kernel class named without a call constructs nothing; the op still builds the kernel.
NOT_CONSTRUCTED = [
    pytest.param("class StandIn(GemmKernel):\n    pass\n", id="subclass-definition"),
    pytest.param(
        "def test_ref(op):\n    assert isinstance(op.kernel, GemmKernel)\n", id="isinstance-check"
    ),
    pytest.param(
        'def test_ref():\n    op = GemmFwdOp(kernel_map={"gemm": GemmKernel})\n    op(x)\n',
        id="class-handed-to-the-op",
    ),
    pytest.param("def test_ref(op):\n    op.kernel(x)\n", id="lowercase-kernel-attribute"),
]


@pytest.mark.parametrize("source", NOT_CONSTRUCTED)
def test_a_kernel_class_that_is_not_called_is_left_alone(tmp_path: Path, source: str) -> None:
    assert _findings(tmp_path, source) == [], source


def test_a_line_allowance_with_a_reason_suppresses_the_finding(tmp_path: Path) -> None:
    source = "def test_ref():\n    GemmKernel(64)  # lint: test-placement allow: layout probe\n"

    assert _findings(tmp_path, source) == []
