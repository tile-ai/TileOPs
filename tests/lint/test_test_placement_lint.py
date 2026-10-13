"""Unit tests for ``scripts/lint/test_placement_lint.py``."""

from pathlib import Path

import pytest
import test_placement_lint as lint

pytestmark = pytest.mark.smoke


def _findings(tmp_path: Path, source: str, added: set[int] | None = None) -> list[str]:
    path = tmp_path / "test_probe.py"
    path.write_text(source, encoding="utf-8")
    return lint.findings(path, added)


def test_a_kernel_construction_is_reported_with_its_function(tmp_path: Path) -> None:
    (found,) = _findings(tmp_path, "def test_tiles():\n    kernel = GemmKernel(64, 64)\n")

    assert found.endswith(
        "test_probe.py:2: test_tiles constructs GemmKernel directly — a test under "
        "tests/ops/ reaches the kernel through the op"
    )


def test_a_kernel_subclass_definition_is_left_alone(tmp_path: Path) -> None:
    assert _findings(tmp_path, "class StandIn(GemmKernel):\n    pass\n") == []


def test_compare_outputs_without_an_op_is_reported_with_its_function(tmp_path: Path) -> None:
    source = "def test_ref():\n    y = ref(x)\n    compare_outputs(y, ref(x), check)\n"

    (found,) = _findings(tmp_path, source)

    assert "test_probe.py:3: test_ref calls compare_outputs without calling an op" in found


REACHING = [
    pytest.param("()", "    y = GemmFwdOp(m=1)(x)\n", id="op-class"),
    pytest.param("(op)", "    y = op(x)\n", id="op-variable"),
    pytest.param("(op_cls)", "    y = op_cls(dim=0)(x)\n", id="op-class-variable"),
    pytest.param("()", "    y = getattr(ops, 'SumFwdOp')(dim=0)(x)\n", id="op-named-by-a-string"),
    pytest.param("()", "    y = _build()(x)\n", id="through-a-helper-of-the-module"),
    pytest.param("()", "    y = _case()(x)\n", id="through-a-helper-of-a-helper"),
]


@pytest.mark.parametrize(("params", "call"), REACHING)
def test_compare_outputs_after_an_op_call_is_left_alone(
    tmp_path: Path, params: str, call: str
) -> None:
    helper = "def _build():\n    return GemmFwdOp(m=1)\n\n\ndef _case():\n    return _build()\n\n\n"
    source = f"{helper}def test_ref{params}:\n{call}    compare_outputs(y, ref(x), check)\n"

    assert _findings(tmp_path, source) == [], call


UNTOUCHED = [
    pytest.param("    GemmKernel(64)\n", id="kernel-construction"),
    pytest.param("    compare_outputs(y, ref(x), check)\n", id="compare-outputs-caller"),
]


@pytest.mark.parametrize("body", UNTOUCHED)
def test_only_what_the_added_lines_touch_is_reported(tmp_path: Path, body: str) -> None:
    source = f"def test_a():\n{body}\n\ndef test_b():\n{body}"

    (found,) = _findings(tmp_path, source, added={6})

    assert "test_probe.py:6: test_b " in found
