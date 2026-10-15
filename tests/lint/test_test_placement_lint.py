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


def test_names_that_look_like_ops_do_not_replace_an_op_call(tmp_path: Path) -> None:
    source = (
        'def test_ref(op, op_name="FooOp"):\n'
        "    ref_op = ref\n"
        "    y = ref_op(x)\n"
        "    compare_outputs(y, ref_op(x), check)\n"
    )

    (found,) = _findings(tmp_path, source)

    assert "test_ref calls compare_outputs without calling an op" in found


def test_constructing_an_op_without_invoking_it_does_not_replace_an_op_call(
    tmp_path: Path,
) -> None:
    source = "def test_ref():\n    candidate = GemmFwdOp(m=1)\n    compare_outputs(y, y, check)\n"

    assert _findings(tmp_path, source)


@pytest.mark.parametrize(
    "inspection",
    [
        "print(op)",
        "isinstance(op, object)",
        "type(op)",
        "repr(op)",
        "str(op)",
        "len(op)",
        "id(op)",
        'getattr(op, "forward")',
        'hasattr(op, "forward")',
    ],
    ids=["print", "isinstance", "type", "repr", "str", "len", "id", "getattr", "hasattr"],
)
def test_passing_an_op_to_an_inspection_builtin_is_not_an_op_call(
    tmp_path: Path, inspection: str
) -> None:
    source = f"def test_ref(op):\n    {inspection}\n    compare_outputs(ref(x), ref(x), check)\n"

    (found,) = _findings(tmp_path, source)

    assert "test_ref calls compare_outputs without calling an op" in found


def test_calling_a_reference_class_named_cls_is_not_an_op_call(tmp_path: Path) -> None:
    source = (
        "def test_ref():\n"
        "    cls = ReferenceImpl\n"
        "    cls()(x)\n"
        "    compare_outputs(ref(x), ref(x), check)\n"
    )

    (found,) = _findings(tmp_path, source)

    assert "test_ref calls compare_outputs without calling an op" in found


@pytest.mark.parametrize(
    "body",
    [
        "    got = run(lambda: _op(op_name, dim=dim)(x))\n",
        ("    def closure():\n        return _op(op_name, dim=dim)(x)\n    got = run(closure)\n"),
    ],
    ids=["lambda", "nested-def"],
)
def test_an_op_call_in_a_nested_scope_counts(tmp_path: Path, body: str) -> None:
    source = (
        "import tileops.ops.reduction as mod\n\n"
        "def _op(name, **kwargs):\n"
        "    return getattr(mod, name)(**kwargs)\n\n"
        "def run(fn):\n"
        "    return fn()\n\n"
        f"def test_ref(op_name, dim):\n{body}"
        "    compare_outputs(got, want, check)\n"
    )

    assert _findings(tmp_path, source) == []


def test_a_reference_call_in_a_lambda_is_not_an_op_call(tmp_path: Path) -> None:
    source = (
        "def run(fn):\n"
        "    return fn()\n\n"
        "def test_ref():\n"
        "    got = run(lambda: ref(x))\n"
        "    compare_outputs(got, ref(x), check)\n"
    )

    assert _findings(tmp_path, source)


def test_a_dynamic_torch_reference_is_not_an_op_call(tmp_path: Path) -> None:
    source = (
        "def test_ref(name):\n"
        "    ref_fn = getattr(torch, name)\n"
        "    got = ref_fn(x)\n"
        "    compare_outputs(got, ref_fn(x), check)\n"
    )

    assert _findings(tmp_path, source)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            "import torch\n\n"
            "def test_torch_getattr(name):\n"
            "    ref_fn = getattr(torch, name)\n"
            "    y = ref_fn(x)\n"
            "    torch.testing.assert_close(y, y)\n"
            "    compare_outputs(y, ref_fn(x), check)\n",
            id="torch",
        ),
        pytest.param(
            "import torch\nimport tileops.ops.elementwise as elementwise_mod\n\n"
            "def test_mixed(op_name, flag):\n"
            "    module = elementwise_mod if flag else torch\n"
            "    y = getattr(module, op_name)(x)\n"
            "    torch.testing.assert_close(y, y)\n"
            "    compare_outputs(y, ref(x), check)\n",
            id="one-branch-not-a-tileops-module",
        ),
    ],
)
def test_getattr_by_name_on_a_non_tileops_module_is_not_an_op_call(
    tmp_path: Path, source: str
) -> None:
    (found,) = _findings(tmp_path, source)

    assert "calls compare_outputs without calling an op" in found


def _op_class_call(module: str) -> str:
    return f"    cls = getattr({module}, op_name)\n    out = cls()(a, b)\n"


@pytest.mark.parametrize(
    ("imports", "call"),
    [
        pytest.param(
            "import tileops.ops.elementwise as elementwise_mod",
            _op_class_call("elementwise_mod"),
            id="import-as",
        ),
        pytest.param(
            "from tileops.ops import elementwise as elementwise_mod",
            _op_class_call("elementwise_mod"),
            id="from-import-as",
        ),
        pytest.param(
            "from tileops.ops import elementwise", _op_class_call("elementwise"), id="from-import"
        ),
        pytest.param(
            "import tileops.ops.elementwise", _op_class_call("tileops.ops.elementwise"), id="dotted"
        ),
        pytest.param(
            "import tileops.elementwise as elementwise_mod",
            _op_class_call("elementwise_mod"),
            id="public-path",
        ),
        pytest.param(
            "import tileops.reduction as reduction",
            "    out = getattr(reduction, name)(dim=0)(x)\n",
            id="public-path-called-inline",
        ),
        pytest.param(
            "import tileops.ops.elementwise as a\nimport tileops.ops.reduction as b\n"
            "elementwise_mod = a if flag else b",
            _op_class_call("elementwise_mod"),
            id="either-of-two-tileops-modules",
        ),
    ],
)
def test_getattr_by_name_on_a_tileops_module_is_an_op_class(
    tmp_path: Path, imports: str, call: str
) -> None:
    source = (
        f"{imports}\n\ndef test_ref(name, op_name):\n{call}    compare_outputs(out, ref, check)\n"
    )

    assert _findings(tmp_path, source) == []


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


def test_a_line_allowance_with_a_reason_suppresses_the_finding(tmp_path: Path) -> None:
    source = (
        "def test_ref():\n"
        "    compare_outputs(y, ref(x), check)  # lint: test-placement allow: contract probe\n"
    )

    assert _findings(tmp_path, source) == []


def test_a_bare_line_allowance_does_not_suppress_the_finding(tmp_path: Path) -> None:
    source = (
        "def test_ref():\n    compare_outputs(y, ref(x), check)  # lint: test-placement allow:\n"
    )

    assert _findings(tmp_path, source)
