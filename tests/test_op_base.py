"""Tests for tileops.ops.op_base.

Covers:
- ``Op._cache_key`` default behavior and runtime warnings for missing
  subclass overrides when ``_static_axes`` is empty.
- ``Op.eval_roofline`` and the module-level whitelist AST evaluator
  ``_safe_eval`` introduced to align the L1 base class with
  docs/ops-design.md §``eval_roofline``.
- Base-class stubs for ``_infer_output_shapes`` / ``_validate_dtypes`` that
  raise :class:`NotImplementedError` pointing at the design doc.
"""

import warnings
from typing import Dict

import pytest
import torch

from tileops.ops import op_base
from tileops.ops.op_base import Op, _safe_eval

pytestmark = pytest.mark.smoke


@pytest.fixture(autouse=True)
def _reset_warned_types():
    """Clear the module-level dedup set so each test sees a fresh warn state."""
    op_base._EMPTY_STATIC_DIMS_WARNED.clear()
    yield
    op_base._EMPTY_STATIC_DIMS_WARNED.clear()


def _make_op_subclass(*, static_axes=frozenset(), override_cache_key=False):
    """Build a minimal concrete Op subclass for testing.

    ``static_axes`` populates ``_static_axes``.
    ``override_cache_key=True`` attaches a subclass-level override.
    """
    attrs = {
        "_static_axes": static_axes,
        "default_kernel_map": property(lambda self: {}),
        "forward": lambda self, *a, **kw: None,
    }
    if override_cache_key:
        attrs["_cache_key"] = lambda self, *shapes: ("overridden",)
    return type("TestOp", (Op,), attrs)


class TestCacheKeyDefault:
    def test_static_axes_exclude_single_input(self):
        """_static_axes=[(0,1)] on a 3D input excludes axis 1 from the key."""
        Cls = _make_op_subclass(static_axes=frozenset({(0, 1)}))
        op = Cls()
        key = op._cache_key((2, 4, 8))
        assert key == (2, 8)

    def test_static_axes_across_multiple_inputs(self):
        """_static_axes can reference axes in different input positions."""
        Cls = _make_op_subclass(static_axes=frozenset({(0, 1), (1, 0)}))
        op = Cls()
        key = op._cache_key((2, 4, 8), (16, 32))
        # Input 0: exclude axis 1 -> (2, 8); Input 1: exclude axis 0 -> (32,)
        assert key == (2, 8, 32)

    def test_empty_static_axes_returns_full_shape(self):
        """With no static axes, the key concatenates all input shape values."""
        Cls = _make_op_subclass(static_axes=frozenset())
        op = Cls()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # warning tested separately
            key = op._cache_key((2, 4, 8), (3, 5))
        assert key == (2, 4, 8, 3, 5)


class TestCacheKeyWarning:
    def test_empty_static_axes_warns_once_per_type(self):
        """Default path with empty _static_axes warns exactly once per subclass,
        even across multiple instances and repeated calls."""
        Cls = _make_op_subclass(static_axes=frozenset())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Cls()._cache_key((2, 4))
            Cls()._cache_key((3, 5))
            Cls()._cache_key((7, 9))
            Cls()._cache_key((11, 13))

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "TestOp" in str(user_warnings[0].message)
        assert "_cache_key" in str(user_warnings[0].message)

    def test_override_suppresses_warning(self):
        """When the subclass overrides _cache_key, no warning fires."""
        Cls = _make_op_subclass(
            static_axes=frozenset(), override_cache_key=True
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = Cls()._cache_key((2, 4))

        assert result == ("overridden",)
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warnings == []

    def test_populated_static_axes_suppresses_warning(self):
        """Non-empty _static_axes means the user committed at ctor; no warning
        fires regardless of whether _cache_key was overridden."""
        Cls = _make_op_subclass(static_axes=frozenset({(0, 0)}))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Cls()._cache_key((2, 4))

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert user_warnings == []

    def test_distinct_subclasses_each_warn_once(self):
        """Two different subclasses each warn once; the dedup set is keyed by
        type, not globally suppressed after the first warning."""
        ClsA = _make_op_subclass(static_axes=frozenset())
        ClsB = _make_op_subclass(static_axes=frozenset())

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ClsA()._cache_key((1,))
            ClsA()._cache_key((2,))  # no re-warn for A
            ClsB()._cache_key((3,))  # fresh warn for B
            ClsB()._cache_key((4,))  # no re-warn for B

        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 2


# ---------------------------------------------------------------------------
# L1 codegen contract: eval_roofline, _safe_eval, NotImplementedError stubs
# ---------------------------------------------------------------------------


class _MinimalOp(Op):
    """Smallest Op subclass that can be instantiated.

    Implements only the strictly-abstract members so tests can exercise
    base-class behavior without pulling any kernel code.
    """

    def __init__(self, *, dtype=None):
        self.dtype = dtype

    @property
    def default_kernel_map(self) -> Dict[str, object]:  # pragma: no cover - unused
        return {}

    def forward(self, *args, **kwargs):  # pragma: no cover - unused
        return None


class _RooflineOp(_MinimalOp):
    """Op subclass that declares real roofline slots."""

    _roofline_vars = ["M", "N"]
    _flops_expr = "4 * M * N"
    _bytes_expr = "(2 * M * N + N) * elem_bytes"

    def __init__(self, *, M: int, N: int, dtype: torch.dtype):
        super().__init__(dtype=dtype)
        self.M = M
        self.N = N


class TestEvalRoofline:
    def test_defaults_return_zero(self):
        """Class-level slots at defaults -> eval_roofline returns (0, 0)."""
        op = _MinimalOp(dtype=torch.float16)
        assert op.eval_roofline() == (0, 0)

    def test_real_expression(self):
        """``4*M*N`` and ``(2*M*N + N)*elem_bytes`` with M=128, N=256, fp16."""
        op = _RooflineOp(M=128, N=256, dtype=torch.float16)
        flops, nbytes = op.eval_roofline()
        # 4 * 128 * 256 = 131072
        assert flops == 4 * 128 * 256
        assert flops == 131072
        # elem_bytes = 2 (fp16); (2*128*256 + 256) * 2 = 131584
        elem_bytes = torch.tensor([], dtype=torch.float16).element_size()
        assert elem_bytes == 2
        assert nbytes == (2 * 128 * 256 + 256) * 2
        assert nbytes == 131584

    def test_returns_int_tuple(self):
        op = _RooflineOp(M=4, N=8, dtype=torch.float32)
        flops, nbytes = op.eval_roofline()
        assert isinstance(flops, int)
        assert isinstance(nbytes, int)


class TestSafeEval:
    def test_rejects_call(self):
        with pytest.raises(ValueError) as excinfo:
            _safe_eval("f(1)", {})
        assert "Call" in str(excinfo.value)

    def test_rejects_attribute(self):
        with pytest.raises(ValueError) as excinfo:
            _safe_eval("x.y", {"x": 1})
        assert "Attribute" in str(excinfo.value)

    def test_rejects_subscript(self):
        with pytest.raises(ValueError) as excinfo:
            _safe_eval("x[0]", {"x": 1})
        assert "Subscript" in str(excinfo.value)

    def test_rejects_undefined_name(self):
        with pytest.raises(ValueError) as excinfo:
            _safe_eval("missing + 1", {})
        assert "missing" in str(excinfo.value)

    def test_accepts_basic_arithmetic(self):
        assert _safe_eval("2 + 3 * 4", {}) == 14
        assert _safe_eval("-a + b", {"a": 5, "b": 7}) == 2
        assert _safe_eval("2 ** 10", {}) == 1024

    def test_rejects_bool_literal_true(self):
        """``bool`` subclasses ``int`` in Python; ensure a bare ``True``
        literal is rejected rather than silently treated as ``1``."""
        with pytest.raises(ValueError) as excinfo:
            _safe_eval("True", {})
        assert "bool" in str(excinfo.value)

    def test_rejects_bool_in_binop_left(self):
        """``False + 1`` must not evaluate to ``1``."""
        with pytest.raises(ValueError) as excinfo:
            _safe_eval("False + 1", {})
        assert "bool" in str(excinfo.value)

    def test_rejects_bool_in_binop_right(self):
        """``1 + True`` must not evaluate to ``2``."""
        with pytest.raises(ValueError) as excinfo:
            _safe_eval("1 + True", {})
        assert "bool" in str(excinfo.value)

    def test_rejects_bool_name_binding(self):
        """Name lookup must reject a ``bool`` ctx binding (subclass of int)."""
        with pytest.raises(ValueError) as excinfo:
            _safe_eval("x", {"x": True})
        assert "bool" in str(excinfo.value)
        assert "'x'" in str(excinfo.value)

    def test_rejects_bool_name_binding_in_binop(self):
        """``x + 1`` with ``x=True`` must not evaluate to ``2``."""
        with pytest.raises(ValueError) as excinfo:
            _safe_eval("x + 1", {"x": True})
        assert "bool" in str(excinfo.value)

    def test_rejects_str_name_binding(self):
        """Name lookup must reject a non-numeric (e.g. ``str``) binding."""
        with pytest.raises(ValueError) as excinfo:
            _safe_eval("x", {"x": "3"})
        assert "str" in str(excinfo.value)
        assert "'x'" in str(excinfo.value)


class TestBaseClassStubs:
    def test_infer_output_shapes_raises_not_implemented(self):
        op = _MinimalOp()
        with pytest.raises(NotImplementedError) as excinfo:
            op._infer_output_shapes()
        assert "docs/ops-design.md" in str(excinfo.value)
        assert "_infer_output_shapes" in str(excinfo.value)

    def test_validate_dtypes_raises_not_implemented(self):
        op = _MinimalOp()
        x = torch.empty(1)
        with pytest.raises(NotImplementedError) as excinfo:
            op._validate_dtypes(x)
        assert "docs/ops-design.md" in str(excinfo.value)
        assert "_validate_dtypes" in str(excinfo.value)
