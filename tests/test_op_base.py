"""Tests for tileops.ops.op_base.

Covers:
- ``Op._cache_key`` default behavior and runtime warnings for missing
  subclass overrides when ``_static_axes`` is empty.
- L1 stubs for ``_infer_output_shapes`` / ``_validate_dtypes`` /
  ``eval_roofline`` that raise :class:`NotImplementedError` pointing at the
  design docs. Per docs/roofline.md §4.4.6, the L1 base deliberately does
  not provide a generic roofline evaluator — each op's ``eval_roofline``
  body is emitted by codegen as plain Python.
"""

import warnings
from typing import Dict

import pytest
import torch

from tileops.ops import op_base
from tileops.ops.op_base import Op

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

    def test_eval_roofline_raises_not_implemented(self):
        """L1 eval_roofline is a stub; per docs/roofline.md §4.4.6 each
        concrete op's body is emitted by codegen, not fed through a
        generic L1 evaluator."""
        op = _MinimalOp()
        with pytest.raises(NotImplementedError) as excinfo:
            op.eval_roofline()
        msg = str(excinfo.value)
        assert "docs/roofline.md" in msg
        assert "eval_roofline" in msg
