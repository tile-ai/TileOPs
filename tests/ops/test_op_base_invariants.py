"""Base-class invariant tests for ``Op`` subclasses.

Two invariants gate against the bypass class of bug where a subclass
``__init__`` constructs a kernel inline, dropping the ``kernel_map=``
override and the per-kernel ``supported_archs`` arch check:

1. **Static invariant**: every concrete ``Op`` subclass with a non-empty
   ``default_kernel_map`` must route kernel construction through
   ``self.dispatch_kernel(...)`` — either directly in its own ``__init__``,
   or via ``super().__init__(...)``, or by not overriding ``__init__`` at
   all (in which case it inherits a base that does).

2. **Runtime invariant**: for each touched op, constructing with
   ``kernel_map={name: SentinelKernel}`` results in
   ``op.kernel_map[name] is SentinelKernel``.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
from typing import Any, Callable, Optional

import pytest
import torch

import tileops.ops as ops_pkg
from tileops.ops.op_base import Op

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# Subclass discovery
# ---------------------------------------------------------------------------


def _import_all_op_modules() -> None:
    """Walk ``tileops.ops`` recursively so every Op subclass is imported.

    Tolerate ``ImportError`` only — a missing optional backend should not
    block invariant scanning. Any other exception (``SyntaxError``,
    ``RuntimeError``, ...) is a real bug that would silently drop ops from
    the invariant scan, so re-raise to surface it as a test failure.
    """
    for mod_info in pkgutil.walk_packages(
        ops_pkg.__path__, prefix=ops_pkg.__name__ + ".",
    ):
        try:
            importlib.import_module(mod_info.name)
        except ImportError:
            continue


def _all_concrete_op_subclasses() -> list[type]:
    """Return every concrete (non-abstract) ``Op`` subclass currently loaded."""
    _import_all_op_modules()
    seen: set[type] = set()
    stack: list[type] = list(Op.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
    # Filter out classes that are still abstract (have unimplemented
    # ``default_kernel_map`` / ``forward``).
    concrete: list[type] = []
    for cls in seen:
        if inspect.isabstract(cls):
            continue
        concrete.append(cls)
    return concrete


def _default_kernel_map_node(cls: type) -> Optional[ast.AST]:
    """Return the AST body of ``default_kernel_map`` defined directly on ``cls``.

    Returns ``None`` if the class does not define it (it inherits the
    property) or if the source cannot be parsed.
    """
    if "default_kernel_map" not in cls.__dict__:
        return None
    try:
        src = inspect.getsource(cls.__dict__["default_kernel_map"].fget)
    except (TypeError, OSError):
        return None
    try:
        return ast.parse(inspect.cleandoc(src))
    except SyntaxError:
        return None


def _has_nonempty_default_kernel_map(cls: type) -> bool:
    """Best-effort static check that ``cls.default_kernel_map`` is non-empty.

    The check walks the MRO: a class with ``default_kernel_map`` defined
    directly is inspected; otherwise the search continues up the MRO. If no
    ancestor defines it, the class is treated as having an empty map (the
    abstract base raises ``NotImplementedError``).
    """
    for ancestor in cls.__mro__:
        if "default_kernel_map" not in ancestor.__dict__:
            continue
        try:
            src = inspect.getsource(ancestor.__dict__["default_kernel_map"].fget)
        except (TypeError, OSError):
            return True  # Can't inspect; assume non-empty (conservative).
        # Look for any dict literal with at least one key.
        try:
            tree = ast.parse(inspect.cleandoc(src))
        except SyntaxError:
            return True
        return any(
            isinstance(node, ast.Dict) and node.keys
            for node in ast.walk(tree)
        )
    return False


# ---------------------------------------------------------------------------
# Static invariant: __init__ routes through dispatch_kernel
# ---------------------------------------------------------------------------


def _own_init(cls: type) -> Optional[ast.FunctionDef]:
    """Return the AST FunctionDef of ``__init__`` defined directly on ``cls``.

    Returns ``None`` if the class does not override ``__init__`` (it
    inherits it from a base) or if the source cannot be parsed.
    """
    init = cls.__dict__.get("__init__")
    if init is None or not callable(init):
        return None
    try:
        src = inspect.getsource(init)
    except (TypeError, OSError):
        return None
    try:
        tree = ast.parse(inspect.cleandoc(src))
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            return node
    return None


def _init_routes_through_dispatch(init_node: ast.FunctionDef) -> bool:
    """Return True if ``init_node`` calls ``self.dispatch_kernel(...)`` or
    ``super().__init__(...)`` somewhere in its body.
    """
    for stmt in ast.walk(init_node):
        if not isinstance(stmt, ast.Call):
            continue
        func = stmt.func
        # self.dispatch_kernel(...) — require the call target to be ``self``
        # so ``other.dispatch_kernel(...)`` does not satisfy the invariant.
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "dispatch_kernel"
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            return True
        # super().__init__(...)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "__init__"
            and isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Name)
            and func.value.func.id == "super"
        ):
            return True
    return False


@pytest.mark.parametrize(
    "op_cls",
    [
        cls for cls in _all_concrete_op_subclasses()
        if _has_nonempty_default_kernel_map(cls)
    ],
    ids=lambda c: f"{c.__module__.split('.')[-1]}.{c.__name__}",
)
def test_op_init_routes_through_dispatch_kernel(op_cls: type) -> None:
    """Every concrete Op subclass with a non-empty ``default_kernel_map``
    must route kernel construction through ``self.dispatch_kernel`` (either
    directly or via ``super().__init__``).

    Classes that don't override ``__init__`` inherit a base ``__init__`` and
    are exempt from the check.
    """
    init_node = _own_init(op_cls)
    if init_node is None:
        # Inherits __init__ from a base — base must already comply.
        return
    assert _init_routes_through_dispatch(init_node), (
        f"{op_cls.__module__}.{op_cls.__name__}.__init__ does not call "
        f"self.dispatch_kernel(...) or super().__init__(...). The "
        f"`kernel_map=` constructor override and per-kernel `supported_archs` "
        f"check will be silently dropped."
    )


# ---------------------------------------------------------------------------
# Runtime invariant: kernel_map override is honored
# ---------------------------------------------------------------------------


def _make_sentinel(default_kernel_cls: type) -> type:
    """Return a marker subclass of the default kernel class.

    Subclassing keeps any kernel-side ``SUPPORTED_DTYPES`` /
    ``supported_archs`` filters intact, so construction proceeds with the
    same dtype contract as the default but with a distinct class identity.
    """

    class SentinelKernel(default_kernel_cls):  # type: ignore[misc, valid-type]
        """Marker subclass; identical behavior, distinct identity."""

    SentinelKernel.__name__ = f"Sentinel_{default_kernel_cls.__name__}"
    return SentinelKernel


# Fixture registry: each entry returns a kwargs dict for the op's __init__.
# All ops use a float dtype to avoid the `_IntIdentityUnaryOp` integer
# fallback at `tileops/ops/elementwise.py` (the float kernel cannot be
# instantiated for integer dtypes by design).
_FLOAT_DTYPE = torch.float16


def _unary_kwargs() -> dict[str, Any]:
    return {"N_total": 16, "dtype": _FLOAT_DTYPE}


def _binary_kwargs() -> dict[str, Any]:
    return {"a_shape": (4, 4), "b_shape": (4, 4), "dtype": _FLOAT_DTYPE}


def _fused_gated_kwargs() -> dict[str, Any]:
    return {"M": 4, "N": 4, "dtype": _FLOAT_DTYPE}


# Touched ops: the bypass-fix sites covered by AC-3.
def _prelu_kwargs() -> dict[str, Any]:
    return {"shape": (2, 4), "dtype": _FLOAT_DTYPE, "num_channels": 4}


def _where_kwargs() -> dict[str, Any]:
    return {
        "condition": (4,),
        "input": (4,),
        "other": (4,),
        "dtype": _FLOAT_DTYPE,
    }


def _masked_fill_kwargs() -> dict[str, Any]:
    return {
        "input": (4,),
        "mask": (4,),
        "value": (),
        "dtype": _FLOAT_DTYPE,
    }


def _masked_fill_scalar_kwargs() -> dict[str, Any]:
    return {
        "input": (4,),
        "mask": (4,),
        "value": 0.0,
        "dtype": _FLOAT_DTYPE,
    }


def _alibi_kwargs() -> dict[str, Any]:
    return {"seq_len": 8, "num_heads": 4, "dtype": _FLOAT_DTYPE}


def _sinusoidal_kwargs() -> dict[str, Any]:
    return {"seq_len": 8, "d_model": 16, "dtype": _FLOAT_DTYPE}


# Extra kwargs the issue explicitly enumerates as "touched": the
# bypass sites in elementwise.py that this fix routes through
# `self.dispatch_kernel(kernel_map)`.
_TOUCHED_OPS_KWARGS: dict[str, Callable[[], dict[str, Any]]] = {
    "PreluFwdOp": _prelu_kwargs,
    "WhereFwdOp": _where_kwargs,
    "MaskedFillFwdOp": _masked_fill_kwargs,
    "MaskedFillScalarFwdOp": _masked_fill_scalar_kwargs,
    "AlibiFwdOp": _alibi_kwargs,
    "SinusoidalFwdOp": _sinusoidal_kwargs,
}


def _resolve_touched_op(name: str) -> type:
    import tileops.ops.elementwise as mod
    return getattr(mod, name)


@pytest.mark.parametrize(
    ("op_name", "kwargs_factory"),
    list(_TOUCHED_OPS_KWARGS.items()),
)
def test_touched_op_honors_kernel_map_override(
    op_name: str, kwargs_factory: Callable[[], dict[str, Any]],
) -> None:
    """Constructing a touched op with ``kernel_map={name: Sentinel}`` must
    install the sentinel class onto ``self.kernel_map[name]``.
    """
    op_cls = _resolve_touched_op(op_name)
    # Build a baseline instance to read off the dispatch keys.
    baseline = op_cls(**kwargs_factory())
    overrides: dict[str, type] = {}
    for key, default_kernel_cls in baseline.default_kernel_map.items():
        overrides[key] = _make_sentinel(default_kernel_cls)
    overridden = op_cls(**kwargs_factory(), kernel_map=overrides)
    for key, sentinel_cls in overrides.items():
        assert overridden.kernel_map[key] is sentinel_cls, (
            f"{op_name}: kernel_map override for {key!r} was dropped; "
            f"got {overridden.kernel_map[key]!r}"
        )
