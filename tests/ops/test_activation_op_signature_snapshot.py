"""Inspect-based signature snapshot for the activation Op family.

Pins the exact ``__init__`` and ``forward`` signatures (parameter
names, kinds, defaults, type annotations) for the ten activation Ops
covered by the shared base/mixin refactor. Any change here means a
public-API change for these Ops; review must be deliberate.

The snapshot is the canonical pre-refactor surface captured against
the live classes; it doubles as the AC-3 regression gate.
"""

import inspect

import pytest

# Snapshot of the constructor signature ``str`` (excluding ``self``)
# for each activation Op class, captured against the pre-refactor
# implementation. Both keys and ordering are load-bearing.
_INIT_SIGNATURES = {
    "ReluFwdOp": (
        "(N_total: int, dtype: torch.dtype, *, "
        "strategy: Optional[str] = None, "
        "kernel_map: Optional[Dict[str, tileops.kernels.kernel_base.Kernel]] = None, "
        "tune: bool = False, inplace: bool = False)"
    ),
    "SiluFwdOp": (
        "(N_total: int, dtype: torch.dtype, *, "
        "strategy: Optional[str] = None, "
        "kernel_map: Optional[Dict[str, tileops.kernels.kernel_base.Kernel]] = None, "
        "tune: bool = False, inplace: bool = False)"
    ),
    "HardswishFwdOp": (
        "(N_total: int, dtype: torch.dtype, *, "
        "strategy: Optional[str] = None, "
        "kernel_map: Optional[Dict[str, tileops.kernels.kernel_base.Kernel]] = None, "
        "tune: bool = False, inplace: bool = False)"
    ),
    "HardsigmoidFwdOp": (
        "(N_total: int, dtype: torch.dtype, *, "
        "strategy: Optional[str] = None, "
        "kernel_map: Optional[Dict[str, tileops.kernels.kernel_base.Kernel]] = None, "
        "tune: bool = False, inplace: bool = False)"
    ),
    "MishFwdOp": (
        "(N_total: int, dtype: torch.dtype, *, "
        "strategy: Optional[str] = None, "
        "kernel_map: Optional[Dict[str, tileops.kernels.kernel_base.Kernel]] = None, "
        "tune: bool = False, inplace: bool = False)"
    ),
    "SeluFwdOp": (
        "(N_total: int, dtype: torch.dtype, *, "
        "strategy: Optional[str] = None, "
        "kernel_map: Optional[Dict[str, tileops.kernels.kernel_base.Kernel]] = None, "
        "tune: bool = False, inplace: bool = False)"
    ),
    "LeakyReluFwdOp": (
        "(N_total: int, dtype: torch.dtype, "
        "negative_slope: float = 0.01, *, "
        "kernel_map: Optional[Dict[str, tileops.kernels.kernel_base.Kernel]] = None, "
        "tune: bool = False, inplace: bool = False)"
    ),
    "EluFwdOp": (
        "(N_total: int, dtype: torch.dtype, alpha: float = 1.0, *, "
        "kernel_map: Optional[Dict[str, tileops.kernels.kernel_base.Kernel]] = None, "
        "tune: bool = False, inplace: bool = False)"
    ),
    "HardtanhFwdOp": (
        "(N_total: int, dtype: torch.dtype, "
        "min_val: float = -1.0, max_val: float = 1.0, *, "
        "kernel_map: Optional[Dict[str, tileops.kernels.kernel_base.Kernel]] = None, "
        "tune: bool = False, inplace: bool = False)"
    ),
    "SoftplusFwdOp": (
        "(N_total: int, dtype: torch.dtype, "
        "beta: float = 1.0, threshold: float = 20.0, *, "
        "kernel_map: Optional[Dict[str, tileops.kernels.kernel_base.Kernel]] = None, "
        "tune: bool = False)"
    ),
}


_FORWARD_SIGNATURE = "(input: torch.Tensor) -> torch.Tensor"


def _stringify_init(cls: type) -> str:
    sig = inspect.signature(cls.__init__)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    return str(sig.replace(parameters=params))


def _stringify_forward(cls: type) -> str:
    sig = inspect.signature(cls.forward)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    return str(sig.replace(parameters=params))


@pytest.mark.smoke
@pytest.mark.parametrize("op_name, expected_sig", list(_INIT_SIGNATURES.items()))
def test_init_signature_matches_snapshot(op_name: str, expected_sig: str) -> None:
    """Constructor signature must remain bit-identical across the refactor."""
    import tileops.ops.elementwise as mod

    cls = getattr(mod, op_name)
    actual = _stringify_init(cls)
    assert actual == expected_sig, (
        f"{op_name}.__init__ drift:\n  expected: {expected_sig}\n  actual:   {actual}"
    )


@pytest.mark.smoke
@pytest.mark.parametrize("op_name", list(_INIT_SIGNATURES.keys()))
def test_forward_signature_matches_snapshot(op_name: str) -> None:
    """Forward signature must remain ``(input: Tensor) -> Tensor``."""
    import tileops.ops.elementwise as mod

    cls = getattr(mod, op_name)
    actual = _stringify_forward(cls)
    assert actual == _FORWARD_SIGNATURE, (
        f"{op_name}.forward drift:\n  expected: {_FORWARD_SIGNATURE}\n  actual:   {actual}"
    )


@pytest.mark.smoke
@pytest.mark.parametrize("op_name", list(_INIT_SIGNATURES.keys()))
def test_kwarg_kinds_keyword_only(op_name: str) -> None:
    """``kernel_map``, ``tune``, ``inplace`` must be keyword-only on each Op.

    Catches accidental promotion to positional-or-keyword (which would
    silently widen the public API).
    """
    import tileops.ops.elementwise as mod

    cls = getattr(mod, op_name)
    params = inspect.signature(cls.__init__).parameters
    kw_only_required = ["kernel_map", "tune"]
    if op_name != "SoftplusFwdOp":
        kw_only_required.append("inplace")
    for kw in kw_only_required:
        assert kw in params, f"{op_name}.__init__ missing {kw!r}"
        assert params[kw].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{op_name}.__init__ {kw!r} kind is "
            f"{params[kw].kind!r}, expected KEYWORD_ONLY"
        )
