"""Tests for fp8 dtype rejection in elementwise kernels.

After narrowing `_FLOAT_DTYPES` to drop fp8, the elementwise / rope / dropout
kernels no longer advertise fp8 in `SUPPORTED_DTYPES`. The tests here are
sentinel checks that float and bitwise kernels correctly reject fp8 inputs
at the kernel layer (exercising `SUPPORTED_DTYPES`, not `Op._validate_dtypes`).
"""

import pytest
import torch

_N = 1024 * 16


@pytest.mark.smoke
def test_float_unary_kernel_rejects_fp8():
    """ReluFwdKernel raises ValueError for fp8 (not in narrowed _FLOAT_DTYPES)."""
    from tileops.kernels.elementwise import ReluFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        ReluFwdKernel(N_total=_N, dtype=torch.float8_e4m3fn)


@pytest.mark.smoke
def test_bitwise_kernel_rejects_fp8():
    """BitwiseNotFwdKernel raises ValueError for fp8 (not in _BITWISE_DTYPES)."""
    from tileops.kernels.elementwise import BitwiseNotFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        BitwiseNotFwdKernel(N_total=_N, dtype=torch.float8_e4m3fn)


@pytest.mark.smoke
def test_binary_bitwise_kernel_rejects_fp8():
    """BitwiseAndFwdKernel raises ValueError for fp8 (not in _BITWISE_DTYPES)."""
    from tileops.kernels.elementwise import BitwiseAndFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        BitwiseAndFwdKernel(
            N_total=_N, dtype=torch.float8_e4m3fn,
            coalesced_shape=(_N,), a_strides=(1,), b_strides=(1,),
            a_numel=_N, b_numel=_N,
        )


@pytest.mark.smoke
def test_binary_arith_kernel_rejects_fp8():
    """MulFwdKernel raises ValueError for fp8 (not in _BINARY_FULL_DTYPES).

    Regression sentinel: prevents MulFwdKernel.SUPPORTED_DTYPES from drifting
    back to a dtype set that admits fp8 (e.g. None or _FLOAT_DTYPES superset).
    """
    from tileops.kernels.elementwise import MulFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        MulFwdKernel(
            N_total=_N, dtype=torch.float8_e4m3fn,
            coalesced_shape=(_N,), a_strides=(1,), b_strides=(1,),
            a_numel=_N, b_numel=_N,
        )


@pytest.mark.smoke
def test_no_concrete_kernel_inherits_none_supported_dtypes():
    """AC-1 guard: every concrete ``Kernel`` subclass must declare
    SUPPORTED_DTYPES (directly or via a non-None base). Inheriting the
    ``None`` default from the abstract ``Kernel`` base hides the rejection
    contract and lets fp8 slip through silently.
    """
    import inspect

    import tileops.kernels.elementwise as ew

    abstract = {
        ew.Kernel, ew.BinaryKernel, ew.UnaryKernel, ew.FloatUnaryKernel,
        ew.LogicalUnaryKernel, ew.FloatPredicateKernel,
        ew.FusedGatedKernel, ew.ParametricUnaryKernel,
        ew._AlphaScaledBinaryKernel,
    }
    roots = (ew.Kernel,)
    offenders = []
    for _name, cls in inspect.getmembers(ew, inspect.isclass):
        if cls in abstract:
            continue
        if not any(issubclass(cls, b) for b in roots):
            continue
        if getattr(cls, "SUPPORTED_DTYPES", None) is None:
            offenders.append(cls.__name__)
    assert not offenders, (
        f"Concrete kernels with SUPPORTED_DTYPES=None: {offenders}"
    )


def _binary_kwargs(dtype):
    return dict(
        N_total=_N, dtype=dtype,
        coalesced_shape=(_N,), a_strides=(1,), b_strides=(1,),
        a_numel=_N, b_numel=_N,
    )


@pytest.mark.smoke
def test_comparison_kernel_rejects_fp8():
    """Comparison family (Eq/Ne/Gt/Lt/Ge/Le) rejects fp8 at the kernel layer."""
    from tileops.kernels.elementwise import EqFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        EqFwdKernel(**_binary_kwargs(torch.float8_e4m3fn))


@pytest.mark.smoke
def test_pow_kernel_rejects_fp8():
    """PowFwdKernel rejects fp8 (narrowed _FLOAT_DTYPES)."""
    from tileops.kernels.elementwise import PowFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        PowFwdKernel(**_binary_kwargs(torch.float8_e4m3fn))


@pytest.mark.smoke
def test_division_family_kernel_rejects_fp8():
    """Division family (Div/FloorDivide/Remainder) rejects fp8 at the kernel layer."""
    from tileops.kernels.elementwise import DivFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        DivFwdKernel(**_binary_kwargs(torch.float8_e4m3fn))


@pytest.mark.smoke
def test_lerp_kernel_rejects_fp8():
    """LerpFwdKernel rejects fp8 (narrowed _FLOAT_DTYPES)."""
    from tileops.kernels.elementwise import LerpFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        LerpFwdKernel(**_binary_kwargs(torch.float8_e4m3fn))


@pytest.mark.smoke
def test_maximum_minimum_kernel_rejects_fp8():
    """Maximum/Minimum family rejects fp8 at the kernel layer."""
    from tileops.kernels.elementwise import MaximumFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        MaximumFwdKernel(**_binary_kwargs(torch.float8_e4m3fn))


@pytest.mark.smoke
def test_logical_binary_kernel_rejects_fp8():
    """LogicalAnd/Or family rejects fp8 at the kernel layer."""
    from tileops.kernels.elementwise import LogicalAndFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        LogicalAndFwdKernel(**_binary_kwargs(torch.float8_e4m3fn))


@pytest.mark.smoke
def test_logical_unary_kernel_rejects_fp8():
    """LogicalNotFwdKernel (LogicalUnaryKernel base) rejects fp8."""
    from tileops.kernels.elementwise import LogicalNotFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        LogicalNotFwdKernel(N_total=_N, dtype=torch.float8_e4m3fn)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
