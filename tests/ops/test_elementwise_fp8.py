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
    """Every concrete ``Kernel`` subclass must declare SUPPORTED_DTYPES as a
    non-empty tuple that excludes every fp8 dtype.

    The ``None`` default lives on the elementwise template bases
    (``UnaryKernel`` / ``BinaryKernel`` and their float / logical / predicate
    siblings), not on the abstract ``Kernel`` root. Inheriting that default
    silently hides the rejection contract; admitting an fp8 entry would let
    fp8 reach codegen paths that PR-time guards no longer cover.
    """
    import inspect

    import tileops.kernels.elementwise as ew

    abstract = {
        ew.Kernel, ew.BinaryKernel, ew.UnaryKernel, ew.FloatUnaryKernel,
        ew.LogicalUnaryKernel, ew.FloatPredicateKernel,
        ew.FusedGatedKernel, ew.ParametricUnaryKernel,
        ew._AlphaScaledBinaryKernel,
    }
    fp8_dtypes = set(ew._FP8_DTYPES)
    roots = (ew.Kernel,)
    none_offenders = []
    type_offenders = []
    fp8_offenders = []
    for _name, cls in inspect.getmembers(ew, inspect.isclass):
        if cls in abstract:
            continue
        if not any(issubclass(cls, b) for b in roots):
            continue
        supported = getattr(cls, "SUPPORTED_DTYPES", None)
        if supported is None:
            none_offenders.append(cls.__name__)
            continue
        if not isinstance(supported, tuple):
            type_offenders.append((cls.__name__, type(supported).__name__))
            continue
        leaked = [dt for dt in supported if dt in fp8_dtypes]
        if leaked:
            fp8_offenders.append((cls.__name__, leaked))
    assert not none_offenders, (
        f"Concrete kernels with SUPPORTED_DTYPES=None: {none_offenders}"
    )
    assert not type_offenders, (
        f"Concrete kernels with non-tuple SUPPORTED_DTYPES: {type_offenders}"
    )
    assert not fp8_offenders, (
        f"Concrete kernels admitting fp8 in SUPPORTED_DTYPES: {fp8_offenders}"
    )


def _binary_kwargs(dtype):
    return dict(
        N_total=_N, dtype=dtype,
        coalesced_shape=(_N,), a_strides=(1,), b_strides=(1,),
        a_numel=_N, b_numel=_N,
    )


@pytest.mark.smoke
def test_comparison_family_kernel_rejects_fp8():
    """Comparison family (Eq/Lt/Ge representatives) rejects fp8 at the kernel layer."""
    from tileops.kernels.elementwise import (
        EqFwdKernel,
        GeFwdKernel,
        LtFwdKernel,
    )

    for cls in (EqFwdKernel, LtFwdKernel, GeFwdKernel):
        with pytest.raises(ValueError, match="only supports dtypes"):
            cls(**_binary_kwargs(torch.float8_e4m3fn))


@pytest.mark.smoke
def test_pow_kernel_rejects_fp8():
    """PowFwdKernel rejects fp8 (narrowed _FLOAT_DTYPES)."""
    from tileops.kernels.elementwise import PowFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        PowFwdKernel(**_binary_kwargs(torch.float8_e4m3fn))


@pytest.mark.smoke
def test_division_family_kernel_rejects_fp8():
    """Division family (Div/FloorDivide/Remainder) rejects fp8 at the kernel layer."""
    from tileops.kernels.elementwise import (
        DivFwdKernel,
        FloorDivideFwdKernel,
        RemainderFwdKernel,
    )

    for cls in (DivFwdKernel, FloorDivideFwdKernel, RemainderFwdKernel):
        with pytest.raises(ValueError, match="only supports dtypes"):
            cls(**_binary_kwargs(torch.float8_e4m3fn))


@pytest.mark.smoke
def test_lerp_kernel_rejects_fp8():
    """LerpFwdKernel rejects fp8 (narrowed _FLOAT_DTYPES)."""
    from tileops.kernels.elementwise import LerpFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        LerpFwdKernel(**_binary_kwargs(torch.float8_e4m3fn))


@pytest.mark.smoke
def test_maximum_minimum_family_kernel_rejects_fp8():
    """Maximum/Minimum family rejects fp8 at the kernel layer."""
    from tileops.kernels.elementwise import MaximumFwdKernel, MinimumFwdKernel

    for cls in (MaximumFwdKernel, MinimumFwdKernel):
        with pytest.raises(ValueError, match="only supports dtypes"):
            cls(**_binary_kwargs(torch.float8_e4m3fn))


@pytest.mark.smoke
def test_logical_binary_family_kernel_rejects_fp8():
    """LogicalAnd/LogicalOr family rejects fp8 at the kernel layer."""
    from tileops.kernels.elementwise import LogicalAndFwdKernel, LogicalOrFwdKernel

    for cls in (LogicalAndFwdKernel, LogicalOrFwdKernel):
        with pytest.raises(ValueError, match="only supports dtypes"):
            cls(**_binary_kwargs(torch.float8_e4m3fn))


@pytest.mark.smoke
def test_logical_unary_kernel_rejects_fp8():
    """LogicalNotFwdKernel (LogicalUnaryKernel base) rejects fp8."""
    from tileops.kernels.elementwise import LogicalNotFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        LogicalNotFwdKernel(N_total=_N, dtype=torch.float8_e4m3fn)


@pytest.mark.smoke
def test_pow_kernel_rejects_bool():
    """PowFwdKernel (float-only family) rejects bool inputs.

    Companion sentinel to fp8 rejection: ``PowFwdKernel.SUPPORTED_DTYPES``
    is ``_FLOAT_DTYPES``, so int/bool must also raise at the kernel layer.
    """
    from tileops.kernels.elementwise import PowFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        PowFwdKernel(**_binary_kwargs(torch.bool))


@pytest.mark.smoke
def test_lerp_kernel_rejects_int():
    """LerpFwdKernel (float-only family) rejects int32 inputs."""
    from tileops.kernels.elementwise import LerpFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        LerpFwdKernel(**_binary_kwargs(torch.int32))


@pytest.mark.smoke
def test_division_family_kernel_rejects_bool_and_int():
    """Division family (float-only ``_FLOAT_DTYPES``) rejects bool and int."""
    from tileops.kernels.elementwise import DivFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        DivFwdKernel(**_binary_kwargs(torch.bool))
    with pytest.raises(ValueError, match="only supports dtypes"):
        DivFwdKernel(**_binary_kwargs(torch.int32))


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
