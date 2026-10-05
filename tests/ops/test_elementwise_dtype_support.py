"""The dtypes the elementwise kernels reject at the kernel layer (``SUPPORTED_DTYPES``, not
``Op._validate_dtypes``), and the dtypes ``WhereFwdOp`` accepts and rejects.
"""

import pytest
import torch

from workloads.device import run_device, run_device_available
from workloads.elementwise import ElementwiseWorkload
from workloads.numerics import compare_outputs


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_float_unary_kernel_rejects_fp8():
    """ReluFwdKernel raises ValueError for fp8 (not in narrowed _FLOAT_DTYPES)."""
    numel = 1024 * 16
    from tileops.kernels.elementwise import ReluFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        ReluFwdKernel(N_total=numel, dtype=torch.float8_e4m3fn)


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_bitwise_kernel_rejects_fp8():
    """BitwiseNotFwdKernel raises ValueError for fp8 (not in _BITWISE_DTYPES)."""
    numel = 1024 * 16
    from tileops.kernels.elementwise import BitwiseNotFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        BitwiseNotFwdKernel(N_total=numel, dtype=torch.float8_e4m3fn)


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_binary_bitwise_kernel_rejects_fp8():
    """BitwiseAndFwdKernel raises ValueError for fp8 (not in _BITWISE_DTYPES)."""
    numel = 1024 * 16
    from tileops.kernels.elementwise import BitwiseAndFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        BitwiseAndFwdKernel(
            a_shape=(numel,),
            b_shape=(numel,),
            dtype=torch.float8_e4m3fn,
        )


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_binary_arith_kernel_rejects_fp8():
    """MulFwdKernel raises ValueError for fp8 (not in _BINARY_FULL_DTYPES).

    Regression sentinel: prevents MulFwdKernel.SUPPORTED_DTYPES from drifting
    back to a dtype set that admits fp8 (e.g. None or _FLOAT_DTYPES superset).
    """
    numel = 1024 * 16
    from tileops.kernels.elementwise import MulFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        MulFwdKernel(
            a_shape=(numel,),
            b_shape=(numel,),
            dtype=torch.float8_e4m3fn,
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
    import importlib
    import inspect
    import pkgutil

    import tileops.kernels.elementwise as ew
    from tileops.kernels.elementwise._dtype import _FP8_DTYPES
    from tileops.kernels.kernel_base import Kernel

    fp8_dtypes = set(_FP8_DTYPES)
    none_offenders = []
    type_offenders = []
    empty_offenders = []
    fp8_offenders = []
    # Walk the package's modules, not the names ``__init__`` re-exports, so a kernel
    # left out of ``__all__`` cannot drop out of this audit. The ``FwdKernel`` /
    # ``BwdKernel`` suffix is what separates a concrete kernel from a template base,
    # which keeps the filter stable as new templates appear.
    candidates = {}
    for module in pkgutil.iter_modules(ew.__path__, ew.__name__ + "."):
        for name, obj in inspect.getmembers(importlib.import_module(module.name), inspect.isclass):
            candidates[name] = obj
    for cls_name, cls in sorted(candidates.items()):
        if not issubclass(cls, Kernel):
            continue
        if not (cls_name.endswith("FwdKernel") or cls_name.endswith("BwdKernel")):
            continue
        supported = getattr(cls, "SUPPORTED_DTYPES", None)
        if supported is None:
            none_offenders.append(cls.__name__)
            continue
        if not isinstance(supported, tuple):
            type_offenders.append((cls.__name__, type(supported).__name__))
            continue
        if len(supported) == 0:
            empty_offenders.append(cls.__name__)
            continue
        leaked = [dt for dt in supported if dt in fp8_dtypes]
        if leaked:
            fp8_offenders.append((cls.__name__, leaked))
    assert not none_offenders, f"Concrete kernels with SUPPORTED_DTYPES=None: {none_offenders}"
    assert not type_offenders, f"Concrete kernels with non-tuple SUPPORTED_DTYPES: {type_offenders}"
    assert not empty_offenders, (
        f"Concrete kernels with empty SUPPORTED_DTYPES tuple: {empty_offenders}"
    )
    assert not fp8_offenders, f"Concrete kernels admitting fp8 in SUPPORTED_DTYPES: {fp8_offenders}"


def _binary_kwargs(dtype):
    numel = 1024 * 16
    return {"a_shape": (numel,), "b_shape": (numel,), "dtype": dtype}


@pytest.mark.cuda_only
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


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_pow_kernel_rejects_fp8():
    """PowFwdKernel rejects fp8 (narrowed _FLOAT_DTYPES)."""
    from tileops.kernels.elementwise import PowFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        PowFwdKernel(**_binary_kwargs(torch.float8_e4m3fn))


@pytest.mark.cuda_only
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


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_lerp_kernel_rejects_fp8():
    """LerpFwdKernel rejects fp8 (narrowed _FLOAT_DTYPES)."""
    from tileops.kernels.elementwise import LerpFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        LerpFwdKernel(**_binary_kwargs(torch.float8_e4m3fn))


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_maximum_minimum_family_kernel_rejects_fp8():
    """Maximum/Minimum family rejects fp8 at the kernel layer."""
    from tileops.kernels.elementwise import MaximumFwdKernel, MinimumFwdKernel

    for cls in (MaximumFwdKernel, MinimumFwdKernel):
        with pytest.raises(ValueError, match="only supports dtypes"):
            cls(**_binary_kwargs(torch.float8_e4m3fn))


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_logical_binary_family_kernel_rejects_fp8():
    """LogicalAnd/LogicalOr family rejects fp8 at the kernel layer."""
    from tileops.kernels.elementwise import LogicalAndFwdKernel, LogicalOrFwdKernel

    for cls in (LogicalAndFwdKernel, LogicalOrFwdKernel):
        with pytest.raises(ValueError, match="only supports dtypes"):
            cls(**_binary_kwargs(torch.float8_e4m3fn))


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_logical_unary_kernel_rejects_fp8():
    """LogicalNotFwdKernel (LogicalUnaryKernel base) rejects fp8."""
    numel = 1024 * 16
    from tileops.kernels.elementwise import LogicalNotFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        LogicalNotFwdKernel(N_total=numel, dtype=torch.float8_e4m3fn)


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_pow_kernel_rejects_bool_and_int():
    """PowFwdKernel (float-only family) rejects both bool and int inputs.

    Companion sentinel to fp8 rejection: ``PowFwdKernel.SUPPORTED_DTYPES``
    is ``_FLOAT_DTYPES``, so int and bool must also raise at the kernel layer.
    """
    from tileops.kernels.elementwise import PowFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        PowFwdKernel(**_binary_kwargs(torch.bool))
    with pytest.raises(ValueError, match="only supports dtypes"):
        PowFwdKernel(**_binary_kwargs(torch.int32))


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_lerp_kernel_rejects_int():
    """LerpFwdKernel (float-only family) rejects int32 inputs."""
    from tileops.kernels.elementwise import LerpFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        LerpFwdKernel(**_binary_kwargs(torch.int32))


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_division_family_kernel_rejects_bool_and_int():
    """Division family (Div/FloorDivide/Remainder, float-only ``_FLOAT_DTYPES``)
    rejects bool and int at the kernel layer."""
    from tileops.kernels.elementwise import (
        DivFwdKernel,
        FloorDivideFwdKernel,
        RemainderFwdKernel,
    )

    for cls in (DivFwdKernel, FloorDivideFwdKernel, RemainderFwdKernel):
        with pytest.raises(ValueError, match="only supports dtypes"):
            cls(**_binary_kwargs(torch.bool))
        with pytest.raises(ValueError, match="only supports dtypes"):
            cls(**_binary_kwargs(torch.int32))


# The independent elementwise ops reject fp8 and accept the manifest's non-fp8 dtypes.
@pytest.mark.smoke
@pytest.mark.parametrize(
    "bad_dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2],
)
def test_where_rejects_fp8_dtype(bad_dtype: torch.dtype) -> None:
    """WhereFwdOp must reject fp8 dtypes (manifest contract).

    The element type arrives with the tensors, so the rejection does too.
    """
    from tileops.ops.elementwise import WhereFwdOp

    shape = (4, 8)
    op = WhereFwdOp()
    cond = torch.zeros(shape, device=run_device(), dtype=torch.bool)
    x = torch.zeros(shape, device=run_device()).to(bad_dtype)
    with pytest.raises((ValueError, TypeError)):
        op(cond, x, x)


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
def test_where_accepts_manifest_dtypes(dtype: torch.dtype) -> None:
    """WhereFwdOp constructs and runs for every manifest-declared dtype."""
    from tileops.ops.elementwise import WhereFwdOp

    shape = (4, 8)
    cond = torch.randint(0, 2, shape, device=run_device()).bool()
    inp = torch.randn(shape, device=run_device(), dtype=dtype)
    other = torch.randn(shape, device=run_device(), dtype=dtype)
    op = WhereFwdOp()
    out = op(cond, inp, other)
    ref = torch.where(cond, inp, other)
    compare_outputs(
        out,
        ref,
        ElementwiseWorkload(type(op).__name__, (cond, inp, other)).verification(
            *(cond, inp, other)
        ),
    )
