"""Elementwise and RoPE kernels validate the element type and shape they are built for."""

import pytest
import torch

from tileops.kernels import elementwise as ew
from tileops.kernels.rope import RoPENeoxKernel

_FP8 = torch.float8_e4m3fn
_UNARY = {"N_total": 16}
_BINARY = {"a_shape": (16,), "b_shape": (16,)}


@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    "kernels, shape, dtypes",
    [
        pytest.param(
            (ew.ReluFwdKernel, ew.GeluFwdKernel), _UNARY, (_FP8, torch.int32), id="float-unary"
        ),
        pytest.param((ew.ExpFwdKernel,), _UNARY, (torch.int32,), id="math-unary"),
        pytest.param(
            (ew.LeakyReluFwdKernel, ew.ClampFwdKernel), _UNARY, (torch.int32,), id="independent"
        ),
        pytest.param((ew.IsnanFwdKernel,), _UNARY, (torch.int32,), id="predicate"),
        pytest.param((ew.LogicalNotFwdKernel,), _UNARY, (_FP8,), id="logical-unary"),
        pytest.param((ew.BitwiseNotFwdKernel,), _UNARY, (_FP8, torch.float16), id="bitwise-unary"),
        pytest.param((ew.BitwiseAndFwdKernel,), _BINARY, (_FP8,), id="bitwise-binary"),
        pytest.param(
            (ew.MulFwdKernel, ew.MaximumFwdKernel, ew.MinimumFwdKernel),
            _BINARY,
            (_FP8,),
            id="arith",
        ),
        pytest.param(
            (ew.EqFwdKernel, ew.LtFwdKernel, ew.GeFwdKernel), _BINARY, (_FP8,), id="comparison"
        ),
        pytest.param(
            (ew.LogicalAndFwdKernel, ew.LogicalOrFwdKernel), _BINARY, (_FP8,), id="logical-binary"
        ),
        pytest.param(
            (
                ew.PowFwdKernel,
                ew.LerpFwdKernel,
                ew.DivFwdKernel,
                ew.FloorDivideFwdKernel,
                ew.RemainderFwdKernel,
            ),
            _BINARY,
            (_FP8, torch.bool, torch.int32),
            id="float-binary",
        ),
        pytest.param((RoPENeoxKernel,), {"seq_len": 16, "head_dim": 64}, (torch.int32,), id="rope"),
    ],
)
def test_kernel_refuses_a_dtype_outside_its_family(kernels, shape, dtypes) -> None:
    for kernel in kernels:
        for dtype in dtypes:
            with pytest.raises(ValueError, match="only supports dtypes"):
                kernel(**shape, dtype=dtype)


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


@pytest.mark.cuda_only
@pytest.mark.smoke
def test_sinusoidal_rejects_odd_d_model() -> None:
    """An odd d_model has a dimension with no pair, which the kernel cannot place."""
    from tileops.kernels.elementwise import SinusoidalFwdKernel

    with pytest.raises(ValueError, match="even d_model"):
        SinusoidalFwdKernel(8, 7, torch.float16)
