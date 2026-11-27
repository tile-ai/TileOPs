"""VarFwdOp, StdFwdOp and VarMeanFwdOp over a reduced extent off the 256-element alignment:
the zero padding must not reach the Welford statistics."""

import pytest
import torch

from tests.workload_test_base import TestBase
from tileops._ops.reduction.reduce import StdFwdOp, VarFwdOp, VarMeanFwdOp
from workloads.device import run_device
from workloads.numerics import compare_outputs
from workloads.reduction import WelfordNonAlignedCase, _reduction_reference, reduction_verification

_OPS = {"var": VarFwdOp, "std": StdFwdOp, "var_mean": VarMeanFwdOp}


class WelfordNonAlignedTest(WelfordNonAlignedCase, TestBase):
    pass


def _row_cases() -> list:
    """Every op one above the alignment at both 16-bit dtypes is smoke; the shared Welford
    kernel's padding amounts follow on var: heavy (7, 33), moderate (100), one below the
    alignment (255) and a partial second tile (513)."""
    cases = [
        pytest.param(k, 32, 257, dtype, marks=pytest.mark.smoke)
        for k in _OPS
        for dtype in (torch.float16, torch.bfloat16)
    ]
    cases += [
        pytest.param("var", 32, n, torch.float16, marks=pytest.mark.full)
        for n in (7, 33, 100, 255, 513)
    ]
    return cases


@pytest.mark.parametrize("op_kind, m, n, dtype", _row_cases())
def test_welford_non_aligned_rows(op_kind: str, m: int, n: int, dtype: torch.dtype) -> None:
    test = WelfordNonAlignedTest((m, n), dtype, op_kind, correction=1)
    test.check(_OPS[op_kind](dim=-1), *test.gen_inputs())


def _nd_cases() -> list:
    """Leading axes folded into rows, and two reduced axes flattening to 255, 63 and 258."""
    cases = [
        pytest.param(k, (2, 5, 51), [1, 2], False, torch.float16, marks=pytest.mark.smoke)
        for k in _OPS
    ]
    cases += [
        pytest.param("var", (2, 16, 255), -1, False, torch.float16, marks=pytest.mark.smoke),
        pytest.param("var", (4, 7, 9), [1, 2], False, torch.bfloat16, marks=pytest.mark.smoke),
        pytest.param("var", (3, 3, 86), [1, 2], False, torch.float16, marks=pytest.mark.full),
        pytest.param("var_mean", (2, 5, 51), [1, 2], True, torch.float16, marks=pytest.mark.full),
    ]
    return cases


@pytest.mark.parametrize("op_kind, shape, dim, keepdim, dtype", _nd_cases())
def test_welford_non_aligned_axes(
    op_kind: str, shape: tuple, dim, keepdim: bool, dtype: torch.dtype
) -> None:
    x = torch.randn(*shape, dtype=dtype, device=run_device())
    out = _OPS[op_kind](dim=dim, keepdim=keepdim)(x)
    ref = _reduction_reference(
        _OPS[op_kind].__name__, x, dict(dim=dim, keepdim=keepdim, correction=1)
    )
    pairs = zip(out, ref, strict=True) if isinstance(ref, tuple) else [(out, ref)]
    for got, want in pairs:
        assert got.shape == want.shape, f"shape {got.shape} vs ref {want.shape}"
        compare_outputs(got, want, reduction_verification(want.dtype))
