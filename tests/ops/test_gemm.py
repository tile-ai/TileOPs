
import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops import GemmOp
from workloads.gemm import GemmTest as _GemmTestWorkload


class GemmTest(_GemmTestWorkload, TestBase):
    def ref_program(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.trans_a:
            a = a.T
        if self.trans_b:
            b = b.T
        return torch.matmul(a, b)


class GemmFixture(FixtureBase):
    PARAMS = [
        ("m, n, k, dtype, trans_a, trans_b, tune", [
            pytest.param(
                1024, 1024, 1024, torch.float16, False, False, False,
                marks=[pytest.mark.smoke, pytest.mark.packaging],
                id="smoke-fp16-square",
            ),
            pytest.param(
                1024, 1024, 1024, torch.bfloat16, False, False, False,
                marks=pytest.mark.smoke,
                id="smoke-bf16-square",
            ),
        ]),
    ]


class GemvBoundaryFixture(FixtureBase):
    """GEMV cases with non-aligned n/k to exercise partial-tile paths."""
    PARAMS = [
        ("n, k, dtype, tune", [
            # lhs_row: m=1, trans_b=True — non-aligned n
            pytest.param(3000, 1024, torch.float16, False, marks=pytest.mark.smoke),
            pytest.param(3000, 1024, torch.bfloat16, False, marks=pytest.mark.smoke),
        ]),
    ]


@GemmFixture
def test_gemm(m: int, n: int, k: int, dtype: torch.dtype, trans_a: bool, trans_b: bool,
              tune: bool) -> None:
    test = GemmTest(m, n, k, dtype, trans_a, trans_b)
    op = GemmOp(m, n, k, trans_a=trans_a, trans_b=trans_b, dtype=dtype, tune=tune)
    if dtype == torch.float16:
        tolerances = {"atol": 1e-3, "rtol": 1e-3}
    else:
        tolerances = {"atol": 1.6e-2, "rtol": 1.6e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


@GemvBoundaryFixture
def test_gemv_boundary_lhs_row(n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    """GEMV lhs_row path (m=1, trans_b=True) with non-aligned n or k."""
    test = GemmTest(1, n, k, dtype, trans_a=False, trans_b=True)
    op = GemmOp(1, n, k, trans_a=False, trans_b=True, dtype=dtype, tune=tune)
    tolerances = {"atol": 1e-2, "rtol": 1e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


@GemvBoundaryFixture
def test_gemv_boundary_rhs_col(n: int, k: int, dtype: torch.dtype, tune: bool) -> None:
    """GEMV rhs_col path (n=1, no transpose) with non-aligned m or k."""
    m = n  # reuse fixture's n as the non-aligned m dimension
    test = GemmTest(m, 1, k, dtype, trans_a=False, trans_b=False)
    op = GemmOp(m, 1, k, trans_a=False, trans_b=False, dtype=dtype, tune=tune)
    tolerances = {"atol": 1e-2, "rtol": 1e-2}
    test.check(op, *test.gen_inputs(), **tolerances)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
