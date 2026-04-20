"""Correctness tests for cumulative ops (cumsum, cumprod).

Covers: CumsumFwdOp, CumprodFwdOp.
Each op computes an inclusive prefix scan along dim=-1 and supports 1D-4D input.
Output has the same shape as input.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class CumulativeBasicFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# TestBase helpers
# ---------------------------------------------------------------------------


class CumulativeTest(TestBase):
    """Parameterized test helper for cumulative ops."""

    def __init__(
        self, m: int, n: int, dtype: torch.dtype, op_kind: str, use_small_range: bool = False
    ):
        self.m = m
        self.n = n
        self.dtype = dtype
        self.op_kind = op_kind
        self.use_small_range = use_small_range

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self.use_small_range:
            # For cumprod, use small values to avoid overflow
            x = torch.rand(self.m, self.n, dtype=self.dtype, device="cuda") * 0.01 + 0.99
        else:
            x = torch.randn(self.m, self.n, dtype=self.dtype, device="cuda")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        if self.op_kind == "cumsum":
            return x_f32.cumsum(dim=-1).to(x.dtype)
        elif self.op_kind == "cumprod":
            return x_f32.cumprod(dim=-1).to(x.dtype)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


# ---------------------------------------------------------------------------
# Helper to get tolerances
# ---------------------------------------------------------------------------


def _tol(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    return {"atol": 1e-2, "rtol": 1e-2}


def _cumprod_tol(dtype: torch.dtype) -> dict:
    """Tolerances for cumprod tests (more numerically sensitive)."""
    if dtype == torch.float32:
        return {"atol": 1e-3, "rtol": 1e-3}
    return {"atol": 5e-2, "rtol": 5e-2}


# ---------------------------------------------------------------------------
# CumsumFwdOp tests
# ---------------------------------------------------------------------------


@CumulativeBasicFixture
def test_cumsum_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumsum import CumsumFwdOp

    test = CumulativeTest(m, n, dtype, "cumsum")
    op = CumsumFwdOp(M=m, N=n, dtype=dtype)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


# ---------------------------------------------------------------------------
# CumprodFwdOp tests
# ---------------------------------------------------------------------------


@CumulativeBasicFixture
def test_cumprod_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumprod import CumprodFwdOp

    test = CumulativeTest(m, n, dtype, "cumprod", use_small_range=True)
    op = CumprodFwdOp(M=m, N=n, dtype=dtype)
    test.check(op, *test.gen_inputs(), **_cumprod_tol(dtype))


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
