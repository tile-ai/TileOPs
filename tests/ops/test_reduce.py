"""Correctness tests for the 8 basic reduce ops.

Covers: SumFwdOp, MeanFwdOp, AminFwdOp, AmaxFwdOp, ProdFwdOp, StdFwdOp, VarFwdOp, VarMeanFwdOp.
Each op reduces along dim=-1 and supports 1D-4D input.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from workloads.reduce import (
    ProdTest as _ProdTest,
)
from workloads.reduce import (
    StdTest as _StdTest,
)
from workloads.reduce import (
    SumTest as _SumTest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class ReduceBasicFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float16, marks=[pytest.mark.smoke, pytest.mark.packaging]),
                pytest.param(128, 512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class BesselFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype, correction",
            [
                pytest.param(128, 512, torch.float16, 0, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, 0, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.float32, 0, marks=pytest.mark.smoke),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# TestBase helpers — inherit gen_inputs() from workload classes
# ---------------------------------------------------------------------------


class ReduceTest(_SumTest, TestBase):
    """Parameterized test helper for simple reduce ops (sum/mean/amax/amin)."""

    def __init__(
        self, m: int, n: int, dtype: torch.dtype, op_kind: str,
    ):
        super().__init__((m, n), dtype)
        self.op_kind = op_kind

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.float()
        if self.op_kind == "sum":
            return x_f32.sum(dim=-1).to(x.dtype)
        elif self.op_kind == "mean":
            return x_f32.mean(dim=-1).to(x.dtype)
        elif self.op_kind == "amax":
            return x_f32.amax(dim=-1).to(x.dtype)
        elif self.op_kind == "amin":
            return x_f32.amin(dim=-1).to(x.dtype)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


class ProdTest(_ProdTest, TestBase):
    """Parameterized test helper for prod op (uses small-range inputs)."""

    def __init__(self, m: int, n: int, dtype: torch.dtype):
        super().__init__((m, n), dtype)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return x.float().prod(dim=-1).to(x.dtype)


class WelfordTest(_StdTest, TestBase):
    """Test helper for Welford-based ops (std, var, var_mean)."""

    def __init__(self, m: int, n: int, dtype: torch.dtype, op_kind: str, correction: int = 1):
        super().__init__((m, n), dtype)
        self.op_kind = op_kind
        self.correction = correction

    def ref_program(self, x: torch.Tensor) -> object:
        x_f32 = x.float()
        if self.op_kind == "var":
            return x_f32.var(dim=-1, correction=self.correction).to(x.dtype)
        elif self.op_kind == "std":
            return x_f32.std(dim=-1, correction=self.correction).to(x.dtype)
        elif self.op_kind == "var_mean":
            v = x_f32.var(dim=-1, correction=self.correction).to(x.dtype)
            m = x_f32.mean(dim=-1).to(x.dtype)
            return (v, m)
        raise ValueError(f"Unknown op_kind: {self.op_kind}")


# ---------------------------------------------------------------------------
# Helper to get tolerances
# ---------------------------------------------------------------------------


def _tol(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    return {"atol": 1e-2, "rtol": 1e-2}


# ---------------------------------------------------------------------------
# SumFwdOp tests
# ---------------------------------------------------------------------------


@ReduceBasicFixture
def test_sum_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import SumFwdOp

    test = ReduceTest(m, n, dtype, "sum")
    op = SumFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


# ---------------------------------------------------------------------------
# MeanFwdOp tests
# ---------------------------------------------------------------------------


@ReduceBasicFixture
def test_mean_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import MeanFwdOp

    test = ReduceTest(m, n, dtype, "mean")
    op = MeanFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


# ---------------------------------------------------------------------------
# AminFwdOp tests
# ---------------------------------------------------------------------------


@ReduceBasicFixture
def test_amin_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import AminFwdOp

    test = ReduceTest(m, n, dtype, "amin")
    op = AminFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


# ---------------------------------------------------------------------------
# AmaxFwdOp tests
# ---------------------------------------------------------------------------


@ReduceBasicFixture
def test_amax_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import AmaxFwdOp

    test = ReduceTest(m, n, dtype, "amax")
    op = AmaxFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


# ---------------------------------------------------------------------------
# ProdFwdOp tests
# ---------------------------------------------------------------------------


@ReduceBasicFixture
def test_prod_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import ProdFwdOp

    test = ProdTest(m, n, dtype)
    op = ProdFwdOp(dtype=dtype)
    # Prod is more numerically sensitive
    tol = {"atol": 5e-2, "rtol": 5e-2} if dtype != torch.float32 else {"atol": 1e-3, "rtol": 1e-3}
    test.check(op, *test.gen_inputs(), **tol)


# ---------------------------------------------------------------------------
# StdFwdOp tests
# ---------------------------------------------------------------------------


@ReduceBasicFixture
def test_std_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import StdFwdOp

    test = WelfordTest(m, n, dtype, "std", correction=1)
    op = StdFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


@BesselFixture
def test_std_bessel(m: int, n: int, dtype: torch.dtype, correction: int) -> None:
    from tileops.ops.reduction.reduce import StdFwdOp

    test = WelfordTest(m, n, dtype, "std", correction=correction)
    op = StdFwdOp(dtype=dtype, correction=correction)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


# ---------------------------------------------------------------------------
# VarFwdOp tests
# ---------------------------------------------------------------------------


@ReduceBasicFixture
def test_var_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import VarFwdOp

    test = WelfordTest(m, n, dtype, "var", correction=1)
    op = VarFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


@BesselFixture
def test_var_bessel(m: int, n: int, dtype: torch.dtype, correction: int) -> None:
    from tileops.ops.reduction.reduce import VarFwdOp

    test = WelfordTest(m, n, dtype, "var", correction=correction)
    op = VarFwdOp(dtype=dtype, correction=correction)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


# ---------------------------------------------------------------------------
# VarMeanFwdOp tests
# ---------------------------------------------------------------------------


@ReduceBasicFixture
def test_var_mean_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import VarMeanFwdOp

    test = WelfordTest(m, n, dtype, "var_mean", correction=1)
    op = VarMeanFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


@BesselFixture
def test_var_mean_bessel(m: int, n: int, dtype: torch.dtype, correction: int) -> None:
    from tileops.ops.reduction.reduce import VarMeanFwdOp

    test = WelfordTest(m, n, dtype, "var_mean", correction=correction)
    op = VarMeanFwdOp(dtype=dtype, correction=correction)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
