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
                pytest.param(256, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(256, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-aligned N
                pytest.param(128, 300, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 300, torch.bfloat16, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(129, 512, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class ReduceBasicFullFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 512, torch.float32, marks=pytest.mark.full),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(256, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(256, 4096, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(128, 300, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 300, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(129, 512, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class ReduceTiledFixture(FixtureBase):
    """Large-N cases that exercise the tiled reduce path (N > MAX_SINGLE_TILE_COLS).

    One representative op per kernel family: sum (simple), prod, var (welford).
    """

    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(64, 32768, torch.float16, marks=pytest.mark.full),
                pytest.param(64, 32768, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(64, 32768, torch.float32, marks=pytest.mark.full),
                pytest.param(64, 32769, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class ReduceNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(128, 512, torch.float32, marks=pytest.mark.full),
            ],
        ),
    ]


class Reduce3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 64, 512, torch.float16, marks=pytest.mark.full),
                pytest.param(2, 64, 512, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(2, 64, 512, torch.float32, marks=pytest.mark.full),
            ],
        ),
    ]


class Reduce4DFixture(FixtureBase):
    PARAMS = [
        (
            "b0, b1, b2, n, dtype",
            [
                pytest.param(2, 4, 8, 512, torch.float16, marks=pytest.mark.full),
                pytest.param(2, 4, 8, 512, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(2, 4, 8, 512, torch.float32, marks=pytest.mark.full),
            ],
        ),
    ]


class Reduce1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(512, torch.float16, marks=pytest.mark.full),
                pytest.param(512, torch.float32, marks=pytest.mark.full),
                pytest.param(512, torch.bfloat16, marks=pytest.mark.full),
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
                pytest.param(128, 512, torch.float16, 1, marks=pytest.mark.full),
                pytest.param(128, 512, torch.bfloat16, 1, marks=pytest.mark.full),
                pytest.param(128, 512, torch.float32, 1, marks=pytest.mark.full),
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


@ReduceTiledFixture
def test_sum_tiled(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import SumFwdOp

    test = ReduceTest(m, n, dtype, "sum")
    op = SumFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


@ReduceTiledFixture
def test_prod_tiled(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import ProdFwdOp

    test = ProdTest(m, n, dtype)
    op = ProdFwdOp(dtype=dtype)
    tol = {"atol": 5e-2, "rtol": 5e-2} if dtype != torch.float32 else {"atol": 1e-3, "rtol": 1e-3}
    test.check(op, *test.gen_inputs(), **tol)


@ReduceTiledFixture
def test_var_tiled(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import VarFwdOp

    test = WelfordTest(m, n, dtype, "var", correction=1)
    op = VarFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


@ReduceNonContigFixture
def test_sum_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import SumFwdOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]
    op = SumFwdOp(dtype=dtype)
    ref = x.contiguous().float().sum(dim=-1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@Reduce3DFixture
def test_sum_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = SumFwdOp(dtype=dtype)
    ref = x.float().sum(dim=-1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"3D max err: {(y - ref).abs().max()}"


@Reduce4DFixture
def test_sum_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = SumFwdOp(dtype=dtype)
    ref = x.float().sum(dim=-1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"4D max err: {(y - ref).abs().max()}"


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


# ---------------------------------------------------------------------------
# Multi-dim tests for non-contiguous (3D) for Welford ops
# ---------------------------------------------------------------------------


@Reduce3DFixture
def test_var_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import VarFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = VarFwdOp(dtype=dtype)
    ref = x.float().var(dim=-1, correction=1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"3D var max err: {(y - ref).abs().max()}"


@Reduce3DFixture
def test_std_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import StdFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = StdFwdOp(dtype=dtype)
    ref = x.float().std(dim=-1, correction=1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"3D std max err: {(y - ref).abs().max()}"


# ---------------------------------------------------------------------------
# 1D input tests (F004)
# ---------------------------------------------------------------------------


@Reduce1DFixture
def test_sum_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.randn(n, dtype=dtype, device="cuda")
    op = SumFwdOp(dtype=dtype)
    ref = x.float().sum(dim=-1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y.view_as(ref), ref, **tol), (
        f"1D sum max err: {(y.view_as(ref) - ref).abs().max()}"
    )


@Reduce1DFixture
def test_var_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import VarFwdOp

    x = torch.randn(n, dtype=dtype, device="cuda")
    op = VarFwdOp(dtype=dtype)
    ref = x.float().var(dim=-1, correction=1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y.view_as(ref), ref, **tol), (
        f"1D var max err: {(y.view_as(ref) - ref).abs().max()}"
    )


# ---------------------------------------------------------------------------
# Non-contiguous tests for Welford ops (F005)
# ---------------------------------------------------------------------------


@ReduceNonContigFixture
def test_var_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import VarFwdOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]
    op = VarFwdOp(dtype=dtype)
    ref = x.contiguous().float().var(dim=-1, correction=1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"non-contig var max err: {(y - ref).abs().max()}"


@ReduceNonContigFixture
def test_std_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.reduce import StdFwdOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]
    op = StdFwdOp(dtype=dtype)
    ref = x.contiguous().float().std(dim=-1, correction=1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"non-contig std max err: {(y - ref).abs().max()}"


# ---------------------------------------------------------------------------
# Spec-conformant tests (dim + keepdim interface)
# ---------------------------------------------------------------------------


class SpecReduceFixture(FixtureBase):
    PARAMS = [
        (
            "shape, dim, keepdim, dtype",
            [
                pytest.param((128, 512), -1, False, torch.float16, marks=pytest.mark.full),
                pytest.param((128, 512), -1, False, torch.bfloat16, marks=pytest.mark.full),
                pytest.param((128, 512), -1, False, torch.float32, marks=pytest.mark.full),
                pytest.param((128, 512), -1, True, torch.float16, marks=pytest.mark.full),
                pytest.param((4, 32, 512), 0, False, torch.float16, marks=pytest.mark.full),
                pytest.param((4, 32, 512), 1, False, torch.float16, marks=pytest.mark.full),
                pytest.param((4, 32, 512), -1, True, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


@ReduceBasicFullFixture
def test_sum_spec_basic(m: int, n: int, dtype: torch.dtype) -> None:
    """Spec interface: SumFwdOp(dtype=..., dim=-1) on 2D input, multiple dtypes."""
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.randn(m, n, dtype=dtype, device="cuda")
    op = SumFwdOp(dtype=dtype, dim=-1)
    ref = torch.sum(x.float(), dim=-1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"spec basic max err: {(y - ref).abs().max()}"


@SpecReduceFixture
def test_sum_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: reduction along arbitrary dim (0, 1, -1) for 2D/3D tensors."""
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = SumFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
    ref = torch.sum(x.float(), dim=dim, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"spec dim={dim} max err: {(y - ref).abs().max()}"


@SpecReduceFixture
def test_sum_spec_keepdim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: keepdim=True preserves the reduced dimension as size 1."""
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    # Force keepdim=True regardless of fixture param to specifically test shape preservation
    op = SumFwdOp(dtype=dtype, dim=dim, keepdim=True)
    ref = torch.sum(x.float(), dim=dim, keepdim=True).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"keepdim shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"spec keepdim max err: {(y - ref).abs().max()}"


@Reduce1DFixture
def test_sum_spec_1d(n: int, dtype: torch.dtype) -> None:
    """Spec interface: 1D input reduces to scalar."""
    from tileops.ops.reduction.reduce import SumFwdOp

    x = torch.randn(n, dtype=dtype, device="cuda")
    op = SumFwdOp(dtype=dtype, dim=-1)
    ref = torch.sum(x.float(), dim=-1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y.view_as(ref), ref, **tol), (
        f"spec 1D max err: {(y.view_as(ref) - ref).abs().max()}"
    )


@SpecReduceFixture
def test_mean_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: MeanFwdOp with dim + keepdim."""
    from tileops.ops.reduction.reduce import MeanFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = MeanFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
    ref = torch.mean(x.float(), dim=dim, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"mean spec max err: {(y - ref).abs().max()}"


@SpecReduceFixture
def test_amax_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: AmaxFwdOp with dim + keepdim."""
    from tileops.ops.reduction.reduce import AmaxFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = AmaxFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
    ref = torch.amax(x.float(), dim=dim, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"amax spec max err: {(y - ref).abs().max()}"


@SpecReduceFixture
def test_amin_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: AminFwdOp with dim + keepdim."""
    from tileops.ops.reduction.reduce import AminFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = AminFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
    ref = torch.amin(x.float(), dim=dim, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"amin spec max err: {(y - ref).abs().max()}"


@SpecReduceFixture
def test_prod_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: ProdFwdOp with dim + keepdim."""
    from tileops.ops.reduction.reduce import ProdFwdOp

    x = torch.rand(*shape, dtype=dtype, device="cuda") * 0.01 + 0.99
    op = ProdFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
    ref = torch.prod(x.float(), dim=dim, keepdim=keepdim).to(dtype)
    y = op(x)
    tol = {"atol": 5e-2, "rtol": 5e-2} if dtype != torch.float32 else {"atol": 1e-3, "rtol": 1e-3}
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"prod spec max err: {(y - ref).abs().max()}"


@SpecReduceFixture
def test_var_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: VarFwdOp with dim + keepdim + correction."""
    from tileops.ops.reduction.reduce import VarFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = VarFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
    ref = torch.var(x.float(), dim=dim, keepdim=keepdim, correction=1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"var spec max err: {(y - ref).abs().max()}"


@SpecReduceFixture
def test_std_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: StdFwdOp with dim + keepdim + correction."""
    from tileops.ops.reduction.reduce import StdFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = StdFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
    ref = torch.std(x.float(), dim=dim, keepdim=keepdim, correction=1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), f"std spec max err: {(y - ref).abs().max()}"


@SpecReduceFixture
def test_var_mean_spec_dim(shape: tuple, dim: int, keepdim: bool, dtype: torch.dtype) -> None:
    """Spec interface: VarMeanFwdOp with dim + keepdim."""
    from tileops.ops.reduction.reduce import VarMeanFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = VarMeanFwdOp(dtype=dtype, dim=dim, keepdim=keepdim)
    ref_var = torch.var(x.float(), dim=dim, keepdim=keepdim, correction=1).to(dtype)
    ref_mean = torch.mean(x.float(), dim=dim, keepdim=keepdim).to(dtype)
    var_out, mean_out = op(x)
    tol = _tol(dtype)
    assert var_out.shape == ref_var.shape, f"var shape mismatch: {var_out.shape} vs {ref_var.shape}"
    assert mean_out.shape == ref_mean.shape, "mean shape mismatch"
    assert torch.allclose(var_out, ref_var, **tol), f"var_mean spec var err: {(var_out - ref_var).abs().max()}"
    assert torch.allclose(mean_out, ref_mean, **tol), f"var_mean spec mean err: {(mean_out - ref_mean).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
