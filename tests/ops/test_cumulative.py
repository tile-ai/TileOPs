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
                pytest.param(256, 4096, torch.float16, marks=pytest.mark.full),
                pytest.param(256, 4096, torch.bfloat16, marks=pytest.mark.full),
                # Non-aligned N (non-pow2)
                pytest.param(128, 300, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 300, torch.bfloat16, marks=pytest.mark.full),
                # Tail-M: M not divisible by block_m
                pytest.param(129, 512, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


class CumulativeNonContigFixture(FixtureBase):
    PARAMS = [
        (
            "m, n, dtype",
            [
                pytest.param(128, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class Cumulative3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 64, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 64, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class Cumulative4DFixture(FixtureBase):
    PARAMS = [
        (
            "b0, b1, b2, n, dtype",
            [
                pytest.param(2, 4, 8, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 4, 8, 512, torch.bfloat16, marks=pytest.mark.smoke),
            ],
        ),
    ]


class Cumulative1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(512, torch.bfloat16, marks=pytest.mark.smoke),
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
    op = CumsumFwdOp(N=n, dtype=dtype)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


@CumulativeNonContigFixture
def test_cumsum_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumsum import CumsumFwdOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]
    op = CumsumFwdOp(N=n, dtype=dtype)
    ref = x.contiguous().float().cumsum(dim=-1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@Cumulative3DFixture
def test_cumsum_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumsum import CumsumFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = CumsumFwdOp(N=hidden, dtype=dtype)
    ref = x.float().cumsum(dim=-1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"3D max err: {(y - ref).abs().max()}"


@Cumulative4DFixture
def test_cumsum_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumsum import CumsumFwdOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    op = CumsumFwdOp(N=n, dtype=dtype)
    ref = x.float().cumsum(dim=-1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"4D max err: {(y - ref).abs().max()}"


@Cumulative1DFixture
def test_cumsum_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumsum import CumsumFwdOp

    x = torch.randn(n, dtype=dtype, device="cuda")
    op = CumsumFwdOp(N=n, dtype=dtype)
    ref = x.float().cumsum(dim=-1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"1D cumsum max err: {(y - ref).abs().max()}"


# ---------------------------------------------------------------------------
# CumprodFwdOp tests
# ---------------------------------------------------------------------------


@CumulativeBasicFixture
def test_cumprod_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumprod import CumprodFwdOp

    test = CumulativeTest(m, n, dtype, "cumprod", use_small_range=True)
    op = CumprodFwdOp(N=n, dtype=dtype)
    test.check(op, *test.gen_inputs(), **_cumprod_tol(dtype))


@CumulativeNonContigFixture
def test_cumprod_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumprod import CumprodFwdOp

    x_full = torch.rand(m, n * 2, dtype=dtype, device="cuda") * 0.01 + 0.99
    x = x_full[:, :n]
    op = CumprodFwdOp(N=n, dtype=dtype)
    ref = x.contiguous().float().cumprod(dim=-1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@Cumulative3DFixture
def test_cumprod_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumprod import CumprodFwdOp

    x = torch.rand(batch, seq, hidden, dtype=dtype, device="cuda") * 0.01 + 0.99
    op = CumprodFwdOp(N=hidden, dtype=dtype)
    ref = x.float().cumprod(dim=-1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), f"3D cumprod max err: {(y - ref).abs().max()}"


@Cumulative4DFixture
def test_cumprod_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumprod import CumprodFwdOp

    x = torch.rand(b0, b1, b2, n, dtype=dtype, device="cuda") * 0.01 + 0.99
    op = CumprodFwdOp(N=n, dtype=dtype)
    ref = x.float().cumprod(dim=-1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), f"4D cumprod max err: {(y - ref).abs().max()}"


@Cumulative1DFixture
def test_cumprod_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumprod import CumprodFwdOp

    x = torch.rand(n, dtype=dtype, device="cuda") * 0.01 + 0.99
    op = CumprodFwdOp(N=n, dtype=dtype)
    ref = x.float().cumprod(dim=-1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), f"1D cumprod max err: {(y - ref).abs().max()}"


class CumulativeDimAxis1Fixture(FixtureBase):
    PARAMS = [
        ("batch, hidden, seq, dtype", [
            pytest.param(2, 512, 256, torch.float16, marks=pytest.mark.smoke),
            pytest.param(2, 512, 256, torch.bfloat16, marks=pytest.mark.smoke),
        ]),
    ]


@CumulativeDimAxis1Fixture
def test_cumsum_dim_axis1(
    batch: int, hidden: int, seq: int, dtype: torch.dtype
) -> None:
    """Cumsum along dim=1 (3D) — exercises movedim choreography in `_run`."""
    from tileops.ops.reduction.cumsum import CumsumFwdOp

    x = torch.randn(batch, hidden, seq, dtype=dtype, device="cuda")
    op = CumsumFwdOp(N=hidden, dtype=dtype, dim=1)
    ref = x.float().cumsum(dim=1).to(dtype)
    y = op(x)
    atol = 1e-2 if dtype == torch.float16 else 1.6e-2
    assert torch.allclose(y, ref, atol=atol, rtol=atol), \
        f"cumsum dim=1 max err: {(y - ref).abs().max()}"


@CumulativeDimAxis1Fixture
def test_cumprod_dim_axis1(
    batch: int, hidden: int, seq: int, dtype: torch.dtype
) -> None:
    """Cumprod along dim=1 (3D) — exercises movedim choreography in `_run`."""
    from tileops.ops.reduction.cumprod import CumprodFwdOp

    # Values close to 1 to avoid over/underflow in cumprod over hidden dim.
    x = torch.rand(batch, hidden, seq, dtype=dtype, device="cuda") * 0.01 + 0.99
    op = CumprodFwdOp(N=hidden, dtype=dtype, dim=1)
    ref = x.float().cumprod(dim=1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), \
        f"cumprod dim=1 max err: {(y - ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
