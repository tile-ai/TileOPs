"""Correctness tests for cumulative ops (cumsum, cumprod).

Covers: CumsumOp, CumprodOp.
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
                pytest.param(128, 512, torch.float16, marks=pytest.mark.full),
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.full),
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
                pytest.param(128, 512, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class Cumulative3DFixture(FixtureBase):
    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                pytest.param(2, 64, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 64, 512, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class Cumulative4DFixture(FixtureBase):
    PARAMS = [
        (
            "b0, b1, b2, n, dtype",
            [
                pytest.param(2, 4, 8, 512, torch.float16, marks=pytest.mark.smoke),
                pytest.param(2, 4, 8, 512, torch.bfloat16, marks=pytest.mark.full),
            ],
        ),
    ]


class Cumulative1DFixture(FixtureBase):
    PARAMS = [
        (
            "n, dtype",
            [
                pytest.param(512, torch.float32, marks=pytest.mark.smoke),
                pytest.param(512, torch.float16, marks=pytest.mark.full),
                pytest.param(512, torch.bfloat16, marks=pytest.mark.full),
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
# CumsumOp tests
# ---------------------------------------------------------------------------


@CumulativeBasicFixture
def test_cumsum_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumsum import CumsumOp

    test = CumulativeTest(m, n, dtype, "cumsum")
    op = CumsumOp(M=m, N=n, dtype=dtype)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


@CumulativeNonContigFixture
def test_cumsum_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumsum import CumsumOp

    x_full = torch.randn(m, n * 2, dtype=dtype, device="cuda")
    x = x_full[:, :n]
    op = CumsumOp(M=m, N=n, dtype=dtype)
    ref = x.contiguous().float().cumsum(dim=-1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@Cumulative3DFixture
def test_cumsum_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumsum import CumsumOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    M = batch * seq
    op = CumsumOp(M=M, N=hidden, dtype=dtype)
    ref = x.float().cumsum(dim=-1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"3D max err: {(y - ref).abs().max()}"


@Cumulative4DFixture
def test_cumsum_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumsum import CumsumOp

    x = torch.randn(b0, b1, b2, n, dtype=dtype, device="cuda")
    M = b0 * b1 * b2
    op = CumsumOp(M=M, N=n, dtype=dtype)
    ref = x.float().cumsum(dim=-1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"4D max err: {(y - ref).abs().max()}"


@Cumulative1DFixture
def test_cumsum_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumsum import CumsumOp

    x = torch.randn(n, dtype=dtype, device="cuda")
    op = CumsumOp(M=1, N=n, dtype=dtype)
    ref = x.float().cumsum(dim=-1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"1D cumsum max err: {(y - ref).abs().max()}"


# ---------------------------------------------------------------------------
# CumprodOp tests
# ---------------------------------------------------------------------------


@CumulativeBasicFixture
def test_cumprod_op(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumprod import CumprodOp

    test = CumulativeTest(m, n, dtype, "cumprod", use_small_range=True)
    op = CumprodOp(M=m, N=n, dtype=dtype)
    test.check(op, *test.gen_inputs(), **_cumprod_tol(dtype))


@CumulativeNonContigFixture
def test_cumprod_non_contiguous(m: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumprod import CumprodOp

    x_full = torch.rand(m, n * 2, dtype=dtype, device="cuda") * 0.01 + 0.99
    x = x_full[:, :n]
    op = CumprodOp(M=m, N=n, dtype=dtype)
    ref = x.contiguous().float().cumprod(dim=-1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), f"max err: {(y - ref).abs().max()}"


@Cumulative3DFixture
def test_cumprod_3d(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumprod import CumprodOp

    x = torch.rand(batch, seq, hidden, dtype=dtype, device="cuda") * 0.01 + 0.99
    M = batch * seq
    op = CumprodOp(M=M, N=hidden, dtype=dtype)
    ref = x.float().cumprod(dim=-1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), f"3D cumprod max err: {(y - ref).abs().max()}"


@Cumulative4DFixture
def test_cumprod_4d(b0: int, b1: int, b2: int, n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumprod import CumprodOp

    x = torch.rand(b0, b1, b2, n, dtype=dtype, device="cuda") * 0.01 + 0.99
    M = b0 * b1 * b2
    op = CumprodOp(M=M, N=n, dtype=dtype)
    ref = x.float().cumprod(dim=-1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), f"4D cumprod max err: {(y - ref).abs().max()}"


@Cumulative1DFixture
def test_cumprod_1d(n: int, dtype: torch.dtype) -> None:
    from tileops.ops.reduction.cumprod import CumprodOp

    x = torch.rand(n, dtype=dtype, device="cuda") * 0.01 + 0.99
    op = CumprodOp(M=1, N=n, dtype=dtype)
    ref = x.float().cumprod(dim=-1).to(dtype)
    y = op(x)
    tol = _cumprod_tol(dtype)
    assert torch.allclose(y, ref, **tol), f"1D cumprod max err: {(y - ref).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
