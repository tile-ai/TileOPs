"""Correctness tests for Welford ops when N triggers zero-padding.

Validates VarFwdOp, StdFwdOp, and VarMeanFwdOp with reduction dimensions
that are NOT multiples of 256 (the default alignment). Zero-padded elements
must not corrupt the Welford online statistics.

Covers both single-dim (dim=-1) and multi-dim cases where the flattened
reduction dimension is non-aligned.
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class WelfordNonAlignedTest(TestBase):
    """Test helper for Welford ops with non-aligned N values."""

    def __init__(
        self,
        shape: tuple,
        dtype: torch.dtype,
        op_kind: str,
        correction: int = 1,
    ):
        self.shape = shape
        self.dtype = dtype
        self.op_kind = op_kind
        self.correction = correction

    def gen_inputs(self) -> tuple[torch.Tensor]:
        x = torch.randn(*self.shape, dtype=self.dtype, device="cuda")
        return (x,)

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


def _tol(dtype: torch.dtype) -> dict:
    if dtype == torch.float32:
        return {"atol": 1e-4, "rtol": 1e-4}
    return {"atol": 1e-2, "rtol": 1e-2}


# ---------------------------------------------------------------------------
# Fixtures — non-aligned N values (not multiples of 256)
# ---------------------------------------------------------------------------

# N values chosen to exercise zero-padding edge cases:
#   7   — small prime, heavy padding
#   100 — non-power-of-two, moderate padding
#   255 — one below alignment boundary
#   257 — one above alignment boundary
#   513 — one above 2*256, second tile partially filled
#   33  — small non-power-of-two, odd padding amount
_NON_ALIGNED_N = [7, 33, 100, 255, 257, 513]


class WelfordNonAlignedFixture(FixtureBase):
    """2D input with non-aligned N values."""

    PARAMS = [
        (
            "m, n, dtype",
            [
                # One representative smoke case (one above alignment boundary)
                pytest.param(
                    32, 257, torch.float16,
                    marks=pytest.mark.smoke,
                    id="m32_n257_fp16",
                ),
            ]
            + [
                pytest.param(
                    32, n, torch.float16,
                    marks=pytest.mark.full,
                    id=f"m32_n{n}_fp16",
                )
                for n in _NON_ALIGNED_N
                if n != 257
            ]
            + [
                pytest.param(
                    32, n, torch.bfloat16,
                    marks=pytest.mark.full,
                    id=f"m32_n{n}_bf16",
                )
                for n in _NON_ALIGNED_N
            ],
        ),
    ]


class WelfordNonAligned3DFixture(FixtureBase):
    """3D input where the last dim (reduction dim) is non-aligned."""

    PARAMS = [
        (
            "batch, seq, hidden, dtype",
            [
                # One representative smoke case (one below alignment boundary)
                pytest.param(
                    2, 16, 255, torch.float16,
                    marks=pytest.mark.smoke,
                    id="b2_s16_h255_fp16",
                ),
            ]
            + [
                pytest.param(
                    2, 16, n, torch.float16,
                    marks=pytest.mark.full,
                    id=f"b2_s16_h{n}_fp16",
                )
                for n in [7, 100, 257]
            ]
            + [
                pytest.param(
                    2, 16, n, torch.bfloat16,
                    marks=pytest.mark.full,
                    id=f"b2_s16_h{n}_bf16",
                )
                for n in [7, 255, 257]
            ],
        ),
    ]


class WelfordNonAlignedMultiDimFixture(FixtureBase):
    """Multi-dim reduction where flattened reduction size is non-aligned."""

    PARAMS = [
        (
            "shape, dims, keepdim, dtype",
            [
                # One representative smoke case (flattened N = 5*51 = 255)
                pytest.param(
                    (2, 5, 51), [1, 2], False, torch.float16,
                    marks=pytest.mark.smoke,
                    id="flat255_fp16",
                ),
                # (4, 7, 9): reducing dims [1,2] -> flattened N = 7*9 = 63
                pytest.param(
                    (4, 7, 9), [1, 2], False, torch.float16,
                    marks=pytest.mark.full,
                    id="flat63_fp16",
                ),
                # (3, 3, 86): reducing dims [1,2] -> flattened N = 3*86 = 258
                pytest.param(
                    (3, 3, 86), [1, 2], False, torch.float16,
                    marks=pytest.mark.full,
                    id="flat258_fp16",
                ),
                # keepdim variant
                pytest.param(
                    (2, 5, 51), [1, 2], True, torch.float16,
                    marks=pytest.mark.full,
                    id="flat255_keepdim_fp16",
                ),
                # bfloat16 variant
                pytest.param(
                    (4, 7, 9), [1, 2], False, torch.bfloat16,
                    marks=pytest.mark.full,
                    id="flat63_bf16",
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# VarFwdOp — non-aligned N (single-dim, dim=-1)
# ---------------------------------------------------------------------------


@WelfordNonAlignedFixture
def test_var_non_aligned(m: int, n: int, dtype: torch.dtype) -> None:
    """VarFwdOp correctness with non-aligned N (zero-padding active)."""
    from tileops.ops.reduction.reduce import VarFwdOp

    test = WelfordNonAlignedTest((m, n), dtype, "var", correction=1)
    op = VarFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


# ---------------------------------------------------------------------------
# StdFwdOp — non-aligned N (single-dim, dim=-1)
# ---------------------------------------------------------------------------


@WelfordNonAlignedFixture
def test_std_non_aligned(m: int, n: int, dtype: torch.dtype) -> None:
    """StdFwdOp correctness with non-aligned N (zero-padding active)."""
    from tileops.ops.reduction.reduce import StdFwdOp

    test = WelfordNonAlignedTest((m, n), dtype, "std", correction=1)
    op = StdFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


# ---------------------------------------------------------------------------
# VarMeanFwdOp — non-aligned N (single-dim, dim=-1)
# ---------------------------------------------------------------------------


@WelfordNonAlignedFixture
def test_var_mean_non_aligned(m: int, n: int, dtype: torch.dtype) -> None:
    """VarMeanFwdOp correctness with non-aligned N (zero-padding active)."""
    from tileops.ops.reduction.reduce import VarMeanFwdOp

    test = WelfordNonAlignedTest((m, n), dtype, "var_mean", correction=1)
    op = VarMeanFwdOp(dtype=dtype)
    test.check(op, *test.gen_inputs(), **_tol(dtype))


# ---------------------------------------------------------------------------
# 3D tests — non-aligned hidden dim
# ---------------------------------------------------------------------------


@WelfordNonAligned3DFixture
def test_var_3d_non_aligned(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """VarFwdOp on 3D input with non-aligned last dim."""
    from tileops.ops.reduction.reduce import VarFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = VarFwdOp(dtype=dtype)
    ref = x.float().var(dim=-1, correction=1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"3D var non-aligned max err: {(y - ref).abs().max()}"


@WelfordNonAligned3DFixture
def test_std_3d_non_aligned(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """StdFwdOp on 3D input with non-aligned last dim."""
    from tileops.ops.reduction.reduce import StdFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = StdFwdOp(dtype=dtype)
    ref = x.float().std(dim=-1, correction=1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert torch.allclose(y, ref, **tol), f"3D std non-aligned max err: {(y - ref).abs().max()}"


@WelfordNonAligned3DFixture
def test_var_mean_3d_non_aligned(batch: int, seq: int, hidden: int, dtype: torch.dtype) -> None:
    """VarMeanFwdOp on 3D input with non-aligned last dim."""
    from tileops.ops.reduction.reduce import VarMeanFwdOp

    x = torch.randn(batch, seq, hidden, dtype=dtype, device="cuda")
    op = VarMeanFwdOp(dtype=dtype, correction=1)
    ref_var = x.float().var(dim=-1, correction=1).to(dtype)
    ref_mean = x.float().mean(dim=-1).to(dtype)
    var_out, mean_out = op(x)
    tol = _tol(dtype)
    assert torch.allclose(var_out, ref_var, **tol), (
        f"3D var_mean var non-aligned max err: {(var_out - ref_var).abs().max()}"
    )
    assert torch.allclose(mean_out, ref_mean, **tol), (
        f"3D var_mean mean non-aligned max err: {(mean_out - ref_mean).abs().max()}"
    )


# ---------------------------------------------------------------------------
# Multi-dim tests — flattened reduction size is non-aligned
# ---------------------------------------------------------------------------


@WelfordNonAlignedMultiDimFixture
def test_var_multidim_non_aligned(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype
) -> None:
    """VarFwdOp multi-dim reduction where flattened N is non-aligned."""
    from tileops.ops.reduction.reduce import VarFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = VarFwdOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.var(x.float(), dim=dims, keepdim=keepdim, correction=1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), (
        f"var multidim non-aligned max err: {(y - ref).abs().max()}"
    )


@WelfordNonAlignedMultiDimFixture
def test_std_multidim_non_aligned(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype
) -> None:
    """StdFwdOp multi-dim reduction where flattened N is non-aligned."""
    from tileops.ops.reduction.reduce import StdFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = StdFwdOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref = torch.std(x.float(), dim=dims, keepdim=keepdim, correction=1).to(dtype)
    y = op(x)
    tol = _tol(dtype)
    assert y.shape == ref.shape, f"shape mismatch: {y.shape} vs {ref.shape}"
    assert torch.allclose(y, ref, **tol), (
        f"std multidim non-aligned max err: {(y - ref).abs().max()}"
    )


@WelfordNonAlignedMultiDimFixture
def test_var_mean_multidim_non_aligned(
    shape: tuple, dims: list, keepdim: bool, dtype: torch.dtype
) -> None:
    """VarMeanFwdOp multi-dim reduction where flattened N is non-aligned."""
    from tileops.ops.reduction.reduce import VarMeanFwdOp

    x = torch.randn(*shape, dtype=dtype, device="cuda")
    op = VarMeanFwdOp(dtype=dtype, dim=dims, keepdim=keepdim)
    ref_var = torch.var(x.float(), dim=dims, keepdim=keepdim, correction=1).to(dtype)
    ref_mean = torch.mean(x.float(), dim=dims, keepdim=keepdim).to(dtype)
    var_out, mean_out = op(x)
    tol = _tol(dtype)
    assert var_out.shape == ref_var.shape, f"var shape mismatch: {var_out.shape} vs {ref_var.shape}"
    assert mean_out.shape == ref_mean.shape, f"mean shape mismatch: {mean_out.shape} vs {ref_mean.shape}"
    assert torch.allclose(var_out, ref_var, **tol), (
        f"var_mean multidim var non-aligned max err: {(var_out - ref_var).abs().max()}"
    )
    assert torch.allclose(mean_out, ref_mean, **tol), (
        f"var_mean multidim mean non-aligned max err: {(mean_out - ref_mean).abs().max()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
