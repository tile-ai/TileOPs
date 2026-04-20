"""Tests for special predicate elementwise ops (isnan, isinf, isfinite).

Covers L1 smoke correctness (fp16, 1M) and L4 edge cases (fp32, 4K).
"""

import pytest
import torch

from tests.test_base import FixtureBase, TestBase, exact_compare
from tileops.ops.elementwise import IsfiniteFwdOp, IsinfFwdOp, IsnanFwdOp


class SpecialFixture(FixtureBase):
    """Parametrize over shapes / dtypes for special predicate ops."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


class SpecialTest(TestBase):
    """Generic test harness for special predicate ops."""

    def __init__(self, n_total: int, dtype: torch.dtype, ref_fn, gen_fn=None):
        self.n_total = n_total
        self.dtype = dtype
        self._ref_fn = ref_fn
        self._gen_fn = gen_fn

    def gen_inputs(self) -> tuple[torch.Tensor]:
        if self._gen_fn is not None:
            return (self._gen_fn(self.n_total, self.dtype),)
        x = torch.randn(self.n_total, device="cuda", dtype=self.dtype)
        quarter = self.n_total // 4
        x[:quarter] = float("nan")
        x[quarter:2 * quarter] = float("inf")
        x[2 * quarter:3 * quarter] = float("-inf")
        return (x,)

    def ref_program(self, x: torch.Tensor) -> torch.Tensor:
        return self._ref_fn(x)


def _make_special_test(n_total, dtype, op_cls, ref_fn, gen_fn=None) -> None:
    test = SpecialTest(n_total, dtype, ref_fn=ref_fn, gen_fn=gen_fn)
    op = op_cls(N_total=n_total, dtype=dtype)
    test.check(op, *test.gen_inputs(), compare=exact_compare)


@SpecialFixture
def test_isnan(n_total: int, dtype: torch.dtype) -> None:
    _make_special_test(n_total, dtype, IsnanFwdOp, torch.isnan)


@SpecialFixture
def test_isinf(n_total: int, dtype: torch.dtype) -> None:
    _make_special_test(n_total, dtype, IsinfFwdOp, torch.isinf)


@SpecialFixture
def test_isfinite(n_total: int, dtype: torch.dtype) -> None:
    _make_special_test(n_total, dtype, IsfiniteFwdOp, torch.isfinite)


# ===========================================================================
# Independent special ops: where, clamp, masked_fill, nan_to_num,
# alibi, sinusoidal
# ===========================================================================


class IndependentFixture(FixtureBase):
    """Parametrize over shapes / dtypes for independent custom-signature ops."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(1_048_576, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


# --- L1: where ---

@IndependentFixture
def test_where(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import WhereFwdOp

    cond = torch.randint(0, 2, (n_total,), device="cuda").bool()
    x = torch.randn(n_total, device="cuda", dtype=dtype)
    y = torch.randn(n_total, device="cuda", dtype=dtype)
    ref = torch.where(cond, x, y)
    op = WhereFwdOp(N_total=n_total, dtype=dtype)
    out = op(cond, x, y)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    print("All checks passed for WhereFwdOp.")


# --- L1: clamp ---

@IndependentFixture
def test_clamp(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import ClampFwdOp

    x = torch.randn(n_total, device="cuda", dtype=dtype)
    ref = torch.clamp(x, -0.5, 0.5)
    op = ClampFwdOp(N_total=n_total, dtype=dtype, min_val=-0.5, max_val=0.5)
    out = op(x)
    if dtype == torch.float16:
        tol = {"atol": 1e-3, "rtol": 1e-3}
    elif dtype == torch.bfloat16:
        tol = {"atol": 1.6e-2, "rtol": 1.6e-2}
    else:
        tol = {"atol": 1e-5, "rtol": 1e-5}
    torch.testing.assert_close(out, ref, **tol)
    print("All checks passed for ClampFwdOp.")


# --- L1: masked_fill ---

@IndependentFixture
def test_masked_fill(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import MaskedFillFwdOp

    x = torch.randn(n_total, device="cuda", dtype=dtype)
    mask = torch.randint(0, 2, (n_total,), device="cuda").bool()
    # Use -100.0 to avoid fp16 overflow (fp16 max ~65504)
    fill_value = -100.0
    ref = x.masked_fill(mask, fill_value)
    op = MaskedFillFwdOp(N_total=n_total, dtype=dtype, fill_value=fill_value)
    out = op(x, mask)
    if dtype == torch.float16:
        tol = {"atol": 1e-3, "rtol": 1e-3}
    elif dtype == torch.bfloat16:
        tol = {"atol": 1.6e-2, "rtol": 1.6e-2}
    else:
        tol = {"atol": 1e-5, "rtol": 1e-5}
    torch.testing.assert_close(out, ref, **tol)
    print("All checks passed for MaskedFillFwdOp.")


# --- L1: nan_to_num ---

@IndependentFixture
def test_nan_to_num(n_total: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import NanToNumFwdOp

    x = torch.randn(n_total, device="cuda", dtype=dtype)
    quarter = n_total // 4
    x[:quarter] = float("nan")
    x[quarter:2 * quarter] = float("inf")
    x[2 * quarter:3 * quarter] = float("-inf")
    ref = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    op = NanToNumFwdOp(N_total=n_total, dtype=dtype, nan_val=0.0, posinf_val=1e4, neginf_val=-1e4)
    out = op(x)
    if dtype == torch.float16:
        tol = {"atol": 1e-3, "rtol": 1e-3}
    elif dtype == torch.bfloat16:
        tol = {"atol": 1.6e-2, "rtol": 1.6e-2}
    else:
        tol = {"atol": 1e-5, "rtol": 1e-5}
    torch.testing.assert_close(out, ref, **tol, equal_nan=True)
    print("All checks passed for NanToNumFwdOp.")


# --- L1: alibi ---

class AlibiFixture(FixtureBase):
    PARAMS = [
        ("seq_len, num_heads, dtype", [
            pytest.param(128, 8, torch.float16, marks=pytest.mark.smoke),
            pytest.param(128, 8, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(128, 8, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@AlibiFixture
def test_alibi(seq_len: int, num_heads: int, dtype: torch.dtype) -> None:
    from tileops.ops.elementwise import AlibiFwdOp

    op = AlibiFwdOp(seq_len=seq_len, num_heads=num_heads, dtype=dtype)
    out = op()

    # Reference: slope_h = 2^(-8*(h+1)/H), bias = -slope * |i - j|
    positions = torch.arange(seq_len, device="cuda", dtype=torch.float32)
    dist = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()
    slopes = torch.pow(
        2.0,
        -8.0 * torch.arange(1, num_heads + 1, device="cuda", dtype=torch.float32) / num_heads,
    )
    ref = (-slopes[:, None, None] * dist[None, :, :]).to(dtype)

    if dtype == torch.float16:
        tol = {"atol": 1e-2, "rtol": 1e-2}
    elif dtype == torch.bfloat16:
        tol = {"atol": 1.6e-2, "rtol": 1.6e-2}
    else:
        tol = {"atol": 1e-5, "rtol": 1e-5}
    torch.testing.assert_close(out, ref, **tol)
    print("All checks passed for AlibiFwdOp.")


# --- L1: sinusoidal ---

class SinusoidalFixture(FixtureBase):
    PARAMS = [
        ("seq_len, d_model, dtype", [
            pytest.param(512, 256, torch.float16, marks=pytest.mark.smoke),
            pytest.param(512, 256, torch.bfloat16, marks=pytest.mark.smoke),
            pytest.param(512, 256, torch.float32, marks=pytest.mark.smoke),
        ]),
    ]


@SinusoidalFixture
def test_sinusoidal(seq_len: int, d_model: int, dtype: torch.dtype) -> None:

    from tileops.ops.elementwise import SinusoidalFwdOp

    op = SinusoidalFwdOp(seq_len=seq_len, d_model=d_model, dtype=dtype)
    out = op()

    # Reference
    pos = torch.arange(seq_len, device="cuda", dtype=torch.float32).unsqueeze(1)
    dim_pairs = torch.arange(0, d_model, 2, device="cuda", dtype=torch.float32)
    angles = pos / torch.pow(10000.0, dim_pairs / d_model)
    ref = torch.zeros(seq_len, d_model, device="cuda", dtype=torch.float32)
    ref[:, 0::2] = torch.sin(angles)
    ref[:, 1::2] = torch.cos(angles)
    ref = ref.to(dtype)

    if dtype == torch.float16:
        tol = {"atol": 1e-3, "rtol": 1e-3}
    elif dtype == torch.bfloat16:
        tol = {"atol": 1.6e-2, "rtol": 1.6e-2}
    else:
        tol = {"atol": 1e-5, "rtol": 1e-5}
    torch.testing.assert_close(out, ref, **tol)
    print("All checks passed for SinusoidalFwdOp.")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
