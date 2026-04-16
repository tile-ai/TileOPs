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


class SpecialEdgeFixture(FixtureBase):
    """L4 edge-case fixture: fp32, 4K elements."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4096, torch.float32, marks=pytest.mark.full),
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


# ---------------------------------------------------------------------------
# L4 edge-case tests (fp32, 4K)
# ---------------------------------------------------------------------------


@SpecialEdgeFixture
def test_isnan_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: all NaN input."""
    def _all_nan(n, dtype):
        return torch.full((n,), float("nan"), device="cuda", dtype=dtype)

    _make_special_test(n_total, dtype, IsnanFwdOp, torch.isnan, gen_fn=_all_nan)


@SpecialEdgeFixture
def test_isinf_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: mix of +inf and -inf."""
    def _all_inf(n, dtype):
        x = torch.full((n,), float("inf"), device="cuda", dtype=dtype)
        x[:n // 2] = float("-inf")
        return x

    _make_special_test(n_total, dtype, IsinfFwdOp, torch.isinf, gen_fn=_all_inf)


@SpecialEdgeFixture
def test_isfinite_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: all finite input."""
    def _all_finite(n, dtype):
        return torch.randn(n, device="cuda", dtype=dtype)

    _make_special_test(n_total, dtype, IsfiniteFwdOp, torch.isfinite, gen_fn=_all_finite)


@pytest.mark.smoke
def test_special_predicates_reject_non_float_dtype() -> None:
    from tileops.kernels.elementwise import IsnanFwdKernel

    with pytest.raises(ValueError, match="only supports dtypes"):
        IsnanFwdKernel(N_total=16, dtype=torch.int32)


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


class IndependentEdgeFixture(FixtureBase):
    """L4 edge-case fixture: fp32, 4K elements."""

    PARAMS = [
        ("n_total, dtype", [
            pytest.param(4096, torch.float32, marks=pytest.mark.full),
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


# ===========================================================================
# L2 — Dtype x Size (4 cases for clamp)
# ===========================================================================


class ClampDtypeSizeFixture(FixtureBase):
    PARAMS = [
        ("n_total, dtype", [
            pytest.param(1_048_576, torch.float16, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.float32, marks=pytest.mark.full),
            pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.full),
            pytest.param(4096, torch.float16, marks=pytest.mark.full),
            pytest.param(16_777_216, torch.float16, marks=pytest.mark.full),
        ]),
    ]


@ClampDtypeSizeFixture
def test_clamp_dtype_size(n_total: int, dtype: torch.dtype) -> None:
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
    print("All checks passed for ClampFwdOp dtype/size variant.")


# ===========================================================================
# L4 — Edge Cases (8 cases, fp32, 4K)
# ===========================================================================


@IndependentEdgeFixture
def test_clamp_min_gt_max(n_total: int, dtype: torch.dtype) -> None:
    """Edge: min > max -- PyTorch clamp semantics: min wins (output = min_val)."""
    from tileops.ops.elementwise import ClampFwdOp

    x = torch.randn(n_total, device="cuda", dtype=dtype)
    # When min > max, PyTorch clamp returns min_val for all elements
    ref = torch.clamp(x, min=0.5, max=-0.5)
    op = ClampFwdOp(N_total=n_total, dtype=dtype, min_val=0.5, max_val=-0.5)
    out = op(x)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    print("All checks passed for ClampFwdOp min>max edge case.")


@IndependentEdgeFixture
def test_clamp_upper_only(n_total: int, dtype: torch.dtype) -> None:
    """Edge: min=None, max=0.5 (upper bound only)."""
    from tileops.ops.elementwise import ClampFwdOp

    x = torch.randn(n_total, device="cuda", dtype=dtype)
    ref = torch.clamp(x, min=None, max=0.5)
    op = ClampFwdOp(N_total=n_total, dtype=dtype, min_val=None, max_val=0.5)
    out = op(x)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    print("All checks passed for ClampFwdOp upper-only edge case.")


@IndependentEdgeFixture
def test_clamp_lower_only(n_total: int, dtype: torch.dtype) -> None:
    """Edge: min=-0.5, max=None (lower bound only)."""
    from tileops.ops.elementwise import ClampFwdOp

    x = torch.randn(n_total, device="cuda", dtype=dtype)
    ref = torch.clamp(x, min=-0.5, max=None)
    op = ClampFwdOp(N_total=n_total, dtype=dtype, min_val=-0.5, max_val=None)
    out = op(x)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    print("All checks passed for ClampFwdOp lower-only edge case.")


@IndependentEdgeFixture
def test_masked_fill_all_true(n_total: int, dtype: torch.dtype) -> None:
    """Edge: all True mask -> all values replaced."""
    from tileops.ops.elementwise import MaskedFillFwdOp

    x = torch.randn(n_total, device="cuda", dtype=dtype)
    mask = torch.ones(n_total, device="cuda", dtype=torch.bool)
    fill_value = -1e9
    ref = x.masked_fill(mask, fill_value)
    op = MaskedFillFwdOp(N_total=n_total, dtype=dtype, fill_value=fill_value)
    out = op(x, mask)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    print("All checks passed for MaskedFillFwdOp all-true edge case.")


@IndependentEdgeFixture
def test_masked_fill_all_false(n_total: int, dtype: torch.dtype) -> None:
    """Edge: all False mask -> input unchanged."""
    from tileops.ops.elementwise import MaskedFillFwdOp

    x = torch.randn(n_total, device="cuda", dtype=dtype)
    mask = torch.zeros(n_total, device="cuda", dtype=torch.bool)
    fill_value = -1e9
    ref = x.masked_fill(mask, fill_value)
    op = MaskedFillFwdOp(N_total=n_total, dtype=dtype, fill_value=fill_value)
    out = op(x, mask)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    print("All checks passed for MaskedFillFwdOp all-false edge case.")


@IndependentEdgeFixture
def test_where_all_true(n_total: int, dtype: torch.dtype) -> None:
    """Edge: all True cond -> output = x."""
    from tileops.ops.elementwise import WhereFwdOp

    cond = torch.ones(n_total, device="cuda", dtype=torch.bool)
    x = torch.randn(n_total, device="cuda", dtype=dtype)
    y = torch.randn(n_total, device="cuda", dtype=dtype)
    ref = torch.where(cond, x, y)
    op = WhereFwdOp(N_total=n_total, dtype=dtype)
    out = op(cond, x, y)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    print("All checks passed for WhereFwdOp all-true edge case.")


@IndependentEdgeFixture
def test_where_all_false(n_total: int, dtype: torch.dtype) -> None:
    """Edge: all False cond -> output = y."""
    from tileops.ops.elementwise import WhereFwdOp

    cond = torch.zeros(n_total, device="cuda", dtype=torch.bool)
    x = torch.randn(n_total, device="cuda", dtype=dtype)
    y = torch.randn(n_total, device="cuda", dtype=dtype)
    ref = torch.where(cond, x, y)
    op = WhereFwdOp(N_total=n_total, dtype=dtype)
    out = op(cond, x, y)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)
    print("All checks passed for WhereFwdOp all-false edge case.")


@IndependentEdgeFixture
def test_nan_to_num_edge(n_total: int, dtype: torch.dtype) -> None:
    """Edge: explicit [NaN, Inf, -Inf, 1.0] pattern."""
    from tileops.ops.elementwise import NanToNumFwdOp

    x = torch.zeros(n_total, device="cuda", dtype=dtype)
    # Fill pattern: NaN, Inf, -Inf, 1.0, repeating
    for k in range(0, n_total, 4):
        x[k] = float("nan")
        if k + 1 < n_total:
            x[k + 1] = float("inf")
        if k + 2 < n_total:
            x[k + 2] = float("-inf")
        if k + 3 < n_total:
            x[k + 3] = 1.0

    ref = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    op = NanToNumFwdOp(N_total=n_total, dtype=dtype, nan_val=0.0, posinf_val=1e4, neginf_val=-1e4)
    out = op(x)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5, equal_nan=True)
    print("All checks passed for NanToNumFwdOp edge case.")


@pytest.mark.smoke
def test_independent_special_rejects_non_float_dtype() -> None:
    from tileops.kernels.elementwise import ClampFwdKernel
    with pytest.raises(ValueError, match="only supports dtypes"):
        ClampFwdKernel(N_total=16, dtype=torch.int32)


# ===========================================================================
# Negative tests: forward() dtype / numel validation
# ===========================================================================


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls, kwargs", [
    pytest.param("EluFwdOp", {"alpha": 1.0}, id="elu"),
    pytest.param("HardtanhFwdOp", {"min_val": -1.0, "max_val": 1.0}, id="hardtanh"),
    pytest.param("SoftplusFwdOp", {"beta": 1.0, "threshold": 20.0}, id="softplus"),
    pytest.param("ClampFwdOp", {"min_val": -0.5, "max_val": 0.5}, id="clamp"),
])
def test_forward_rejects_wrong_dtype(op_cls: str, kwargs: dict) -> None:
    """forward() must raise ValueError when input dtype mismatches."""
    import tileops.ops.elementwise as mod
    cls = getattr(mod, op_cls)
    op = cls(N_total=1024, dtype=torch.float16, **kwargs)
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="dtype"):
        op(x)


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls, kwargs", [
    pytest.param("EluFwdOp", {"alpha": 1.0}, id="elu"),
    pytest.param("HardtanhFwdOp", {"min_val": -1.0, "max_val": 1.0}, id="hardtanh"),
    pytest.param("SoftplusFwdOp", {"beta": 1.0, "threshold": 20.0}, id="softplus"),
    pytest.param("ClampFwdOp", {"min_val": -0.5, "max_val": 0.5}, id="clamp"),
])
def test_forward_rejects_wrong_numel(op_cls: str, kwargs: dict) -> None:
    """forward() must raise ValueError when input numel mismatches."""
    import tileops.ops.elementwise as mod
    cls = getattr(mod, op_cls)
    op = cls(N_total=1024, dtype=torch.float16, **kwargs)
    x = torch.randn(512, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="elements"):
        op(x)


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls, kwargs", [
    pytest.param("MaskedFillFwdOp", {"fill_value": -100.0}, id="masked_fill"),
])
def test_masked_fill_forward_rejects_wrong_dtype(op_cls: str, kwargs: dict) -> None:
    """MaskedFillFwdOp forward() must raise ValueError when input dtype mismatches."""
    import tileops.ops.elementwise as mod
    cls = getattr(mod, op_cls)
    op = cls(N_total=1024, dtype=torch.float16, **kwargs)
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    mask = torch.ones(1024, device="cuda", dtype=torch.bool)
    with pytest.raises(ValueError, match="dtype"):
        op(x, mask)


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls, kwargs", [
    pytest.param("MaskedFillFwdOp", {"fill_value": -100.0}, id="masked_fill"),
])
def test_masked_fill_forward_rejects_wrong_numel(op_cls: str, kwargs: dict) -> None:
    """MaskedFillFwdOp forward() must raise ValueError when input numel mismatches."""
    import tileops.ops.elementwise as mod
    cls = getattr(mod, op_cls)
    op = cls(N_total=1024, dtype=torch.float16, **kwargs)
    x = torch.randn(512, device="cuda", dtype=torch.float16)
    mask = torch.ones(512, device="cuda", dtype=torch.bool)
    with pytest.raises(ValueError, match="elements"):
        op(x, mask)


# ===========================================================================
# Negative tests: __init__() scalar parameter validation
# ===========================================================================


@pytest.mark.smoke
def test_elu_rejects_unrepresentable_alpha() -> None:
    """EluFwdOp must reject alpha that overflows the kernel dtype."""
    from tileops.ops.elementwise import EluFwdOp
    with pytest.raises((ValueError, TypeError)):
        EluFwdOp(N_total=1024, dtype=torch.float16, alpha=1e6)


@pytest.mark.smoke
def test_hardtanh_rejects_unrepresentable_min_val() -> None:
    """HardtanhFwdOp must reject min_val that overflows the kernel dtype."""
    from tileops.ops.elementwise import HardtanhFwdOp
    with pytest.raises((ValueError, TypeError)):
        HardtanhFwdOp(N_total=1024, dtype=torch.float16, min_val=1e6)


@pytest.mark.smoke
def test_hardtanh_rejects_unrepresentable_max_val() -> None:
    """HardtanhFwdOp must reject max_val that overflows the kernel dtype."""
    from tileops.ops.elementwise import HardtanhFwdOp
    with pytest.raises((ValueError, TypeError)):
        HardtanhFwdOp(N_total=1024, dtype=torch.float16, max_val=1e6)


@pytest.mark.smoke
def test_softplus_rejects_unrepresentable_beta() -> None:
    """SoftplusFwdOp must reject beta that overflows the kernel dtype."""
    from tileops.ops.elementwise import SoftplusFwdOp
    with pytest.raises((ValueError, TypeError)):
        SoftplusFwdOp(N_total=1024, dtype=torch.float16, beta=1e6)


@pytest.mark.smoke
def test_softplus_rejects_unrepresentable_threshold() -> None:
    """SoftplusFwdOp must reject threshold that overflows the kernel dtype."""
    from tileops.ops.elementwise import SoftplusFwdOp
    with pytest.raises((ValueError, TypeError)):
        SoftplusFwdOp(N_total=1024, dtype=torch.float16, threshold=1e6)


@pytest.mark.smoke
def test_clamp_rejects_unrepresentable_min_val() -> None:
    """ClampFwdOp must reject min_val that overflows the kernel dtype."""
    from tileops.ops.elementwise import ClampFwdOp
    with pytest.raises((ValueError, TypeError)):
        ClampFwdOp(N_total=1024, dtype=torch.float16, min_val=1e6)


@pytest.mark.smoke
def test_clamp_rejects_unrepresentable_max_val() -> None:
    """ClampFwdOp must reject max_val that overflows the kernel dtype."""
    from tileops.ops.elementwise import ClampFwdOp
    with pytest.raises((ValueError, TypeError)):
        ClampFwdOp(N_total=1024, dtype=torch.float16, max_val=1e6)


@pytest.mark.smoke
def test_masked_fill_forward_rejects_cpu_mask() -> None:
    """MaskedFillFwdOp forward() must raise ValueError when mask is not on CUDA."""
    from tileops.ops.elementwise import MaskedFillFwdOp
    op = MaskedFillFwdOp(N_total=1024, dtype=torch.float16, fill_value=-100.0)
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    mask = torch.ones(1024, dtype=torch.bool)  # CPU mask
    with pytest.raises(ValueError, match="Mask must be a CUDA tensor"):
        op(x, mask)


@pytest.mark.smoke
def test_masked_fill_forward_rejects_non_bool_mask() -> None:
    """MaskedFillFwdOp forward() must raise ValueError when mask dtype is not bool."""
    from tileops.ops.elementwise import MaskedFillFwdOp
    op = MaskedFillFwdOp(N_total=1024, dtype=torch.float16, fill_value=-100.0)
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    mask = torch.ones(1024, device="cuda", dtype=torch.float32)  # wrong dtype
    with pytest.raises(ValueError, match="mask.dtype"):
        op(x, mask)


@pytest.mark.smoke
def test_masked_fill_forward_rejects_wrong_mask_numel() -> None:
    """MaskedFillFwdOp forward() must raise ValueError when mask numel mismatches."""
    from tileops.ops.elementwise import MaskedFillFwdOp
    op = MaskedFillFwdOp(N_total=1024, dtype=torch.float16, fill_value=-100.0)
    x = torch.randn(1024, device="cuda", dtype=torch.float16)
    mask = torch.ones(512, device="cuda", dtype=torch.bool)  # wrong numel
    with pytest.raises(ValueError, match="elements"):
        op(x, mask)


@pytest.mark.smoke
def test_elu_rejects_infinite_alpha() -> None:
    """EluFwdOp must reject infinite alpha."""
    from tileops.ops.elementwise import EluFwdOp
    with pytest.raises(ValueError, match="finite"):
        EluFwdOp(N_total=1024, dtype=torch.float32, alpha=float("inf"))


@pytest.mark.smoke
def test_softplus_rejects_non_numeric_beta() -> None:
    """SoftplusFwdOp must reject non-numeric beta."""
    from tileops.ops.elementwise import SoftplusFwdOp
    with pytest.raises(TypeError, match="int/float"):
        SoftplusFwdOp(N_total=1024, dtype=torch.float32, beta="bad")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
