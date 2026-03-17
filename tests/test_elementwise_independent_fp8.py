"""Tests for fp8 dtype support in independent elementwise kernels.

Covers:
- AC1: All independent kernels accept float8_e4m3fn and float8_e5m2
- AC2: fp8 default_config uses num_per_thread=16 for 128-bit alignment
- AC3: Correctness tests for leaky_relu, elu, clamp with both fp8 dtypes
- AC4: Saturation/overflow behavior matches template kernel semantics

fp8 accumulation design:
  fp8 input -> cast to fp16 -> compute -> cast back to fp8
  This avoids precision loss from direct fp8 arithmetic.

Saturation semantics (NVIDIA spec):
  - e4m3fn: no Inf/NaN representation, max value is 448.0, saturates on overflow
  - e5m2: has Inf/NaN, max finite value is 57344.0, overflows to Inf
"""

import pytest
import torch

from tests.test_base import FixtureBase

_FP8_DTYPES = [
    pytest.param(torch.float8_e4m3fn, id="e4m3fn", marks=pytest.mark.smoke),
    pytest.param(torch.float8_e5m2, id="e5m2", marks=pytest.mark.smoke),
]

_N = 1024 * 16  # 16K elements, fits 128-bit alignment with npt=16


# ---------------------------------------------------------------------------
# AC1: All independent kernels accept fp8 dtypes
# ---------------------------------------------------------------------------


class Fp8DtypeFixture(FixtureBase):
    PARAMS = [("dtype", _FP8_DTYPES)]


@Fp8DtypeFixture
def test_leaky_relu_kernel_accepts_fp8(dtype):
    """LeakyReluKernel can be instantiated with fp8 dtype."""
    from tileops.kernels.elementwise import LeakyReluKernel

    kernel = LeakyReluKernel(N_total=_N, dtype=dtype)
    assert kernel.dtype == dtype


@Fp8DtypeFixture
def test_elu_kernel_accepts_fp8(dtype):
    """EluKernel can be instantiated with fp8 dtype."""
    from tileops.kernels.elementwise import EluKernel

    kernel = EluKernel(N_total=_N, dtype=dtype)
    assert kernel.dtype == dtype


@Fp8DtypeFixture
def test_hardtanh_kernel_accepts_fp8(dtype):
    """HardtanhKernel can be instantiated with fp8 dtype."""
    from tileops.kernels.elementwise import HardtanhKernel

    kernel = HardtanhKernel(N_total=_N, dtype=dtype)
    assert kernel.dtype == dtype


@Fp8DtypeFixture
def test_softplus_kernel_accepts_fp8(dtype):
    """SoftplusKernel can be instantiated with fp8 dtype."""
    from tileops.kernels.elementwise import SoftplusKernel

    kernel = SoftplusKernel(N_total=_N, dtype=dtype)
    assert kernel.dtype == dtype


@Fp8DtypeFixture
def test_clamp_kernel_accepts_fp8(dtype):
    """ClampKernel can be instantiated with fp8 dtype."""
    from tileops.kernels.elementwise import ClampKernel

    kernel = ClampKernel(N_total=_N, dtype=dtype, min_val=-1.0, max_val=1.0)
    assert kernel.dtype == dtype


@Fp8DtypeFixture
def test_where_kernel_accepts_fp8(dtype):
    """WhereKernel can be instantiated with fp8 dtype."""
    from tileops.kernels.elementwise import WhereKernel

    kernel = WhereKernel(N_total=_N, dtype=dtype)
    assert kernel.dtype == dtype


@Fp8DtypeFixture
def test_masked_fill_kernel_accepts_fp8(dtype):
    """MaskedFillKernel can be instantiated with fp8 dtype."""
    from tileops.kernels.elementwise import MaskedFillKernel

    kernel = MaskedFillKernel(N_total=_N, dtype=dtype, fill_value=0.0)
    assert kernel.dtype == dtype


@Fp8DtypeFixture
def test_nan_to_num_kernel_accepts_fp8(dtype):
    """NanToNumKernel can be instantiated with fp8 dtype."""
    from tileops.kernels.elementwise import NanToNumKernel

    kernel = NanToNumKernel(N_total=_N, dtype=dtype)
    assert kernel.dtype == dtype


@Fp8DtypeFixture
def test_prelu_kernel_accepts_fp8(dtype):
    """PreluKernel can be instantiated with fp8 dtype."""
    from tileops.kernels.elementwise import PreluKernel

    kernel = PreluKernel(N_total=_N, C=16, inner_size=_N // 16, dtype=dtype)
    assert kernel.dtype == dtype


# ---------------------------------------------------------------------------
# AC2: fp8 default_config uses num_per_thread=16 for 128-bit alignment
# ---------------------------------------------------------------------------


@Fp8DtypeFixture
def test_leaky_relu_fp8_default_config_npt16(dtype):
    """LeakyReluKernel fp8 default_config returns num_per_thread=16."""
    from tileops.kernels.elementwise import LeakyReluKernel

    kernel = LeakyReluKernel(N_total=_N, dtype=dtype)
    assert kernel.config["num_per_thread"] == 16


@Fp8DtypeFixture
def test_elu_fp8_default_config_npt16(dtype):
    """EluKernel fp8 default_config returns num_per_thread=16."""
    from tileops.kernels.elementwise import EluKernel

    kernel = EluKernel(N_total=_N, dtype=dtype)
    assert kernel.config["num_per_thread"] == 16


@Fp8DtypeFixture
def test_clamp_fp8_default_config_npt16(dtype):
    """ClampKernel fp8 default_config returns num_per_thread=16."""
    from tileops.kernels.elementwise import ClampKernel

    kernel = ClampKernel(N_total=_N, dtype=dtype, min_val=-1.0, max_val=1.0)
    assert kernel.config["num_per_thread"] == 16


# ---------------------------------------------------------------------------
# AC3: Correctness tests for leaky_relu, elu, clamp with both fp8 dtypes
# ---------------------------------------------------------------------------


@Fp8DtypeFixture
def test_leaky_relu_fp8_correctness(dtype):
    """LeakyReLU correctness with fp8 input/output."""
    from tileops.ops.elementwise import LeakyReluOp

    n = _N
    negative_slope = 0.01
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 2.0
    x = x_fp16.to(dtype)
    op = LeakyReluOp(N_total=n, dtype=dtype, negative_slope=negative_slope)
    out = op(x)
    # Reference: cast to fp16, leaky_relu, cast back to fp8
    ref = torch.nn.functional.leaky_relu(x.to(torch.float16), negative_slope).to(dtype)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
    assert torch.equal(out, ref), (
        f"LeakyReLU fp8 output does not match reference. "
        f"Max diff: {(out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()}"
    )


@Fp8DtypeFixture
def test_elu_fp8_correctness(dtype):
    """ELU correctness with fp8 input/output."""
    from tileops.ops.elementwise import EluOp

    n = _N
    alpha = 1.0
    # Use small values to stay within fp8 range
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 1.0
    x = x_fp16.to(dtype)
    op = EluOp(N_total=n, dtype=dtype, alpha=alpha)
    out = op(x)
    # Reference: cast to fp16, elu, cast back to fp8
    ref = torch.nn.functional.elu(x.to(torch.float16), alpha).to(dtype)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
    assert torch.equal(out, ref), (
        f"ELU fp8 output does not match reference. "
        f"Max diff: {(out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()}"
    )


@Fp8DtypeFixture
def test_clamp_fp8_correctness(dtype):
    """Clamp correctness with fp8 input/output."""
    from tileops.ops.elementwise import ClampOp

    n = _N
    min_val = -0.5
    max_val = 0.5
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 2.0
    x = x_fp16.to(dtype)
    op = ClampOp(N_total=n, dtype=dtype, min_val=min_val, max_val=max_val)
    out = op(x)
    # Reference: cast to fp16, clamp, cast back to fp8
    ref = torch.clamp(x.to(torch.float16), min_val, max_val).to(dtype)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
    assert torch.equal(out, ref), (
        f"Clamp fp8 output does not match reference. "
        f"Max diff: {(out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# AC4: Saturation/overflow behavior matches template kernel semantics
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_leaky_relu_e4m3fn_saturation():
    """LeakyReLU e4m3fn saturates to max value on overflow."""
    from tileops.ops.elementwise import LeakyReluOp

    n = 1024
    dtype = torch.float8_e4m3fn
    # Large positive values that are already at e4m3fn max
    x_fp16 = torch.full((n,), 448.0, dtype=torch.float16, device="cuda")
    x = x_fp16.to(dtype)
    op = LeakyReluOp(N_total=n, dtype=dtype, negative_slope=0.01)
    out = op(x)
    out_fp32 = out.to(torch.float32)
    e4m3_max = torch.finfo(torch.float8_e4m3fn).max
    assert torch.all(out_fp32 <= e4m3_max), (
        f"e4m3fn output should saturate to <= {e4m3_max}, got max={out_fp32.max().item()}"
    )
    assert not torch.any(torch.isinf(out_fp32)), "e4m3fn should not produce Inf"


@pytest.mark.smoke
def test_clamp_e5m2_preserves_values():
    """Clamp e5m2 preserves values within range correctly."""
    from tileops.ops.elementwise import ClampOp

    n = 1024
    dtype = torch.float8_e5m2
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 0.5
    x = x_fp16.to(dtype)
    op = ClampOp(N_total=n, dtype=dtype, min_val=-1.0, max_val=1.0)
    out = op(x)
    ref = torch.clamp(x.to(torch.float16), -1.0, 1.0).to(dtype)
    assert out.dtype == dtype
    assert torch.equal(out, ref)


@pytest.mark.smoke
def test_elu_e5m2_output_dtype():
    """ELU e5m2 forward returns e5m2 dtype, not fp16."""
    from tileops.ops.elementwise import EluOp

    n = _N
    dtype = torch.float8_e5m2
    x = (torch.randn(n, dtype=torch.float16, device="cuda") * 0.5).to(dtype)
    op = EluOp(N_total=n, dtype=dtype)
    out = op(x)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


# ---------------------------------------------------------------------------
# AC5: Non-aligned N tail-block protection (idx < N guard)
# ---------------------------------------------------------------------------

_N_UNALIGNED = 1024 * 16 + 37  # not divisible by any typical block_size


class UnalignedFp8Fixture(FixtureBase):
    PARAMS = [("dtype", _FP8_DTYPES)]


@UnalignedFp8Fixture
def test_leaky_relu_fp8_unaligned_n(dtype):
    """LeakyReLU fp8 correctness with non-aligned N (tail block guard)."""
    from tileops.ops.elementwise import LeakyReluOp

    n = _N_UNALIGNED
    negative_slope = 0.01
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 2.0
    x = x_fp16.to(dtype)
    op = LeakyReluOp(N_total=n, dtype=dtype, negative_slope=negative_slope)
    out = op(x)
    ref = torch.nn.functional.leaky_relu(x.to(torch.float16), negative_slope).to(dtype)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
    assert torch.equal(out, ref), (
        f"LeakyReLU fp8 unaligned output does not match reference. "
        f"Max diff: {(out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()}"
    )


@UnalignedFp8Fixture
def test_elu_fp8_unaligned_n(dtype):
    """ELU fp8 correctness with non-aligned N (tail block guard)."""
    from tileops.ops.elementwise import EluOp

    n = _N_UNALIGNED
    alpha = 1.0
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 1.0
    x = x_fp16.to(dtype)
    op = EluOp(N_total=n, dtype=dtype, alpha=alpha)
    out = op(x)
    ref = torch.nn.functional.elu(x.to(torch.float16), alpha).to(dtype)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
    assert torch.equal(out, ref), (
        f"ELU fp8 unaligned output does not match reference. "
        f"Max diff: {(out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()}"
    )


@UnalignedFp8Fixture
def test_clamp_fp8_unaligned_n(dtype):
    """Clamp fp8 correctness with non-aligned N (tail block guard)."""
    from tileops.ops.elementwise import ClampOp

    n = _N_UNALIGNED
    min_val = -0.5
    max_val = 0.5
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 2.0
    x = x_fp16.to(dtype)
    op = ClampOp(N_total=n, dtype=dtype, min_val=min_val, max_val=max_val)
    out = op(x)
    ref = torch.clamp(x.to(torch.float16), min_val, max_val).to(dtype)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
    assert torch.equal(out, ref), (
        f"Clamp fp8 unaligned output does not match reference. "
        f"Max diff: {(out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
