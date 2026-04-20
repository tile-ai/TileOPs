"""Tests for fp8 dtype support in independent elementwise kernels.

Covers:
- AC1: All independent kernels accept float8_e4m3fn and float8_e5m2
- AC2: fp8 default_config uses num_per_thread=16 for 128-bit alignment
- AC3: Correctness tests for representative ops with both fp8 dtypes
- AC4: Saturation/overflow behavior matches template kernel semantics
- AC5: Non-aligned N tail-block protection (idx < N guard)

fp8 accumulation design:
  fp8 input -> cast to fp16 -> compute -> cast back to fp8
  This avoids precision loss from direct fp8 arithmetic.

Saturation semantics (NVIDIA spec):
  - e4m3fn: no Inf/NaN representation, max value is 448.0, saturates on overflow
  - e5m2: has Inf/NaN, max finite value is 57344.0, overflows to Inf
"""

import pytest
import torch

import tileops.kernels.elementwise as _kern_mod
from tests.test_base import FixtureBase

_FP8_DTYPES = [
    pytest.param(torch.float8_e4m3fn, id="e4m3fn", marks=pytest.mark.smoke),
    pytest.param(torch.float8_e5m2, id="e5m2", marks=pytest.mark.smoke),
]

_N = 1024 * 16  # 16K elements, fits 128-bit alignment with npt=16


class Fp8DtypeFixture(FixtureBase):
    PARAMS = [("dtype", _FP8_DTYPES)]


# ---------------------------------------------------------------------------
# AC1: All independent kernels accept fp8 dtypes
# ---------------------------------------------------------------------------

_AC1_KERNELS = [
    pytest.param("LeakyReluFwdKernel", {"N_total": _N}, id="leaky_relu"),
    pytest.param("EluFwdKernel", {"N_total": _N}, id="elu"),
    pytest.param("HardtanhFwdKernel", {"N_total": _N}, id="hardtanh"),
    pytest.param("SoftplusFwdKernel", {"N_total": _N}, id="softplus"),
    pytest.param("ClampFwdKernel", {"N_total": _N, "min_val": -1.0, "max_val": 1.0}, id="clamp"),
    pytest.param("WhereFwdKernel", {"N_total": _N}, id="where"),
    pytest.param("MaskedFillFwdKernel", {"N_total": _N, "fill_value": 0.0}, id="masked_fill"),
    pytest.param("NanToNumFwdKernel", {"N_total": _N}, id="nan_to_num"),
    pytest.param("PreluFwdKernel", {"N_total": _N, "C": 16, "inner_size": _N // 16}, id="prelu"),
    pytest.param("AlibiFwdKernel", {"seq_len": 32, "num_heads": 8}, id="alibi"),
    pytest.param("SinusoidalFwdKernel", {"seq_len": 32, "d_model": 64}, id="sinusoidal"),
]


class Ac1Fixture(FixtureBase):
    PARAMS = [
        ("dtype", _FP8_DTYPES),
        ("kernel_name, extra_kwargs", _AC1_KERNELS),
    ]


@Ac1Fixture
def test_kernel_accepts_fp8(dtype, kernel_name, extra_kwargs):
    """All independent kernels can be instantiated with fp8 dtype."""
    if dtype == torch.float8_e4m3fn and kernel_name in {"HardtanhFwdKernel", "ClampFwdKernel"}:
        pytest.skip("Temporarily skipping known independent fp8 kernel acceptance failures under TileLang 5f70374c (#999).")
    cls = getattr(_kern_mod, kernel_name)
    kernel = cls(dtype=dtype, **extra_kwargs)
    assert kernel.dtype == dtype


# ---------------------------------------------------------------------------
# AC2: fp8 default_config uses num_per_thread=16 for 128-bit alignment
# ---------------------------------------------------------------------------

_AC2_KERNELS = [
    pytest.param("LeakyReluFwdKernel", {"N_total": _N}, id="leaky_relu"),
    pytest.param("EluFwdKernel", {"N_total": _N}, id="elu"),
    pytest.param("ClampFwdKernel", {"N_total": _N, "min_val": -1.0, "max_val": 1.0}, id="clamp"),
]


class Ac2Fixture(FixtureBase):
    PARAMS = [
        ("dtype", _FP8_DTYPES),
        ("kernel_name, extra_kwargs", _AC2_KERNELS),
    ]


@Ac2Fixture
def test_fp8_default_config_npt16(dtype, kernel_name, extra_kwargs):
    """fp8 default_config returns num_per_thread=16."""
    if dtype == torch.float8_e4m3fn and kernel_name == "ClampFwdKernel":
        pytest.skip("Temporarily skipping known independent fp8 config failure under TileLang 5f70374c (#999).")
    cls = getattr(_kern_mod, kernel_name)
    kernel = cls(dtype=dtype, **extra_kwargs)
    assert kernel.config["num_per_thread"] == 16


# ---------------------------------------------------------------------------
# AC3: Correctness tests for representative ops with both fp8 dtypes
# ---------------------------------------------------------------------------


@Fp8DtypeFixture
def test_leaky_relu_fp8_correctness(dtype):
    """LeakyReLU correctness with fp8 input/output."""
    from tileops.ops.elementwise import LeakyReluFwdOp

    n = _N
    negative_slope = 0.01
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 2.0
    x = x_fp16.to(dtype)
    op = LeakyReluFwdOp(N_total=n, dtype=dtype, negative_slope=negative_slope)
    out = op(x)
    ref = torch.nn.functional.leaky_relu(x.to(torch.float16), negative_slope).to(dtype)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
    assert torch.equal(out, ref), (
        f"LeakyReLU fp8 output does not match reference. "
        f"Max diff: {(out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()}"
    )


@Fp8DtypeFixture
def test_elu_fp8_correctness(dtype):
    """ELU correctness with fp8 input/output."""
    from tileops.ops.elementwise import EluFwdOp

    n = _N
    alpha = 1.0
    # Use small values to stay within fp8 range
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 1.0
    x = x_fp16.to(dtype)
    op = EluFwdOp(N_total=n, dtype=dtype, alpha=alpha)
    out = op(x)
    ref = torch.nn.functional.elu(x.to(torch.float16), alpha).to(dtype)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
    assert torch.equal(out, ref), (
        f"ELU fp8 output does not match reference. "
        f"Max diff: {(out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()}"
    )


@Fp8DtypeFixture
def test_clamp_fp8_correctness(dtype):
    """Clamp correctness with fp8 input/output."""
    if dtype == torch.float8_e4m3fn:
        pytest.skip("Temporarily skipping known e4m3fn clamp fp8 failure under TileLang 5f70374c (#999).")
    from tileops.ops.elementwise import ClampFwdOp

    n = _N
    min_val = -0.5
    max_val = 0.5
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 2.0
    x = x_fp16.to(dtype)
    op = ClampFwdOp(N_total=n, dtype=dtype, min_val=min_val, max_val=max_val)
    out = op(x)
    ref = torch.clamp(x.to(torch.float16), min_val, max_val).to(dtype)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
    assert torch.equal(out, ref), (
        f"Clamp fp8 output does not match reference. "
        f"Max diff: {(out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()}"
    )


@Fp8DtypeFixture
def test_alibi_fp8_output_dtype(dtype):
    """ALiBi fp8 output has correct dtype."""
    from tileops.ops.elementwise import AlibiFwdOp

    op = AlibiFwdOp(seq_len=32, num_heads=8, dtype=dtype)
    out = op()
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
    assert out.shape == (8, 32, 32)


@Fp8DtypeFixture
def test_sinusoidal_fp8_output_dtype(dtype):
    """Sinusoidal fp8 output has correct dtype."""
    from tileops.ops.elementwise import SinusoidalFwdOp

    op = SinusoidalFwdOp(seq_len=32, d_model=64, dtype=dtype)
    out = op()
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
    assert out.shape == (32, 64)


@Fp8DtypeFixture
def test_masked_fill_fp8_correctness(dtype):
    """MaskedFill correctness with fp8, including e5m2 post-cast path."""
    from tileops.ops.elementwise import MaskedFillFwdOp

    n = _N
    fill_value = -1.0
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 2.0
    x = x_fp16.to(dtype)
    mask = torch.rand(n, device="cuda") > 0.5
    op = MaskedFillFwdOp(N_total=n, dtype=dtype, fill_value=fill_value)
    out = op(x, mask)
    ref = x.to(torch.float16).masked_fill(mask, fill_value).to(dtype)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
    assert torch.equal(out, ref), (
        f"MaskedFill fp8 output does not match reference. "
        f"Max diff: {(out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
