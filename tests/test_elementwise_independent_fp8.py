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
    pytest.param("LeakyReluKernel", {"N_total": _N}, id="leaky_relu"),
    pytest.param("EluKernel", {"N_total": _N}, id="elu"),
    pytest.param("HardtanhKernel", {"N_total": _N}, id="hardtanh"),
    pytest.param("SoftplusKernel", {"N_total": _N}, id="softplus"),
    pytest.param("ClampKernel", {"N_total": _N, "min_val": -1.0, "max_val": 1.0}, id="clamp"),
    pytest.param("WhereKernel", {"N_total": _N}, id="where"),
    pytest.param("MaskedFillKernel", {"N_total": _N, "fill_value": 0.0}, id="masked_fill"),
    pytest.param("NanToNumKernel", {"N_total": _N}, id="nan_to_num"),
    pytest.param("PreluKernel", {"N_total": _N, "C": 16, "inner_size": _N // 16}, id="prelu"),
    pytest.param("AlibiKernel", {"seq_len": 32, "num_heads": 8}, id="alibi"),
    pytest.param("SinusoidalKernel", {"seq_len": 32, "d_model": 64}, id="sinusoidal"),
]


class Ac1Fixture(FixtureBase):
    PARAMS = [
        ("dtype", _FP8_DTYPES),
        ("kernel_name, extra_kwargs", _AC1_KERNELS),
    ]


@Ac1Fixture
def test_kernel_accepts_fp8(dtype, kernel_name, extra_kwargs):
    """All independent kernels can be instantiated with fp8 dtype."""
    cls = getattr(_kern_mod, kernel_name)
    kernel = cls(dtype=dtype, **extra_kwargs)
    assert kernel.dtype == dtype


@pytest.mark.smoke
def test_masked_fill_kernel_clamps_overflow_fill_value():
    """MaskedFillKernel clamps fill_value exceeding e4m3fn max (448)."""
    dtype = torch.float8_e4m3fn
    kernel = _kern_mod.MaskedFillKernel(N_total=_N, dtype=dtype, fill_value=1e4)
    assert kernel.fill_value == torch.finfo(dtype).max


@pytest.mark.smoke
def test_nan_to_num_kernel_clamps_overflow_defaults():
    """NanToNumKernel clamps default posinf_val/neginf_val for e4m3fn."""
    dtype = torch.float8_e4m3fn
    kernel = _kern_mod.NanToNumKernel(N_total=_N, dtype=dtype)
    finfo = torch.finfo(dtype)
    assert kernel.posinf_val == finfo.max
    assert kernel.neginf_val == finfo.min


# ---------------------------------------------------------------------------
# AC2: fp8 default_config uses num_per_thread=16 for 128-bit alignment
# ---------------------------------------------------------------------------

_AC2_KERNELS = [
    pytest.param("LeakyReluKernel", {"N_total": _N}, id="leaky_relu"),
    pytest.param("EluKernel", {"N_total": _N}, id="elu"),
    pytest.param("ClampKernel", {"N_total": _N, "min_val": -1.0, "max_val": 1.0}, id="clamp"),
]


class Ac2Fixture(FixtureBase):
    PARAMS = [
        ("dtype", _FP8_DTYPES),
        ("kernel_name, extra_kwargs", _AC2_KERNELS),
    ]


@Ac2Fixture
def test_fp8_default_config_npt16(dtype, kernel_name, extra_kwargs):
    """fp8 default_config returns num_per_thread=16."""
    cls = getattr(_kern_mod, kernel_name)
    kernel = cls(dtype=dtype, **extra_kwargs)
    assert kernel.config["num_per_thread"] == 16


# ---------------------------------------------------------------------------
# AC3: Correctness tests for representative ops with both fp8 dtypes
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
    ref = torch.clamp(x.to(torch.float16), min_val, max_val).to(dtype)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
    assert torch.equal(out, ref), (
        f"Clamp fp8 output does not match reference. "
        f"Max diff: {(out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()}"
    )


@Fp8DtypeFixture
def test_alibi_fp8_output_dtype(dtype):
    """ALiBi fp8 output has correct dtype."""
    from tileops.ops.elementwise import AlibiOp

    op = AlibiOp(seq_len=32, num_heads=8, dtype=dtype)
    out = op()
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
    assert out.shape == (8, 32, 32)


@Fp8DtypeFixture
def test_sinusoidal_fp8_output_dtype(dtype):
    """Sinusoidal fp8 output has correct dtype."""
    from tileops.ops.elementwise import SinusoidalOp

    op = SinusoidalOp(seq_len=32, d_model=64, dtype=dtype)
    out = op()
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
    assert out.shape == (32, 64)


@Fp8DtypeFixture
def test_masked_fill_fp8_correctness(dtype):
    """MaskedFill correctness with fp8, including e5m2 post-cast path."""
    from tileops.ops.elementwise import MaskedFillOp

    n = _N
    fill_value = -1.0
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 2.0
    x = x_fp16.to(dtype)
    mask = torch.rand(n, device="cuda") > 0.5
    op = MaskedFillOp(N_total=n, dtype=dtype, fill_value=fill_value)
    out = op(x, mask)
    ref = x.to(torch.float16).masked_fill(mask, fill_value).to(dtype)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
    assert torch.equal(out, ref), (
        f"MaskedFill fp8 output does not match reference. "
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


@pytest.mark.smoke
def test_masked_fill_e5m2_overflow_fill_value():
    """MaskedFill rejects fill_value that exceeds effective kernel dtype range."""
    from tileops.ops.elementwise import MaskedFillOp

    n = 1024
    dtype = torch.float8_e5m2
    fill_value = 1e5
    with pytest.raises(ValueError, match="fill_value=.*not representable"):
        MaskedFillOp(N_total=n, dtype=dtype, fill_value=fill_value)


@pytest.mark.smoke
def test_nan_to_num_e5m2_overflow_scalar_params_rejected():
    """NanToNum rejects replacement values that exceed effective kernel dtype range."""
    from tileops.ops.elementwise import NanToNumOp

    n = 1024
    dtype = torch.float8_e5m2
    with pytest.raises(ValueError, match="posinf_val=.*not representable"):
        NanToNumOp(N_total=n, dtype=dtype, nan_val=0.0, posinf_val=1e5, neginf_val=-1.0)
    with pytest.raises(ValueError, match="neginf_val=.*not representable"):
        NanToNumOp(N_total=n, dtype=dtype, nan_val=0.0, posinf_val=1.0, neginf_val=-1e5)


@pytest.mark.smoke
def test_leaky_relu_e5m2_overflow_negative_slope_rejected():
    """LeakyReLU rejects negative_slope that exceeds effective kernel dtype range."""
    from tileops.ops.elementwise import LeakyReluOp

    n = 1024
    dtype = torch.float8_e5m2
    with pytest.raises(ValueError, match="negative_slope=.*not representable"):
        LeakyReluOp(N_total=n, dtype=dtype, negative_slope=1e5)


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
