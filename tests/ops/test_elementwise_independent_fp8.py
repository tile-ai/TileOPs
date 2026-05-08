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
    cls = getattr(_kern_mod, kernel_name)
    kernel = cls(dtype=dtype, **extra_kwargs)
    assert kernel.dtype == dtype


@pytest.mark.smoke
def test_masked_fill_kernel_clamps_overflow_fill_value():
    """MaskedFillFwdKernel clamps fill_value exceeding e4m3fn max (448)."""
    dtype = torch.float8_e4m3fn
    kernel = _kern_mod.MaskedFillFwdKernel(N_total=_N, dtype=dtype, fill_value=1e4)
    assert kernel.fill_value == torch.finfo(dtype).max


@pytest.mark.smoke
def test_nan_to_num_op_default_replacements_are_finite_e4m3fn():
    """NanToNumFwdOp default replacement values stay finite end-to-end on e4m3fn.

    Asserts the public Op contract: with the manifest defaults
    (``posinf=None`` / ``neginf=None``), feeding ``+/-inf`` and ``NaN``
    through the op produces an output that is entirely finite in the
    user-facing fp8 dtype. The internal clamping that the kernel layer
    performs is an implementation detail and is not asserted directly.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from tileops.ops.elementwise import NanToNumFwdOp

    dtype = torch.float8_e4m3fn
    n = 1024
    op = NanToNumFwdOp(N_total=n, dtype=dtype)
    x_fp16 = torch.zeros(n, dtype=torch.float16, device="cuda")
    x_fp16[0] = float("inf")
    x_fp16[1] = float("-inf")
    x_fp16[2] = float("nan")
    out = op(x_fp16.to(dtype))
    assert out.dtype == dtype
    assert torch.isfinite(out.to(torch.float32)).all()


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
    from tileops.ops.elementwise import ClampScalarFwdOp

    n = _N
    min_val = -0.5
    max_val = 0.5
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 2.0
    x = x_fp16.to(dtype)
    op = ClampScalarFwdOp(input=(n,), min=min_val, max=max_val, dtype=dtype)
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
    from tileops.ops.elementwise import MaskedFillScalarFwdOp

    n = _N
    fill_value = -1.0
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 2.0
    x = x_fp16.to(dtype)
    mask = torch.rand(n, device="cuda") > 0.5
    op = MaskedFillScalarFwdOp(input=(n,), mask=(n,), value=fill_value, dtype=dtype)
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
    from tileops.ops.elementwise import LeakyReluFwdOp

    n = 1024
    dtype = torch.float8_e4m3fn
    # Large positive values that are already at e4m3fn max
    x_fp16 = torch.full((n,), 448.0, dtype=torch.float16, device="cuda")
    x = x_fp16.to(dtype)
    op = LeakyReluFwdOp(N_total=n, dtype=dtype, negative_slope=0.01)
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
    from tileops.ops.elementwise import ClampScalarFwdOp

    n = 1024
    dtype = torch.float8_e5m2
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 0.5
    x = x_fp16.to(dtype)
    op = ClampScalarFwdOp(input=(n,), min=-1.0, max=1.0, dtype=dtype)
    out = op(x)
    ref = torch.clamp(x.to(torch.float16), -1.0, 1.0).to(dtype)
    assert out.dtype == dtype
    assert torch.equal(out, ref)


@pytest.mark.smoke
def test_elu_e5m2_output_dtype():
    """ELU e5m2 forward returns e5m2 dtype, not fp16."""
    from tileops.ops.elementwise import EluFwdOp

    n = _N
    dtype = torch.float8_e5m2
    x = (torch.randn(n, dtype=torch.float16, device="cuda") * 0.5).to(dtype)
    op = EluFwdOp(N_total=n, dtype=dtype)
    out = op(x)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"


@pytest.mark.smoke
def test_masked_fill_e5m2_overflow_fill_value():
    """MaskedFill rejects fill_value that exceeds effective kernel dtype range."""
    from tileops.ops.elementwise import MaskedFillScalarFwdOp

    n = 1024
    dtype = torch.float8_e5m2
    fill_value = 1e5
    with pytest.raises(ValueError, match="value=.*not representable"):
        MaskedFillScalarFwdOp(input=(n,), mask=(n,), value=fill_value, dtype=dtype)


@pytest.mark.smoke
def test_nan_to_num_e5m2_overflow_scalar_params_rejected():
    """NanToNum rejects replacement values that exceed user-facing dtype range."""
    from tileops.ops.elementwise import NanToNumFwdOp

    n = 1024
    dtype = torch.float8_e5m2
    # Validation messages use the canonical manifest-aligned names
    # (``posinf`` / ``neginf``) regardless of which alias the user passes.
    with pytest.raises(ValueError, match=r"posinf=.*not representable"):
        NanToNumFwdOp(N_total=n, dtype=dtype, nan=0.0, posinf=1e5, neginf=-1.0)
    with pytest.raises(ValueError, match=r"neginf=.*not representable"):
        NanToNumFwdOp(N_total=n, dtype=dtype, nan=0.0, posinf=1.0, neginf=-1e5)


@pytest.mark.smoke
def test_nan_to_num_e5m2_rejects_value_above_fp8_max_but_within_fp16():
    """fp8 explicit replacements must validate against the user-facing dtype.

    ``torch.finfo(torch.float8_e5m2).max`` is 57344.0 while
    ``torch.finfo(torch.float16).max`` is 65504.0. A replacement like
    60000.0 fits in the kernel's intermediate fp16 buffer but overflows
    on the fp8 post-cast and surfaces as ``+Inf``. Validation must
    therefore target the *user-facing* dtype, not the intermediate, so
    callers learn at construction time that the value is unsafe.
    """
    from tileops.ops.elementwise import NanToNumFwdOp

    n = 1024
    dtype = torch.float8_e5m2
    fp8_max = torch.finfo(dtype).max  # 57344.0
    fp16_max = torch.finfo(torch.float16).max  # 65504.0
    above_fp8 = 60000.0
    assert fp8_max < above_fp8 < fp16_max, (
        "Test premise: 60000 must lie strictly between fp8_e5m2 max "
        f"({fp8_max}) and fp16 max ({fp16_max})"
    )
    with pytest.raises(ValueError, match=r"posinf=.*not representable"):
        NanToNumFwdOp(N_total=n, dtype=dtype, posinf=above_fp8)
    with pytest.raises(ValueError, match=r"neginf=.*not representable"):
        NanToNumFwdOp(N_total=n, dtype=dtype, neginf=-above_fp8)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float8_e4m3fn, id="e4m3fn"),
        pytest.param(torch.float8_e5m2, id="e5m2"),
    ],
)
def test_nan_to_num_fp8_default_ctor_accepts_dtype_sentinels(dtype):
    """NanToNumFwdOp default ctor must succeed for fp8 dtypes.

    Regression guard: when ``posinf`` / ``neginf`` are left at their
    manifest default ``None``, the op must NOT validate the legacy
    fp16-shaped sentinels against the narrow fp8 range. Construction
    must succeed with the manifest-default kwargs and the public Op
    state must reflect that the user did not pin a finite override.
    The end-to-end behavior — that ``+inf`` / ``-inf`` resolve to a
    *finite* value in the final fp8 dtype — is exercised by
    ``test_nan_to_num_fp8_default_replaces_inf_with_finite`` below.
    """
    from tileops.ops.elementwise import NanToNumFwdOp

    op = NanToNumFwdOp(N_total=8, dtype=dtype)
    assert op.posinf is None
    assert op.neginf is None
    assert op.dtype == dtype


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float8_e4m3fn, id="e4m3fn"),
        pytest.param(torch.float8_e5m2, id="e5m2"),
    ],
)
def test_nan_to_num_fp8_default_replaces_inf_with_finite(dtype):
    """End-to-end: default ``+/-inf`` sentinels produce finite outputs.

    Constructs the op with the manifest default (``posinf=None`` /
    ``neginf=None``), feeds an input containing ``+/-inf`` and ``NaN``,
    and asserts the *final* op output is entirely finite. This catches
    the regression where the e5m2 path round-tripped through fp16's
    65504.0 sentinel and then overflowed to ``Inf`` on the cast back to
    e5m2 (whose max is 57344.0).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from tileops.ops.elementwise import NanToNumFwdOp

    n = 1024
    op = NanToNumFwdOp(N_total=n, dtype=dtype)
    x_fp16 = torch.empty(n, dtype=torch.float16, device="cuda")
    x_fp16.fill_(0.0)
    x_fp16[0] = float("inf")
    x_fp16[1] = float("-inf")
    x_fp16[2] = float("nan")
    x = x_fp16.to(dtype)

    out = op(x)
    assert out.dtype == dtype
    # Every replaced position must be finite in the final dtype.
    finite_mask = torch.isfinite(out.to(torch.float32))
    assert finite_mask.all(), (
        f"NanToNum default sentinels surfaced non-finite output for {dtype}: "
        f"out[0]={out[0].item()}, out[1]={out[1].item()}, out[2]={out[2].item()}"
    )


@pytest.mark.smoke
def test_leaky_relu_e5m2_overflow_negative_slope_rejected():
    """LeakyReLU rejects negative_slope that exceeds effective kernel dtype range."""
    from tileops.ops.elementwise import LeakyReluFwdOp

    n = 1024
    dtype = torch.float8_e5m2
    with pytest.raises(ValueError, match="negative_slope=.*not representable"):
        LeakyReluFwdOp(N_total=n, dtype=dtype, negative_slope=1e5)


# ---------------------------------------------------------------------------
# AC5: Non-aligned N tail-block protection (idx < N guard)
# ---------------------------------------------------------------------------

_N_UNALIGNED = 1024 * 16 + 37  # not divisible by any typical block_size


class UnalignedFp8Fixture(FixtureBase):
    PARAMS = [("dtype", _FP8_DTYPES)]


@UnalignedFp8Fixture
def test_leaky_relu_fp8_unaligned_n(dtype):
    """LeakyReLU fp8 correctness with non-aligned N (tail block guard)."""
    from tileops.ops.elementwise import LeakyReluFwdOp

    n = _N_UNALIGNED
    negative_slope = 0.01
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 2.0
    x = x_fp16.to(dtype)
    op = LeakyReluFwdOp(N_total=n, dtype=dtype, negative_slope=negative_slope)
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
    from tileops.ops.elementwise import EluFwdOp

    n = _N_UNALIGNED
    alpha = 1.0
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 1.0
    x = x_fp16.to(dtype)
    op = EluFwdOp(N_total=n, dtype=dtype, alpha=alpha)
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
    from tileops.ops.elementwise import ClampScalarFwdOp

    n = _N_UNALIGNED
    min_val = -0.5
    max_val = 0.5
    x_fp16 = torch.randn(n, dtype=torch.float16, device="cuda") * 2.0
    x = x_fp16.to(dtype)
    op = ClampScalarFwdOp(input=(n,), min=min_val, max=max_val, dtype=dtype)
    out = op(x)
    ref = torch.clamp(x.to(torch.float16), min_val, max_val).to(dtype)
    assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
    assert torch.equal(out, ref), (
        f"Clamp fp8 unaligned output does not match reference. "
        f"Max diff: {(out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# WhereFwdOp dtype contract: manifest declares fp16 | bf16 | fp32.
# fp8 dtypes must be rejected at the op-layer signature.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    "bad_dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2],
)
def test_where_rejects_fp8_dtype(bad_dtype: torch.dtype) -> None:
    """WhereFwdOp must reject fp8 dtypes at construction (manifest contract)."""
    from tileops.ops.elementwise import WhereFwdOp

    shape = (4, 8)
    with pytest.raises((ValueError, TypeError)):
        WhereFwdOp(
            condition=shape, input=shape, other=shape, dtype=bad_dtype,
        )


@pytest.mark.smoke
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
def test_where_accepts_manifest_dtypes(dtype: torch.dtype) -> None:
    """WhereFwdOp constructs and runs for every manifest-declared dtype."""
    from tileops.ops.elementwise import WhereFwdOp

    shape = (4, 8)
    cond = torch.randint(0, 2, shape, device="cuda").bool()
    inp = torch.randn(shape, device="cuda", dtype=dtype)
    other = torch.randn(shape, device="cuda", dtype=dtype)
    op = WhereFwdOp(condition=shape, input=shape, other=shape, dtype=dtype)
    out = op(cond, inp, other)
    ref = torch.where(cond, inp, other)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
