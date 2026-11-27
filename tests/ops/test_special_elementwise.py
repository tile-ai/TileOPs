"""Tests for the special elementwise ops: predicates, where, clamp, masked_fill, nan_to_num,
alibi and sinusoidal."""

import pytest
import torch

from tests.workload_test_base import FixtureBase, TestBase
from tileops._ops.elementwise import (
    ClampFwdOp,
    HardtanhFwdOp,
    IsfiniteFwdOp,
    IsinfFwdOp,
    IsnanFwdOp,
)
from workloads.device import run_device, run_device_available
from workloads.elementwise import (
    ElementwiseWorkload,
    SpecialCase,
    alibi_reference,
    sinusoidal_reference,
)
from workloads.numerics import compare_outputs


class SpecialFixture(FixtureBase):
    """Parametrize over shapes / dtypes for special predicate ops."""

    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


class SpecialTest(SpecialCase, TestBase):
    pass


def _make_special_test(n_total, dtype, op_cls, gen_fn=None) -> None:
    test = SpecialTest(n_total, dtype, op_cls.__name__, gen_fn=gen_fn)
    op = op_cls()
    test.check(op, *test.gen_inputs())


@SpecialFixture
def test_isnan(n_total: int, dtype: torch.dtype) -> None:
    _make_special_test(n_total, dtype, IsnanFwdOp)


@SpecialFixture
def test_isinf(n_total: int, dtype: torch.dtype) -> None:
    _make_special_test(n_total, dtype, IsinfFwdOp)


@SpecialFixture
def test_isfinite(n_total: int, dtype: torch.dtype) -> None:
    _make_special_test(n_total, dtype, IsfiniteFwdOp)


# Edge values


# The 16-bit encodings isinf must tell apart: both infinities, the largest finite values and
# their neighbours, NaNs of the smallest and the quiet payload of either sign and of the largest
# positive payload, zeros, the smallest subnormal and normal, and ones.
_ISINF_ENCODINGS = {
    torch.float16: (
        0x7C00,
        0xFC00,
        0x7BFF,
        0xFBFF,
        0x7BFE,
        0x7C01,
        0xFC01,
        0x7E00,
        0xFE00,
        0x7FFF,
        0x0000,
        0x8000,
        0x0001,
        0x0400,
        0x3C00,
        0xBC00,
        0x5BFF,
    ),
    torch.bfloat16: (
        0x7F80,
        0xFF80,
        0x7F7F,
        0xFF7F,
        0x7F7E,
        0x7F81,
        0xFF81,
        0x7FC0,
        0xFFC0,
        0x7FFF,
        0x0000,
        0x8000,
        0x0001,
        0x0080,
        0x3F80,
        0xBF80,
        0x5F7F,
    ),
}


def _cycled_encodings(n, dtype):
    """``_ISINF_ENCODINGS`` repeated end to end. Seventeen of them, coprime to every power of
    two, so each encoding lands at every position modulo 4, 8 and 16."""
    codes = torch.tensor(_ISINF_ENCODINGS[dtype], dtype=torch.int32, device=run_device())
    index = torch.arange(n, device=run_device()).remainder(codes.numel())
    return codes[index].to(torch.int16).view(dtype)


class SpecialEncodingFixture(FixtureBase):
    """Sizes taking each run length a launch picks for a 16-bit input, the widest block a
    long grid of whole runs takes, and a partial block."""

    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4_194_304, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(262_144, torch.float16, marks=pytest.mark.full),
                pytest.param(131_072, torch.bfloat16, marks=pytest.mark.full),
                pytest.param(1_000_003, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@SpecialEncodingFixture
def test_isinf_encodings(n_total: int, dtype: torch.dtype) -> None:
    """Infinities beside the largest finite values and NaNs of several payloads, each encoding
    at every position of a thread's run."""
    _make_special_test(n_total, dtype, IsinfFwdOp, gen_fn=_cycled_encodings)


@pytest.mark.smoke
def test_isinf_tuned() -> None:
    """Tuned mode picks a launch whose result matches the reference."""
    test = SpecialTest(1_048_576, torch.bfloat16, "IsinfFwdOp", gen_fn=_cycled_encodings)
    op = IsinfFwdOp()
    op.request_tune()
    test.check(op, *test.gen_inputs())


class IndependentFixture(FixtureBase):
    """Parametrize over shapes / dtypes for independent custom-signature ops."""

    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(1_048_576, torch.float16, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


@IndependentFixture
def test_where(n_total: int, dtype: torch.dtype) -> None:
    from tileops._ops.elementwise import WhereFwdOp

    cond = torch.randint(0, 2, (n_total,), device=run_device()).bool()
    x = torch.randn(n_total, device=run_device(), dtype=dtype)
    y = torch.randn(n_total, device=run_device(), dtype=dtype)
    ref = torch.where(cond, x, y)
    op = WhereFwdOp()
    out = op(cond, x, y)
    compare_outputs(
        out, ref, ElementwiseWorkload(type(op).__name__, (cond, x, y)).verification(*(cond, x, y))
    )


@IndependentFixture
def test_masked_fill(n_total: int, dtype: torch.dtype) -> None:
    from tileops._ops.elementwise import MaskedFillFwdOp

    x = torch.randn(n_total, device=run_device(), dtype=dtype)
    mask = torch.randint(0, 2, (n_total,), device=run_device()).bool()
    fill_value = -100.0
    ref = x.masked_fill(mask, fill_value)
    op = MaskedFillFwdOp()
    out = op(x, mask, fill_value)
    compare_outputs(
        out,
        ref,
        ElementwiseWorkload(type(op).__name__, (x, mask, fill_value)).verification(*(x, mask)),
    )


@IndependentFixture
def test_nan_to_num(n_total: int, dtype: torch.dtype) -> None:
    from tileops._ops.elementwise import NanToNumFwdOp

    x = torch.randn(n_total, device=run_device(), dtype=dtype)
    quarter = n_total // 4
    x[:quarter] = float("nan")
    x[quarter : 2 * quarter] = float("inf")
    x[2 * quarter : 3 * quarter] = float("-inf")
    ref = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    op = NanToNumFwdOp(nan=0.0, posinf=1e4, neginf=-1e4)
    out = op(x)
    compare_outputs(
        out,
        ref,
        ElementwiseWorkload(
            type(op).__name__, (x,), nan=0.0, posinf=10000.0, neginf=-10000.0
        ).verification(*(x,)),
    )


@pytest.mark.smoke
def test_nan_to_num_tuned_ragged() -> None:
    """Tuned, on a bfloat16 run whose last block and last vector are partial."""
    from tileops._ops.elementwise import NanToNumFwdOp

    n_total = 1_048_576 + 1003
    x = torch.randn(n_total, device=run_device(), dtype=torch.bfloat16)
    x[::7] = float("nan")
    x[1::7] = float("inf")
    x[2::7] = float("-inf")
    op = NanToNumFwdOp(nan=0.0, posinf=1e4, neginf=-1e4)
    op.request_tune()
    compare_outputs(
        op(x),
        torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4),
        ElementwiseWorkload(
            type(op).__name__, (x,), nan=0.0, posinf=10000.0, neginf=-10000.0
        ).verification(*(x,)),
    )


class AlibiFixture(FixtureBase):
    PARAMS = [
        (
            "seq_len, num_heads, dtype",
            [
                pytest.param(128, 8, torch.float16, marks=pytest.mark.smoke),
                pytest.param(128, 8, torch.float32, marks=pytest.mark.smoke),
            ],
        ),
    ]


@AlibiFixture
def test_alibi(seq_len: int, num_heads: int, dtype: torch.dtype) -> None:
    from tileops._ops.elementwise import AlibiFwdOp

    op = AlibiFwdOp(seq_len=seq_len, num_heads=num_heads, out_dtype=dtype, device=run_device())
    out = op()
    ref = alibi_reference(seq_len, num_heads, dtype)

    compare_outputs(
        out,
        ref,
        ElementwiseWorkload(
            type(op).__name__,
            (),
            seq_len=seq_len,
            num_heads=num_heads,
            out_dtype=dtype,
            device=run_device(),
        ).verification(*()),
    )


class SinusoidalFixture(FixtureBase):
    PARAMS = [
        (
            "seq_len, d_model, dtype",
            [
                pytest.param(512, 256, torch.float16, marks=pytest.mark.smoke),
                pytest.param(512, 256, torch.float32, marks=pytest.mark.smoke),
                # Neither extent divides the block tile.
                pytest.param(65, 130, torch.float16, marks=pytest.mark.smoke),
            ],
        ),
    ]


@SinusoidalFixture
def test_sinusoidal(seq_len: int, d_model: int, dtype: torch.dtype) -> None:
    from tileops._ops.elementwise import SinusoidalFwdOp

    op = SinusoidalFwdOp(seq_len=seq_len, d_model=d_model, out_dtype=dtype, device=run_device())
    out = op()
    ref = sinusoidal_reference(seq_len, d_model, dtype)

    compare_outputs(
        out,
        ref,
        ElementwiseWorkload(
            type(op).__name__,
            (),
            seq_len=seq_len,
            d_model=d_model,
            out_dtype=dtype,
            device=run_device(),
        ).verification(*()),
    )


class ClampDtypeSizeFixture(FixtureBase):
    PARAMS = [
        (
            "n_total, dtype",
            [
                pytest.param(1_048_576, torch.float32, marks=pytest.mark.smoke),
                pytest.param(1_048_576, torch.bfloat16, marks=pytest.mark.smoke),
                pytest.param(4096, torch.float16, marks=pytest.mark.smoke),
                pytest.param(16_777_216, torch.float16, marks=pytest.mark.full),
            ],
        ),
    ]


@ClampDtypeSizeFixture
def test_clamp_dtype_size(n_total: int, dtype: torch.dtype) -> None:
    from tileops._ops.elementwise import ClampFwdOp

    x = torch.randn(n_total, device=run_device(), dtype=dtype)
    ref = torch.clamp(x, -0.5, 0.5)
    op = ClampFwdOp()
    out = op(x, -0.5, 0.5)
    compare_outputs(
        out,
        ref,
        ElementwiseWorkload(type(op).__name__, (x, -0.5, 0.5)).verification(*(x,)),
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "lo, hi",
    [
        pytest.param(0.5, -0.5, id="min-gt-max"),
        pytest.param(None, 0.5, id="upper-only"),
        pytest.param(-0.5, None, id="lower-only"),
    ],
)
def test_clamp_bounds(lo, hi) -> None:
    """min > max gives max everywhere, and an omitted bound is not applied, as torch.clamp."""
    from tileops._ops.elementwise import ClampFwdOp

    x = torch.randn(4096, device=run_device(), dtype=torch.float32)
    op = ClampFwdOp()
    compare_outputs(
        op(x, lo, hi),
        torch.clamp(x, min=lo, max=hi),
        ElementwiseWorkload(type(op).__name__, (x, lo, hi)).verification(x),
    )


_NAN = float("nan")
_NAN_CASES = [
    pytest.param("number", {"min": -1.0, "max": 1.0}, id="clamp-number-nan-input"),
    pytest.param("number", {"min": _NAN}, id="clamp-number-nan-min"),
    pytest.param("number", {"min": -1.0, "max": _NAN}, id="clamp-number-nan-max"),
    pytest.param("tensor", {"min": -1.0, "max": _NAN}, id="clamp-tensor-nan-input-and-bound"),
    pytest.param("HardtanhFwdOp", {}, id="hardtanh-nan-input"),
    pytest.param("HardsigmoidFwdOp", {}, id="hardsigmoid-nan-input"),
]


@pytest.mark.smoke
@pytest.mark.parametrize("op_name, kwargs", _NAN_CASES)
def test_clamp_family_propagates_nan_like_torch(op_name: str, kwargs: dict) -> None:
    """A NaN input or bound gives NaN, as torch's clamp, hardtanh and hardsigmoid do."""
    import torch.nn.functional as F

    import tileops._ops.elementwise as ew

    dtype = torch.float16
    x = torch.tensor(
        [float("nan"), -3.0, -0.5, 0.25, 2.0, 7.0] * 128, device=run_device(), dtype=dtype
    )
    name = "ClampFwdOp" if op_name in ("tensor", "number") else op_name
    if op_name == "tensor":
        lo = torch.full_like(x, kwargs["min"])
        hi = torch.full_like(x, kwargs["max"])
        out, ref = ew.ClampFwdOp()(x, lo, hi), torch.clamp(x, lo, hi)
    elif op_name == "number":
        bounds = (kwargs.get("min"), kwargs.get("max"))
        out, ref = ew.ClampFwdOp()(x, *bounds), torch.clamp(x, *bounds)
    elif op_name == "HardtanhFwdOp":
        out, ref = ew.HardtanhFwdOp()(x), F.hardtanh(x)
    else:
        out, ref = ew.HardsigmoidFwdOp()(x), F.hardsigmoid(x)
    assert torch.isnan(ref).any()
    compare_outputs(out, ref, ElementwiseWorkload(name, (x,)).verification(x))


@pytest.mark.smoke
def test_nan_to_num_stores_an_out_of_range_replacement_as_inf() -> None:
    """``torch.nan_to_num`` casts a replacement to the element type, overflowing to Inf."""
    from tileops._ops.elementwise import NanToNumFwdOp

    x = torch.tensor([float("nan"), float("inf"), float("-inf"), 1.0] * 256, device=run_device())
    x = x.to(torch.float16)
    ref = torch.nan_to_num(x, nan=1e6, posinf=1e6, neginf=-1e6)
    out = NanToNumFwdOp(nan=1e6, posinf=1e6, neginf=-1e6)(x)
    compare_outputs(out, ref, ElementwiseWorkload("NanToNumFwdOp", (x,)).verification(x))


@pytest.mark.smoke
@pytest.mark.parametrize("tensor_value", [False, True], ids=["number", "tensor"])
def test_masked_fill_forward_rejects_unsupported_dtype(tensor_value: bool) -> None:
    """The manifest union is the gate; float32 is inside it and must be accepted."""
    from tileops._ops.elementwise import MaskedFillFwdOp

    op = MaskedFillFwdOp()
    mask = torch.ones(1024, device=run_device(), dtype=torch.bool)

    def value(dtype):
        return torch.tensor(0.0, device=run_device(), dtype=dtype) if tensor_value else -100.0

    for dtype in (torch.float16, torch.float32):
        x = torch.randn(1024, device=run_device(), dtype=dtype)
        assert op(x, mask, value(dtype)).dtype == dtype
    bad = torch.randn(1024, device=run_device(), dtype=torch.float64)
    with pytest.raises(ValueError, match="dtype"):
        op(bad, mask, value(torch.float64))


@pytest.mark.smoke
@pytest.mark.parametrize(
    "make_op",
    [
        pytest.param(lambda: (HardtanhFwdOp(min_val=-1e6), ()), id="hardtanh-min_val"),
        pytest.param(lambda: (HardtanhFwdOp(max_val=1e6), ()), id="hardtanh-max_val"),
        pytest.param(lambda: (ClampFwdOp(), (1e6,)), id="clamp-min"),
        pytest.param(lambda: (ClampFwdOp(), (None, 1e6)), id="clamp-max"),
    ],
)
def test_scalar_param_rejects_unrepresentable(make_op) -> None:
    """A scalar finite in float32 but not in float16 is rejected at the float16 call."""
    op, bounds = make_op()
    fp32 = torch.zeros(1024, device=run_device(), dtype=torch.float32)
    assert op(fp32, *bounds).dtype == torch.float32

    fp16 = torch.zeros(1024, device=run_device(), dtype=torch.float16)
    with pytest.raises(ValueError, match="representable"):
        op(fp16, *bounds)


@pytest.mark.skipif(
    torch.device(run_device()).type == "cpu",
    reason="mixes a run-device tensor with a cpu one, so the run device is not cpu",
)
@pytest.mark.smoke
def test_masked_fill_forward_rejects_cpu_mask() -> None:
    """MaskedFillFwdOp forward() must raise ValueError when mask is not on x's device."""
    from tileops._ops.elementwise import MaskedFillFwdOp

    op = MaskedFillFwdOp()
    x = torch.randn(1024, device=run_device(), dtype=torch.float16)
    mask = torch.ones(1024, dtype=torch.bool)  # CPU mask
    with pytest.raises(ValueError, match="needs every tensor on one device"):
        op(x, mask, -100.0)


@pytest.mark.smoke
def test_masked_fill_forward_rejects_non_bool_mask() -> None:
    """MaskedFillFwdOp forward() must raise ValueError when mask dtype is not bool."""
    from tileops._ops.elementwise import MaskedFillFwdOp

    op = MaskedFillFwdOp()
    x = torch.randn(1024, device=run_device(), dtype=torch.float16)
    mask = torch.ones(1024, device=run_device(), dtype=torch.float32)  # wrong dtype
    with pytest.raises(ValueError, match="mask dtype is not bool"):
        op(x, mask, -100.0)


@pytest.mark.smoke
def test_masked_fill_forward_rejects_a_mask_that_cannot_broadcast() -> None:
    """The manifest states the output as a broadcast, so a shape that does not fit fails."""
    from tileops._ops.elementwise import MaskedFillFwdOp

    op = MaskedFillFwdOp()
    x = torch.randn(1024, device=run_device(), dtype=torch.float16)
    mask = torch.ones(512, device=run_device(), dtype=torch.bool)  # neither shape broadcasts
    with pytest.raises(ValueError, match="not broadcastable"):
        op(x, mask, -100.0)


_MASKED_FILL_INT_DTYPES = [
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
]


def _masked_fill_int_inputs(n_total: int, dtype: torch.dtype):
    iinfo = torch.iinfo(dtype)
    lo = max(iinfo.min, -1000)
    hi = min(iinfo.max, 1000) + 1
    x = torch.randint(lo, hi, (n_total,), device=run_device(), dtype=dtype)
    mask = torch.randint(0, 2, (n_total,), device=run_device()).bool()
    return x, mask


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", _MASKED_FILL_INT_DTYPES)
def test_masked_fill_int_dtypes(dtype: torch.dtype) -> None:
    """L1: each manifest int dtype matches PyTorch on a representative fill."""
    from tileops._ops.elementwise import MaskedFillFwdOp

    n_total = 4096
    fill_value = 7
    x, mask = _masked_fill_int_inputs(n_total, dtype)
    ref = x.masked_fill(mask, fill_value)
    op = MaskedFillFwdOp()
    out = op(x, mask, fill_value)
    compare_outputs(
        out,
        ref,
        ElementwiseWorkload(type(op).__name__, (x, mask, fill_value)).verification(*(x, mask)),
    )


@pytest.mark.smoke
def test_masked_fill_uint8_wraps_negative_int() -> None:
    """uint8 wraps a negative Python int via two's complement (PyTorch: -1 -> 255)."""
    from tileops._ops.elementwise import MaskedFillFwdOp

    n_total = 4096
    x, mask = _masked_fill_int_inputs(n_total, torch.uint8)
    ref = x.masked_fill(mask, -1)
    op = MaskedFillFwdOp()
    compare_outputs(
        op(x, mask, -1),
        ref,
        ElementwiseWorkload(type(op).__name__, (x, mask, -1)).verification(*(x, mask)),
    )


@pytest.mark.smoke
def test_masked_fill_int_truncates_fractional_float() -> None:
    """Integer dtypes truncate a float fill toward zero (PyTorch: 1.5 -> 1)."""
    from tileops._ops.elementwise import MaskedFillFwdOp

    n_total = 4096
    x, mask = _masked_fill_int_inputs(n_total, torch.int32)
    ref = x.masked_fill(mask, 1.5)
    op = MaskedFillFwdOp()
    compare_outputs(
        op(x, mask, 1.5),
        ref,
        ElementwiseWorkload(type(op).__name__, (x, mask, 1.5)).verification(*(x, mask)),
    )


@pytest.mark.smoke
@pytest.mark.parametrize("fill_value", [True, 2, 0])
def test_masked_fill_bool(fill_value) -> None:
    """L1: bool masked_fill takes a bool and coerces a non-zero Number to True."""
    from tileops._ops.elementwise import MaskedFillFwdOp

    n_total = 4096
    x = torch.randint(0, 2, (n_total,), device=run_device()).bool()
    mask = torch.randint(0, 2, (n_total,), device=run_device()).bool()
    ref = x.masked_fill(mask, fill_value)
    op = MaskedFillFwdOp()
    out = op(x, mask, fill_value)
    compare_outputs(
        out,
        ref,
        ElementwiseWorkload(type(op).__name__, (x, mask, fill_value)).verification(*(x, mask)),
    )


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype, fill_value",
    [
        pytest.param(torch.float16, float("inf"), id="fp16-inf"),
        pytest.param(torch.bfloat16, float("-inf"), id="bf16-neg-inf"),
        pytest.param(torch.float32, float("nan"), id="fp32-nan"),
    ],
)
def test_masked_fill_float_nonfinite(dtype: torch.dtype, fill_value: float) -> None:
    """L4: +/-Inf and NaN fill values pass through unchanged (no clamp)."""
    from tileops._ops.elementwise import MaskedFillFwdOp

    n_total = 4096
    x = torch.randn(n_total, device=run_device(), dtype=dtype)
    mask = torch.randint(0, 2, (n_total,), device=run_device()).bool()
    ref = x.masked_fill(mask, fill_value)
    op = MaskedFillFwdOp()
    out = op(x, mask, fill_value)
    compare_outputs(
        out,
        ref,
        ElementwiseWorkload(type(op).__name__, (x, mask, fill_value)).verification(*(x, mask)),
    )


_MASKED_FILL_REJECT_CASES = [
    pytest.param(torch.int8, 200, id="signed-int-overflow"),
    pytest.param(torch.int8, 128.0, id="signed-int-float-just-over"),
    pytest.param(torch.uint8, -256, id="uint8-int-wrap-too-low"),
    pytest.param(torch.uint8, -1.0, id="uint8-float-negative"),
    pytest.param(torch.int32, float("inf"), id="int-inf"),
    pytest.param(torch.int32, float("nan"), id="int-nan"),
]


@pytest.mark.smoke
@pytest.mark.parametrize("dtype, fill_value", _MASKED_FILL_REJECT_CASES)
def test_masked_fill_rejects_when_pytorch_rejects(
    dtype: torch.dtype,
    fill_value,
) -> None:
    """The op rejects, with the representability error, every scalar torch rejects."""
    from tileops._ops.elementwise import MaskedFillFwdOp

    pytorch_mask = torch.tensor([True], device=run_device())
    pytorch_tensor = torch.zeros(1, device=run_device(), dtype=dtype)
    with pytest.raises(Exception):  # noqa: B017
        pytorch_tensor.masked_fill(pytorch_mask, fill_value)

    op = MaskedFillFwdOp()
    x = torch.zeros(1024, device=run_device(), dtype=dtype)
    mask = torch.zeros(1024, device=run_device(), dtype=torch.bool)
    with pytest.raises(ValueError, match="representable"):
        op(x, mask, fill_value)


@pytest.mark.smoke
def test_where_rejects_fp8_dtype() -> None:
    """WhereFwdOp rejects a dtype outside its manifest union."""
    bad_dtype = torch.float8_e4m3fn
    from tileops._ops.elementwise import WhereFwdOp

    shape = (4, 8)
    op = WhereFwdOp()
    cond = torch.zeros(shape, device=run_device(), dtype=torch.bool)
    x = torch.zeros(shape, device=run_device()).to(bad_dtype)
    with pytest.raises((ValueError, TypeError)):
        op(cond, x, x)


@pytest.mark.smoke
@pytest.mark.skipif(not run_device_available(), reason="the run device is not available")
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32],
)
def test_where_accepts_manifest_dtypes(dtype: torch.dtype) -> None:
    """WhereFwdOp constructs and runs for every manifest-declared dtype."""
    from tileops._ops.elementwise import WhereFwdOp

    shape = (4, 8)
    cond = torch.randint(0, 2, shape, device=run_device()).bool()
    inp = torch.randn(shape, device=run_device(), dtype=dtype)
    other = torch.randn(shape, device=run_device(), dtype=dtype)
    op = WhereFwdOp()
    out = op(cond, inp, other)
    ref = torch.where(cond, inp, other)
    compare_outputs(
        out,
        ref,
        ElementwiseWorkload(type(op).__name__, (cond, inp, other)).verification(
            *(cond, inp, other)
        ),
    )
