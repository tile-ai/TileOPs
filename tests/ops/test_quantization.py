import pytest
import torch

from tests.misaligned import misaligned
from tests.workload_test_base import FixtureBase, TestBase
from tileops.quantization import (
    FP8QuantPerBlockFwdOp,
    INT4QuantPerGroupFwdOp,
    INT8DequantPerBlockFwdOp,
    INT8DequantPerChannelFwdOp,
    INT8DequantPerTensorFwdOp,
    INT8QuantPerChannelFwdOp,
    INT8QuantPerTensorFwdOp,
    INT8QuantPerTokenGroupFwdOp,
    SmoothQuantFwdOp,
)
from workloads.device import run_device
from workloads.quantization.int8_dequant import (
    INT8DequantPerBlockWorkload,
    INT8DequantPerChannelWorkload,
    INT8DequantPerTensorWorkload,
)
from workloads.quantization.quantize import (
    FP8QuantPerBlockWorkload,
    INT4QuantPerGroupWorkload,
    INT8QuantPerBlockWorkload,
    INT8QuantPerChannelWorkload,
    INT8QuantPerTensorWorkload,
    SmoothQuantWorkload,
    int4_quant_per_group,
)

_WORKLOADS = {
    INT8QuantPerTensorFwdOp: INT8QuantPerTensorWorkload,
    INT8QuantPerChannelFwdOp: INT8QuantPerChannelWorkload,
    INT8QuantPerTokenGroupFwdOp: INT8QuantPerBlockWorkload,
    FP8QuantPerBlockFwdOp: FP8QuantPerBlockWorkload,
    INT4QuantPerGroupFwdOp: INT4QuantPerGroupWorkload,
    SmoothQuantFwdOp: SmoothQuantWorkload,
}


def _subnormal(x: torch.Tensor) -> torch.Tensor:
    """float32 subnormal elements, and an amax of 190 units of the last place in every
    128-element block.

    The scale rounds to one unit, so the quotient of the amax is 190 before the clamp.
    """
    x = (x * 2.0**-145).clone()
    unit = torch.finfo(torch.float32).smallest_normal * 2.0**-23
    x[:, 0::128] = 190 * unit
    x[:, 1::128] = -190 * unit
    return x


def _case(op_cls, rows, cols, dtype, marks=(), arch=""):
    name = f"{op_cls.__name__}-{dtype}" + (f"-{arch}" if arch else "")
    return pytest.param(op_cls, rows, cols, dtype, marks=marks, id=name)


def _fp8_cases(rows, cols, dtype):
    """The FP8 kernels run on SM89 and SM90: one case per architecture, each run there."""
    return [
        _case(FP8QuantPerBlockFwdOp, rows, cols, dtype, marks=getattr(pytest.mark, arch), arch=arch)
        for arch in ("sm89", "sm90")
    ]


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, rows, cols, dtype",
    [
        _case(
            INT8QuantPerTensorFwdOp,
            128,
            1024,
            torch.float16,
            marks=pytest.mark.packaging(family="quantization"),
        ),
        _case(INT8QuantPerTensorFwdOp, 128, 1024, torch.bfloat16),
        _case(INT8QuantPerTensorFwdOp, 128, 1024, torch.float32),
        _case(INT8QuantPerChannelFwdOp, 256, 1024, torch.float16),
        _case(INT8QuantPerChannelFwdOp, 256, 1024, torch.bfloat16),
        _case(INT8QuantPerChannelFwdOp, 256, 1024, torch.float32),
        _case(INT8QuantPerTokenGroupFwdOp, 64, 1024, torch.float16),
        _case(INT8QuantPerTokenGroupFwdOp, 64, 1024, torch.bfloat16),
        _case(INT8QuantPerTokenGroupFwdOp, 64, 1024, torch.float32),
        *_fp8_cases(256, 512, torch.bfloat16),
        *_fp8_cases(256, 512, torch.float16),
        *_fp8_cases(256, 512, torch.float32),
        _case(INT4QuantPerGroupFwdOp, 256, 1024, torch.float16),
        _case(SmoothQuantFwdOp, 64, 1024, torch.float16),
        _case(SmoothQuantFwdOp, 64, 1024, torch.bfloat16),
    ],
)
def test_quantize_matches_reference(op_cls, rows, cols, dtype) -> None:
    test = type("QuantizeTest", (_WORKLOADS[op_cls], TestBase), {})(rows, cols, dtype)
    test.check(op_cls(), *test.gen_inputs())


@pytest.mark.parametrize(
    "rows, cols, dtype, make",
    [
        # The manifest's convention for an all-zero row: scale 1.0, q all zero.
        pytest.param(
            64,
            1024,
            torch.float16,
            lambda w: w * (torch.arange(64, device=w.device) % 2 == 0)[:, None],
            marks=pytest.mark.smoke,
            id="zero-rows",
        ),
        # A copy of w in a view that starts one element into its storage, off the 16-byte
        # vector boundary.
        pytest.param(
            64,
            1024,
            torch.bfloat16,
            lambda w: (
                torch.empty(w.numel() + 1, dtype=w.dtype, device=w.device)[1:]
                .view(w.shape)
                .copy_(w)
            ),
            marks=pytest.mark.smoke,
            id="misaligned-start",
        ),
        # Subnormal rows and scales, which the quotient is scaled out of and clamped.
        pytest.param(
            64, 1024, torch.float32, _subnormal, marks=pytest.mark.smoke, id="subnormal-scale"
        ),
        # An odd K: rows start inside a vector, and the storage ends inside the last one.
        pytest.param(37, 999, torch.bfloat16, lambda w: w, marks=pytest.mark.full, id="odd-k"),
        # A K shorter than a vector, which then holds codes of several rows.
        pytest.param(
            37, 3, torch.float16, lambda w: w, marks=pytest.mark.full, id="k-below-vector"
        ),
        # Whole vectors that do not split evenly over the threads: the last ones idle.
        pytest.param(
            37, 264, torch.bfloat16, lambda w: w, marks=pytest.mark.full, id="aligned-inexact"
        ),
        # Above 96 MiB of w the loads take the default cache policy instead of evict-first.
        pytest.param(
            14336,
            4096,
            torch.bfloat16,
            lambda w: w,
            marks=pytest.mark.full,
            id="default-policy-loads",
        ),
        # 16-bit rows of 8192 or more take four vectors a thread at five CTAs an SM: a
        # bfloat16 row in whole vectors, and a float16 row of 8193 whose vectors do not.
        pytest.param(37, 8192, torch.bfloat16, lambda w: w, marks=pytest.mark.full, id="wide-row"),
        pytest.param(
            37, 8193, torch.float16, lambda w: w, marks=pytest.mark.full, id="wide-odd-row"
        ),
    ],
)
def test_int8_quant_per_channel_edge_inputs(rows, cols, dtype, make) -> None:
    test = type("QuantizeTest", (INT8QuantPerChannelWorkload, TestBase), {})(rows, cols, dtype)
    (w,) = test.gen_inputs()
    test.check(INT8QuantPerChannelFwdOp(), make(w))


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_int8_per_channel_round_trip(dtype: torch.dtype) -> None:
    """Quantizing then dequantizing moves ``x`` by at most half a step of its row's 8-bit grid.

    ``|x[m, k]| <= amax[m] = 127 * scale[m]``, so each element rounds to its nearest code
    without clamping and lands within ``scale[m] / 2``. The float32 divide and multiply add
    ``2^-24`` each: ``2^-23 * |x| <= eps(dtype) * |x|`` and ``2^-25 * scale[m]``, which the
    ``2^-20 * scale[m]`` slack covers. The cast to ``dtype`` adds half an ulp of
    ``|x| + scale[m] / 2``; a nonzero code has ``|x| >= scale[m] / 2``, so ``eps(dtype) * |x|``
    covers it, and a zero code dequantizes exactly to 0.
    """
    # Rows of different magnitude, so each row has its own grid.
    magnitude = torch.logspace(-2, 2, 256, device=run_device())[:, None]
    x = (torch.randn(256, 1024, device=run_device()) * magnitude).to(dtype)
    q, scale = INT8QuantPerChannelFwdOp()(x)
    x_hat = INT8DequantPerChannelFwdOp(dtype)(q, scale)
    bound = scale[:, None] * (0.5 + 2**-20) + torch.finfo(dtype).eps * x.float().abs()
    err = (x_hat.float() - x.float()).abs()
    assert (err <= bound).all(), f"max excess {(err - bound).max().item()}"


@pytest.mark.parametrize(
    "dtype, make",
    [
        # The manifest's convention for an all-zero input: scale 1.0, q all zero.
        pytest.param(torch.float16, torch.zeros_like, marks=pytest.mark.smoke, id="all-zero"),
        # A storage start off the 16-byte vector boundary.
        pytest.param(
            torch.bfloat16,
            lambda x: (
                torch.empty(x.numel() + 1, dtype=x.dtype, device=x.device)[1:]
                .copy_(x.reshape(-1))
                .view(x.shape)
            ),
            marks=pytest.mark.smoke,
            id="misaligned-start",
        ),
        # Subnormal inputs and scale, which the quotient is scaled out of and clamped.
        pytest.param(torch.float32, _subnormal, marks=pytest.mark.smoke, id="subnormal-scale"),
        # A scale that underflows to zero, which torch divides by.
        pytest.param(torch.float32, lambda x: x * 1e-45, marks=pytest.mark.full, id="zero-scale"),
    ],
)
def test_int8_quant_per_tensor_edge_inputs(dtype, make) -> None:
    test = type("QuantizeTest", (INT8QuantPerTensorWorkload, TestBase), {})(64, 1024, dtype)
    (x,) = test.gen_inputs()
    test.check(INT8QuantPerTensorFwdOp(), make(x))


@pytest.mark.smoke
def test_int8_per_tensor_round_trip() -> None:
    """Dequantizing ``q`` against ``scale`` restores ``x`` to within half a quantization step.

    ``|x / scale|`` is at most 127, so no element clamps and rounding leaves ``|x / scale - q|``
    at most 1/2. The float32 quotient adds at most 2**-24 * 128 to that, and the float32
    product ``q * scale`` at most 2**-24 * 127 * scale, so ``|x - q * scale|`` stays within
    ``scale * (1/2 + 2**-16)``.
    """
    x = torch.randn(1024, 2880, dtype=torch.bfloat16, device=run_device())
    q, scale = INT8QuantPerTensorFwdOp()(x)
    restored = INT8DequantPerTensorFwdOp(torch.float32)(q, scale)
    assert ((restored - x.float()).abs() <= scale * (0.5 + 2.0**-16)).all()


def _zero_even_blocks(x: torch.Tensor) -> torch.Tensor:
    """*x* with every other 128-element block of each row set to zero."""
    x = x.clone()
    x.view(x.shape[0], -1, 128)[:, ::2] = 0
    return x


def _zero_even_rows(x: torch.Tensor) -> torch.Tensor:
    """*x* with every other row set to zero, for a K that is not a whole number of blocks."""
    x = x.clone()
    x[::2] = 0
    return x


def _ties(x: torch.Tensor) -> torch.Tensor:
    """Blocks whose amax is 127, so the scale is exactly 1, and whose other elements are
    halves from -126.5 to 126.5: every quotient is a rounding tie."""
    halves = torch.arange(x.numel(), device=x.device) % 254 - 126.5
    x = halves.view(x.shape).to(x.dtype)
    x.view(x.shape[0], -1, 128)[:, :, 0] = 127
    return x


@pytest.mark.parametrize(
    "rows, cols, dtype, make",
    [
        # The manifest's convention for an all-zero block: scale 1.0, q all zero.
        pytest.param(
            64, 1024, torch.float16, _zero_even_blocks, marks=pytest.mark.smoke, id="zero-blocks"
        ),
        # A storage start off the 16-byte vector boundary.
        pytest.param(
            64, 1024, torch.bfloat16, misaligned, marks=pytest.mark.smoke, id="misaligned-start"
        ),
        # Subnormal scales, which the quotient is scaled out of and clamped.
        pytest.param(
            64, 1024, torch.float32, _subnormal, marks=pytest.mark.smoke, id="subnormal-scale"
        ),
        # Round half to even, on exact ties.
        pytest.param(64, 1024, torch.float16, _ties, marks=pytest.mark.full, id="rounding-ties"),
        # Blocks on the vector boundary, a partial last block of an odd number of vectors,
        # and a partial last CTA.
        pytest.param(
            17, 1000, torch.bfloat16, lambda x: x, marks=pytest.mark.full, id="partial-last-block"
        ),
        # Blocks that start inside a vector, all-zero rows, and a storage end inside the
        # last vector.
        pytest.param(37, 999, torch.float16, _zero_even_rows, marks=pytest.mark.full, id="odd-k"),
        # Subnormal scales in blocks that start inside a vector.
        pytest.param(
            37, 999, torch.float32, _subnormal, marks=pytest.mark.full, id="subnormal-scale-odd-k"
        ),
    ],
)
def test_int8_quant_per_block_edge_inputs(rows, cols, dtype, make) -> None:
    test = type("QuantizeTest", (INT8QuantPerBlockWorkload, TestBase), {})(rows, cols, dtype)
    (x,) = test.gen_inputs()
    test.check(INT8QuantPerTokenGroupFwdOp(), make(x))


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_int8_per_block_round_trip(dtype: torch.dtype) -> None:
    """Quantizing then dequantizing moves ``x`` by at most half a step of its block's grid.

    ``|x[m, k]| <= amax[m, b] = 127 * scale[m, b]`` for the block ``b`` holding ``k``, so each
    element rounds to its nearest code without clamping and lands within ``scale[m, b] / 2``.
    The float32 divide and multiply add ``2^-24`` each: ``2^-23 * |x| <= eps(dtype) * |x|``
    and ``2^-25 * scale[m, b]``, which the ``2^-20 * scale[m, b]`` slack covers. The cast to
    ``dtype`` adds half an ulp of ``|x| + scale[m, b] / 2``; a nonzero code has
    ``|x| >= scale[m, b] / 2``, so ``eps(dtype) * |x|`` covers it, and a zero code
    dequantizes exactly to 0.
    """
    # Blocks of different magnitude, so each block has its own grid; a partial last block.
    rows, cols = 64, 1000
    magnitude = torch.logspace(-2, 2, rows * 8, device=run_device()).view(rows, 8)
    magnitude = magnitude.repeat_interleave(128, 1)[:, :cols]
    x = (torch.randn(rows, cols, device=run_device()) * magnitude).to(dtype)
    q, scale = INT8QuantPerTokenGroupFwdOp()(x)
    x_hat = INT8DequantPerBlockFwdOp(dtype)(q, scale)
    step = scale.repeat_interleave(128, 1)[:, :cols]
    bound = step * (0.5 + 2**-20) + torch.finfo(dtype).eps * x.float().abs()
    err = (x_hat.float() - x.float()).abs()
    assert (err <= bound).all(), f"max excess {(err - bound).max().item()}"


@pytest.mark.parametrize(
    "rows, cols, dtype, offset, ieee",
    [
        # An odd K over more rows than the SMs hold at once: rows start inside a vector, and
        # a CTA quantizes several rows at one column offset.
        pytest.param(1000, 999, torch.bfloat16, 0, False, marks=pytest.mark.smoke, id="odd-k"),
        # Whole vectors that do not split evenly over the threads: the last ones idle.
        pytest.param(
            37, 264, torch.float16, 0, False, marks=pytest.mark.smoke, id="aligned-inexact"
        ),
        # Copies of x and smooth in views that start one element into their storage, off
        # the 16-byte vector boundary.
        pytest.param(
            64, 1024, torch.bfloat16, 1, False, marks=pytest.mark.full, id="misaligned-inputs"
        ),
        # Factors below 2**-60 and an all-zero row, which IEEE division serves.
        pytest.param(64, 1024, torch.bfloat16, 0, True, marks=pytest.mark.full, id="ieee-division"),
    ],
)
def test_smooth_quant_edge_inputs(rows, cols, dtype, offset, ieee) -> None:
    test = type("QuantizeTest", (SmoothQuantWorkload, TestBase), {})(rows, cols, dtype)
    x, smooth = test.gen_inputs()
    if ieee:
        # A reciprocal-corrected quotient of this pair is one float32 ulp off, and it is
        # its row's amax, so the scale differs unless the tiny factors send the row to IEEE
        # division.
        smooth.fill_(6.031807955764232e-29)
        x[0] = 1.0432512363547802e-37
        x[rows // 2] = 0
    # Signed factors; the exact scale compare also fails a divide that is not correctly
    # rounded, since each row's amax is one quotient.
    sign = torch.where(torch.rand(cols, device=smooth.device) < 0.5, -1.0, 1.0)
    shifted = torch.empty(cols + offset, dtype=smooth.dtype, device=smooth.device)[offset:]
    shifted.copy_(smooth * sign)
    if offset:
        x = misaligned(x)
    test.check(SmoothQuantFwdOp(), x, shifted)


@pytest.mark.smoke
def test_per_row_quantize_of_no_rows() -> None:
    """``M = 0`` is a valid call; one CTA per row would launch an empty grid."""
    x = torch.empty(0, 1024, dtype=torch.float16, device=run_device())
    smooth = torch.ones(1024, device=run_device())
    for q, scale in (INT8QuantPerChannelFwdOp()(x), SmoothQuantFwdOp()(x, smooth)):
        assert q.shape == x.shape and q.dtype == torch.int8
        assert scale.shape == (0,) and scale.dtype == torch.float32


@pytest.mark.smoke
@pytest.mark.parametrize("op_cls", list(_WORKLOADS), ids=lambda c: c.__name__)
def test_generated_checks_reject_an_invalid_call(op_cls) -> None:
    inputs = _WORKLOADS[op_cls](256, 256, torch.float64).gen_inputs()
    with pytest.raises(ValueError, match="dtype"):
        op_cls()(*(t.cpu() for t in inputs))


@pytest.mark.smoke
def test_int4_reference_round_trips_a_group_of_one_sign() -> None:
    """Only positive saturation may exceed half a step; tiny ranges keep a scale."""
    w = torch.empty(2, 128, dtype=torch.float16, device=run_device())
    w[0] = torch.linspace(1.0, 1.5, 128)
    w[1] = torch.linspace(0.0, 1e-5, 128)
    packed, params = int4_quant_per_group(w, 128)
    q = torch.stack((packed >> 4, (packed << 4) >> 4), dim=-1).view(2, 128)
    step = params[:, :1]
    restored = q * step + params[:, 1:]
    assert (step > 0).all()
    bound = torch.where(q == 7, step, step / 2)
    assert ((restored - w.float()).abs() <= bound + 1e-6).all()


def _int4_special_groups(w: torch.Tensor) -> torch.Tensor:
    """Constant groups, one sign, tiny ranges and signed rounding boundaries."""
    w = w.clone()
    groups = w.view(-1, 128)
    groups[0] = 0
    groups[1] = 0.75
    groups[2] = groups[2].abs()
    groups[3] = torch.linspace(0.0, 1e-4, 128)
    groups[4] = (torch.arange(128, device=w.device) % 15 - 7.5) * 1.5
    groups[4, :2] = torch.tensor([-12.0, 12.0])
    groups[5] = torch.linspace(-0.7646484375, 1.0, 128)
    return w


@pytest.mark.parametrize(
    "rows, cols, group_size, make",
    [
        pytest.param(
            64, 1024, 128, _int4_special_groups, id="special-groups", marks=pytest.mark.smoke
        ),
        # A partial last CTA on the group-size-64 path.
        pytest.param(37, 1024, 64, lambda w: w, id="group-64", marks=pytest.mark.full),
        # One CTA of two warps to a row, whose 68 chunks leave most slots of the second empty.
        pytest.param(
            19,
            2176,
            2176,
            _int4_special_groups,
            id="per-channel-uneven",
            marks=pytest.mark.full,
        ),
        pytest.param(64, 1024, 128, misaligned, id="misaligned-start", marks=pytest.mark.full),
        # Manifest workload shapes: the op's only reference check in the nightly run.
        pytest.param(14336, 4096, 128, lambda w: w, id="llama-8b-mlp-up", marks=pytest.mark.full),
        pytest.param(4096, 4096, 64, lambda w: w, id="llama-8b-q-proj", marks=pytest.mark.full),
        pytest.param(
            14336, 4096, 4096, lambda w: w, id="llama-8b-mlp-up-chan", marks=pytest.mark.full
        ),
    ],
)
def test_int4_quant_per_group_edge_inputs(rows, cols, group_size, make) -> None:
    test = type("QuantizeTest", (INT4QuantPerGroupWorkload, TestBase), {})(
        rows, cols, torch.float16, group_size
    )
    (w,) = test.gen_inputs()
    test.check(INT4QuantPerGroupFwdOp(group_size), make(w))


@pytest.mark.smoke
@pytest.mark.sm90
@pytest.mark.in_tree_kernels
def test_int4_group64_packed_words_preserve_rounding_and_nibble_order() -> None:
    from tileops.backend import BUILTIN

    test = type("QuantizeTest", (INT4QuantPerGroupWorkload, TestBase), {})(
        512, 4096, torch.float16, 64
    )
    (w,) = test.gen_inputs()
    op = INT4QuantPerGroupFwdOp(64, target=BUILTIN)
    test.check(op, _int4_special_groups(w))
    (kernel,) = op.iter_kernels()
    assert kernel.pack_intrinsic


@pytest.mark.parametrize(
    "group_size",
    [pytest.param(128, marks=pytest.mark.smoke), pytest.param(1024, marks=pytest.mark.full)],
)
def test_int4_per_group_round_trip(group_size: int) -> None:
    """Nearest rounding is within half a step, except the saturated positive endpoint."""
    rows, cols = 256, 1024
    groups = rows * cols // group_size
    magnitude = torch.logspace(-2, 2, groups, device=run_device())
    offset = torch.rand(groups, device=run_device()) - 0.5
    w = (torch.randn(groups, group_size, device=run_device()) + offset[:, None]) * magnitude[
        :, None
    ]
    w = w.view(rows, cols).half()
    packed, params = INT4QuantPerGroupFwdOp(group_size)(w)
    codes = torch.stack((packed >> 4, (packed << 4) >> 4), dim=-1).view(groups, group_size)
    restored = codes.float() * params[:, :1] + params[:, 1:]
    error = (restored - w.float().view(groups, group_size)).abs()
    step = params[:, :1]
    bound = torch.where(codes == 7, step, step / 2)
    assert (error <= bound + step * 1e-5).all()


def _zero_odd_tiles(w: torch.Tensor) -> torch.Tensor:
    """*w* with every other 128x128 tile of each tile row set to zero."""
    w = w.clone()
    for j in range(0, w.shape[1], 256):
        w[:, j : j + 128] = 0
    return w


def _specials(w: torch.Tensor) -> torch.Tensor:
    """*w* with a NaN and +-inf in one tile, +-inf in two others, and signed zeros in every
    row."""
    w = w.clone()
    w[::3] = -0.0
    w[5, 7] = float("nan")
    w[6, 8] = float("inf")
    w[7, 9] = -float("inf")
    w[w.shape[0] - 1, w.shape[1] - 1] = float("inf")
    w[10, 200] = -float("inf")
    return w


def _large_first_column(w: torch.Tensor) -> torch.Tensor:
    """*w* with its first column a thousand times larger, which the partial last tile of the
    row above must not see."""
    w = w.clone()
    w[:, 0] *= 1000
    return w


def _fp8_ties(w: torch.Tensor) -> torch.Tensor:
    """Tiles of amax 3 whose other elements are +-3 * 2**-16: their quotient is 3.5 * 2**-9,
    a tie between two float8 codes that the product with the rounded reciprocal of the
    scale misses."""
    w = torch.where(torch.rand_like(w, dtype=torch.float32) < 0.5, 1.0, -1.0) * 3 * 2.0**-16
    w = w.to(torch.bfloat16)
    w[::128, ::128] = 3.0
    return w


def _tiny(w: torch.Tensor) -> torch.Tensor:
    """float32 tiles of magnitude 2**-120 to 2**-149 along K: their scales are subnormal or
    round to zero, which the IEEE divide serves; the last tile holds only the smallest
    subnormal, so its scale is zero."""
    exponent = torch.linspace(-120, -149, w.shape[1], device=w.device)
    w = w * torch.exp2(exponent)
    w[:, -128:] = torch.finfo(torch.float32).smallest_normal * 2.0**-23
    return w


@pytest.mark.parametrize(
    "rows, cols, dtype, make",
    [
        # The manifest's convention for an all-zero tile: scale 1.0, q all zero.
        pytest.param(
            256, 512, torch.bfloat16, _zero_odd_tiles, marks=pytest.mark.smoke, id="zero-tiles"
        ),
        # Partial tiles on both axes, rows on the vector boundary.
        pytest.param(
            200, 392, torch.float16, lambda w: w, marks=pytest.mark.smoke, id="partial-tiles"
        ),
        # Rows that start inside a vector: the unaligned kernel, and a storage end inside
        # the last vector.
        pytest.param(
            37, 999, torch.float32, lambda w: w, marks=pytest.mark.smoke, id="odd-k-float32"
        ),
        pytest.param(
            37, 999, torch.bfloat16, _large_first_column, marks=pytest.mark.full, id="odd-k"
        ),
        # A storage start off the 16-byte vector boundary.
        pytest.param(
            256, 512, torch.float16, misaligned, marks=pytest.mark.full, id="misaligned-start"
        ),
        # NaN and infinity reach scale and q as in torch; a signed zero stays signed.
        pytest.param(
            200, 392, torch.bfloat16, _specials, marks=pytest.mark.full, id="nan-inf-signed-zero"
        ),
        # Subnormal and zero scales.
        pytest.param(200, 2048, torch.float32, _tiny, marks=pytest.mark.full, id="tiny-scales"),
        # A correctly rounded quotient, on exact ties.
        pytest.param(
            256, 512, torch.bfloat16, _fp8_ties, marks=pytest.mark.full, id="rounding-ties"
        ),
    ],
)
def test_fp8_quant_per_block_edge_inputs(rows, cols, dtype, make) -> None:
    test = type("QuantizeTest", (FP8QuantPerBlockWorkload, TestBase), {})(rows, cols, dtype)
    (w,) = test.gen_inputs()
    test.check(FP8QuantPerBlockFwdOp(), make(w))


class INT8DequantPerTensorTest(INT8DequantPerTensorWorkload, TestBase):
    pass


class INT8DequantPerChannelTest(INT8DequantPerChannelWorkload, TestBase):
    pass


class INT8DequantPerBlockTest(INT8DequantPerBlockWorkload, TestBase):
    pass


_DEQUANT_TESTS = {
    INT8DequantPerTensorFwdOp: INT8DequantPerTensorTest,
    INT8DequantPerChannelFwdOp: INT8DequantPerChannelTest,
    INT8DequantPerBlockFwdOp: INT8DequantPerBlockTest,
}


class INT8DequantFixture(FixtureBase):
    @classmethod
    def get_params(cls):
        # One typical shape per op; the per-block K is not a multiple of 128, so the last block of
        # each row is partial.
        shapes = {
            INT8DequantPerTensorFwdOp: (1024, 1024),
            INT8DequantPerChannelFwdOp: (256, 1024),
            INT8DequantPerBlockFwdOp: (256, 1000),
        }
        return [
            (
                "op_cls, m, k, out_dtype",
                [
                    pytest.param(op_cls, *shape, dtype, marks=pytest.mark.smoke)
                    for op_cls, shape in shapes.items()
                    for dtype in (torch.float16, torch.bfloat16, torch.float32)
                ]
                # Per-channel rows shorter than a thread's eight codes, over whole blocks: no
                # block takes the vector path.
                + [
                    pytest.param(
                        INT8DequantPerChannelFwdOp, 512, 5, torch.bfloat16, marks=pytest.mark.full
                    ),
                    # Per-block rows shorter than a vector: every block converts code by code.
                    pytest.param(
                        INT8DequantPerBlockFwdOp, 512, 5, torch.bfloat16, marks=pytest.mark.full
                    ),
                    # Per-block, K % 128 = 3: a vector crosses a row's short last group and the
                    # row end, so its codes take three scales.
                    pytest.param(
                        INT8DequantPerBlockFwdOp, 16, 131, torch.bfloat16, marks=pytest.mark.full
                    ),
                    # Per-tensor: scalar-only output.
                    pytest.param(
                        INT8DequantPerTensorFwdOp, 1, 5, torch.bfloat16, marks=pytest.mark.full
                    ),
                    # Per-tensor: one full block followed by a scalar-only tail.
                    pytest.param(
                        INT8DequantPerTensorFwdOp, 1, 2051, torch.float32, marks=pytest.mark.full
                    ),
                    # Per-tensor, past the small-matrix kernel's region, with a tail whose length
                    # is not a multiple of any vector width.
                    pytest.param(
                        INT8DequantPerTensorFwdOp,
                        2051,
                        1025,
                        torch.bfloat16,
                        marks=pytest.mark.full,
                    ),
                    # Per-block, past the small-matrix kernel's region, float32 with K % 128 = 1:
                    # the staged kernel with three scales per vector and a tail.
                    pytest.param(
                        INT8DequantPerBlockFwdOp, 4100, 129, torch.float32, marks=pytest.mark.full
                    ),
                ],
            ),
        ]


@INT8DequantFixture
def test_int8_dequant_op(op_cls: type, m: int, k: int, out_dtype: torch.dtype) -> None:
    test = _DEQUANT_TESTS[op_cls](m, k, out_dtype)
    # One float32 multiply and one cast: a conforming kernel is bit-exact.
    test.check(op_cls(out_dtype), *test.gen_inputs())


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, scale_shape",
    [
        (INT8DequantPerTensorFwdOp, (2,)),
        (INT8DequantPerChannelFwdOp, (1,)),
        (INT8DequantPerBlockFwdOp, (64, 2)),
    ],
)
def test_int8_dequant_rejects_wrong_scale_shape(op_cls: type, scale_shape: tuple) -> None:
    q = torch.zeros(64, 1000, dtype=torch.int8, device=run_device())
    scale = torch.ones(scale_shape, dtype=torch.float32, device=run_device())
    with pytest.raises(ValueError, match="scale"):
        op_cls(torch.bfloat16)(q, scale)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls", [INT8DequantPerChannelFwdOp, INT8DequantPerTensorFwdOp, INT8DequantPerBlockFwdOp]
)
def test_int8_dequant_misaligned_input(op_cls: type) -> None:
    """A ``q`` whose storage is off the vector boundary, rows that split a thread's codes, a tail."""
    test = _DEQUANT_TESTS[op_cls](5, 1001, torch.bfloat16)
    q, scale = test.gen_inputs()
    q = misaligned(q)
    test.check(op_cls(torch.bfloat16), q, scale)
