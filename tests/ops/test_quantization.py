from functools import partial

import pytest
import torch

from tests.test_base import TestBase, allclose_compare, exact_compare
from tileops.backend import BUILTIN, OpNotAvailableError
from tileops.kernels.quantization import (
    INT8QuantPerChannelFwdKernel,
    INT8QuantPerTensorFwdKernel,
    QuantizeCall,
)
from tileops.quantization import (
    FP8QuantPerBlockFwdOp,
    INT4QuantPerGroupFwdOp,
    INT8DequantPerBlockFwdOp,
    INT8DequantPerChannelFwdOp,
    INT8DequantPerTensorFwdOp,
    INT8QuantPerBlockFwdOp,
    INT8QuantPerChannelFwdOp,
    INT8QuantPerTensorFwdOp,
    SmoothQuantFwdOp,
)
from workloads.device import run_device
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
    INT8QuantPerBlockFwdOp: INT8QuantPerBlockWorkload,
    FP8QuantPerBlockFwdOp: FP8QuantPerBlockWorkload,
    INT4QuantPerGroupFwdOp: INT4QuantPerGroupWorkload,
    SmoothQuantFwdOp: SmoothQuantWorkload,
}

_scale_compare = partial(allclose_compare, atol=0.0, rtol=1e-6)


def _bitwise_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Bit for bit, so that NaN codes and signed zeros compare too."""
    assert output.dtype == output_ref.dtype and output.shape == output_ref.shape
    assert torch.equal(output.view(torch.uint8), output_ref.view(torch.uint8))


_COMPARE = {
    INT8QuantPerBlockFwdOp: [exact_compare, exact_compare],
    FP8QuantPerBlockFwdOp: [_bitwise_compare, _bitwise_compare],
    INT4QuantPerGroupFwdOp: [exact_compare, _scale_compare],
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


def _case(op_cls, rows, cols, dtype, marks=()):
    return pytest.param(op_cls, rows, cols, dtype, marks=marks, id=f"{op_cls.__name__}-{dtype}")


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
        _case(INT8QuantPerBlockFwdOp, 64, 1024, torch.float16),
        _case(INT8QuantPerBlockFwdOp, 64, 1024, torch.bfloat16),
        _case(INT8QuantPerBlockFwdOp, 64, 1024, torch.float32),
        _case(FP8QuantPerBlockFwdOp, 256, 512, torch.bfloat16),
        _case(FP8QuantPerBlockFwdOp, 256, 512, torch.float16),
        _case(FP8QuantPerBlockFwdOp, 256, 512, torch.float32),
        _case(INT4QuantPerGroupFwdOp, 256, 1024, torch.float16),
        _case(SmoothQuantFwdOp, 64, 1024, torch.float16),
        _case(SmoothQuantFwdOp, 64, 1024, torch.bfloat16),
    ],
)
def test_quantize_matches_reference(op_cls, rows, cols, dtype) -> None:
    test = type("QuantizeTest", (_WORKLOADS[op_cls], TestBase), {})(rows, cols, dtype)
    compare = _COMPARE.get(op_cls, [exact_compare, _scale_compare])
    try:
        test.check(op_cls(), *test.gen_inputs(), compare=compare)
    except OpNotAvailableError as e:
        pytest.skip(str(e))


class _DefaultPolicyLoads(INT8QuantPerChannelFwdKernel):
    """The default-policy loads the launch policy takes only above 96 MB of ``w``."""

    @property
    def default_config(self) -> dict:
        return {**super().default_config, "evict_first": False}


@pytest.mark.smoke
@pytest.mark.parametrize(
    "rows, cols, dtype, make, kernel",
    [
        # The manifest's convention for an all-zero row: scale 1.0, q all zero.
        pytest.param(
            64,
            1024,
            torch.float16,
            lambda w: w * (torch.arange(64, device=w.device) % 2 == 0)[:, None],
            None,
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
            None,
            id="misaligned-start",
        ),
        # An odd K: rows start inside a vector, and the storage ends inside the last one.
        pytest.param(37, 999, torch.bfloat16, lambda w: w, None, id="odd-k"),
        # A K shorter than a vector, which then holds codes of several rows.
        pytest.param(37, 3, torch.float16, lambda w: w, None, id="k-below-vector"),
        # Whole vectors that do not split evenly over the threads: the last ones idle.
        pytest.param(37, 264, torch.bfloat16, lambda w: w, None, id="aligned-inexact"),
        # Subnormal rows and scales, which the quotient is scaled out of and clamped.
        pytest.param(64, 1024, torch.float32, _subnormal, None, id="subnormal-scale"),
        pytest.param(
            64, 1024, torch.bfloat16, lambda w: w, _DefaultPolicyLoads, id="default-policy-loads"
        ),
    ],
)
def test_int8_quant_per_channel_edge_inputs(rows, cols, dtype, make, kernel) -> None:
    test = type("QuantizeTest", (INT8QuantPerChannelWorkload, TestBase), {})(rows, cols, dtype)
    (w,) = test.gen_inputs()
    kernel_map = {"int8_quant_per_channel_fwd": kernel} if kernel else None
    op = INT8QuantPerChannelFwdOp(kernel_map=kernel_map)
    test.check(op, make(w), compare=[exact_compare, exact_compare])


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


class _OneTileHeldEach(INT8QuantPerTensorFwdKernel):
    """One tile per thread in registers and one in shared memory, so a small input also
    reaches the tiles read again after the grid barrier."""

    @property
    def default_config(self) -> dict:
        return {"reg_tiles": 1, "smem_tiles": 1, "batch": 3}


@pytest.mark.smoke
@pytest.mark.in_tree_kernels
# bfloat16 is held as 32-bit words of two elements, float16 element by element.
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_int8_quant_per_tensor_reaches_every_residency(dtype) -> None:
    """Registers, shared memory, the re-read tiles in full batches and a partial one, a
    ragged last round of tiles and a tail that ends in a partial vector, on 114 or 132 SMs."""
    test = type("QuantizeTest", (INT8QuantPerTensorWorkload, TestBase), {})(1497, 4995, dtype)
    op = INT8QuantPerTensorFwdOp(kernel_map={"int8_quant_per_tensor_fwd": _OneTileHeldEach})
    test.check(op, *test.gen_inputs(), compare=[exact_compare, exact_compare])


@pytest.mark.smoke
@pytest.mark.parametrize(
    "dtype, make",
    [
        # The manifest's convention for an all-zero input: scale 1.0, q all zero.
        pytest.param(torch.float16, torch.zeros_like, id="all-zero"),
        # A storage start off the 16-byte vector boundary.
        pytest.param(
            torch.bfloat16,
            lambda x: (
                torch.empty(x.numel() + 1, dtype=x.dtype, device=x.device)[1:]
                .copy_(x.reshape(-1))
                .view(x.shape)
            ),
            id="misaligned-start",
        ),
        # Subnormal inputs and scale, which the quotient is scaled out of and clamped.
        pytest.param(torch.float32, _subnormal, id="subnormal-scale"),
        # A scale that underflows to zero, which torch divides by.
        pytest.param(torch.float32, lambda x: x * 1e-45, id="zero-scale"),
    ],
)
def test_int8_quant_per_tensor_edge_inputs(dtype, make) -> None:
    test = type("QuantizeTest", (INT8QuantPerTensorWorkload, TestBase), {})(64, 1024, dtype)
    (x,) = test.gen_inputs()
    test.check(INT8QuantPerTensorFwdOp(), make(x), compare=[exact_compare, exact_compare])


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


def _misaligned(x: torch.Tensor) -> torch.Tensor:
    """*x* copied into a contiguous view that starts one element into its storage."""
    flat = torch.empty(x.numel() + 1, dtype=x.dtype, device=x.device)
    view = flat[1:].view(x.shape)
    view.copy_(x)
    return view


@pytest.mark.smoke
@pytest.mark.parametrize(
    "rows, cols, dtype, make",
    [
        # The manifest's convention for an all-zero block: scale 1.0, q all zero.
        pytest.param(64, 1024, torch.float16, _zero_even_blocks, id="zero-blocks"),
        # Round half to even, on exact ties.
        pytest.param(64, 1024, torch.float16, _ties, id="rounding-ties"),
        # A storage start off the 16-byte vector boundary.
        pytest.param(64, 1024, torch.bfloat16, _misaligned, id="misaligned-start"),
        # Blocks on the vector boundary, a partial last block of an odd number of vectors,
        # and a partial last CTA.
        pytest.param(17, 1000, torch.bfloat16, lambda x: x, id="partial-last-block"),
        # Blocks that start inside a vector, all-zero rows, and a storage end inside the
        # last vector.
        pytest.param(37, 999, torch.float16, _zero_even_rows, id="odd-k"),
        # Subnormal scales, which the quotient is scaled out of and clamped, in both
        # kernels: blocks on the vector boundary and blocks inside a vector.
        pytest.param(64, 1024, torch.float32, _subnormal, id="subnormal-scale"),
        pytest.param(37, 999, torch.float32, _subnormal, id="subnormal-scale-odd-k"),
    ],
)
def test_int8_quant_per_block_edge_inputs(rows, cols, dtype, make) -> None:
    test = type("QuantizeTest", (INT8QuantPerBlockWorkload, TestBase), {})(rows, cols, dtype)
    (x,) = test.gen_inputs()
    test.check(INT8QuantPerBlockFwdOp(), make(x), compare=[exact_compare, exact_compare])


@pytest.mark.smoke
@pytest.mark.parametrize(
    "cols, dtype, key",
    [
        (7168, torch.bfloat16, "int8_quant_per_block_fwd"),
        (4100, torch.float32, "int8_quant_per_block_fwd"),
        (4099, torch.float16, "int8_quant_per_block_shifted_fwd"),
        (4098, torch.float32, "int8_quant_per_block_shifted_fwd"),
    ],
)
def test_int8_quant_per_block_each_region_selects_its_one_implementation(
    cols: int, dtype: torch.dtype, key: str
) -> None:
    """Blocks that start on a 16-byte vector take the register program, any other the
    shifted one."""
    call = QuantizeCall(arch=90, sm_count=132, rows=64, cols=cols, dtype=dtype)
    assert INT8QuantPerBlockFwdOp().select_implementation("int8_quant_per_block_fwd", call) == key


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
    q, scale = INT8QuantPerBlockFwdOp()(x)
    x_hat = INT8DequantPerBlockFwdOp(dtype)(q, scale)
    step = scale.repeat_interleave(128, 1)[:, :cols]
    bound = step * (0.5 + 2**-20) + torch.finfo(dtype).eps * x.float().abs()
    err = (x_hat.float() - x.float()).abs()
    assert (err <= bound).all(), f"max excess {(err - bound).max().item()}"


@pytest.mark.smoke
@pytest.mark.parametrize(
    "rows, cols, dtype, offset, ieee",
    [
        # An odd K over more rows than the SMs hold at once: rows start inside a vector, and
        # a CTA quantizes several rows at one column offset.
        pytest.param(1000, 999, torch.bfloat16, 0, False, id="odd-k"),
        # Whole vectors that do not split evenly over the threads: the last ones idle.
        pytest.param(37, 264, torch.float16, 0, False, id="aligned-inexact"),
        # A copy of smooth in a view that starts one element into its storage, off the
        # 16-byte vector boundary.
        pytest.param(64, 1024, torch.bfloat16, 1, False, id="misaligned-smooth"),
        # Factors below 2**-60 and an all-zero row, which IEEE division serves.
        pytest.param(64, 1024, torch.bfloat16, 0, True, id="ieee-division"),
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
    test.check(SmoothQuantFwdOp(), x, shifted, compare=[exact_compare, exact_compare])


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op, interface",
    [
        (INT8QuantPerChannelFwdOp(target=BUILTIN), "int8_quant_per_channel_fwd"),
        (SmoothQuantFwdOp(target=BUILTIN), "smooth_quant_fwd"),
    ],
    ids=lambda v: v if isinstance(v, str) else type(v).__name__,
)
def test_each_per_row_region_selects_its_one_implementation(op, interface: str) -> None:
    """Every call whose elements int32 indexes has the one per-row program; a larger has none."""
    served = QuantizeCall(arch=90, sm_count=132, rows=4096, cols=4096, dtype=torch.bfloat16)
    assert op.select_implementation(interface, served) == interface
    too_large = QuantizeCall(arch=90, sm_count=132, rows=2**16, cols=2**15, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="int32"):
        op.select_implementation(interface, too_large)


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


@pytest.mark.smoke
@pytest.mark.parametrize(
    "rows, cols, group_size, make",
    [
        pytest.param(64, 1024, 128, _int4_special_groups, id="special-groups"),
        # Two lanes to a group, and a partial last CTA.
        pytest.param(37, 1024, 64, lambda w: w, id="group-64"),
        # One CTA of two warps to a row, whose 68 chunks leave most slots of the second empty.
        pytest.param(19, 2176, 2176, _int4_special_groups, id="per-channel-uneven"),
        pytest.param(64, 1024, 128, _misaligned, id="misaligned-start"),
    ],
)
def test_int4_quant_per_group_edge_inputs(rows, cols, group_size, make) -> None:
    test = type("QuantizeTest", (INT4QuantPerGroupWorkload, TestBase), {})(
        rows, cols, torch.float16, group_size
    )
    (w,) = test.gen_inputs()
    test.check(INT4QuantPerGroupFwdOp(group_size), make(w), compare=[exact_compare, _scale_compare])


@pytest.mark.smoke
@pytest.mark.parametrize("group_size", [128, 1024])
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


@pytest.mark.smoke
@pytest.mark.parametrize(
    "rows, cols, dtype, make",
    [
        # The manifest's convention for an all-zero tile: scale 1.0, q all zero.
        pytest.param(256, 512, torch.bfloat16, _zero_odd_tiles, id="zero-tiles"),
        # Partial tiles on both axes, rows on the vector boundary.
        pytest.param(200, 392, torch.float16, lambda w: w, id="partial-tiles"),
        # Rows that start inside a vector: the unaligned kernel, and a storage end inside
        # the last vector.
        pytest.param(37, 999, torch.bfloat16, _large_first_column, id="odd-k"),
        pytest.param(37, 999, torch.float32, lambda w: w, id="odd-k-float32"),
        # A storage start off the 16-byte vector boundary.
        pytest.param(256, 512, torch.float16, _misaligned, id="misaligned-start"),
        # NaN and infinity reach scale and q as in torch; a signed zero stays signed.
        pytest.param(200, 392, torch.bfloat16, _specials, id="nan-inf-signed-zero"),
        # Subnormal and zero scales.
        pytest.param(200, 2048, torch.float32, _tiny, id="tiny-scales"),
        # A correctly rounded quotient, on exact ties.
        pytest.param(256, 512, torch.bfloat16, _fp8_ties, id="rounding-ties"),
    ],
)
def test_fp8_quant_per_block_edge_inputs(rows, cols, dtype, make) -> None:
    test = type("QuantizeTest", (FP8QuantPerBlockWorkload, TestBase), {})(rows, cols, dtype)
    (w,) = test.gen_inputs()
    test.check(FP8QuantPerBlockFwdOp(), make(w), compare=[_bitwise_compare, _bitwise_compare])


@pytest.mark.smoke
@pytest.mark.parametrize(
    "cols, dtype, key",
    [
        (4096, torch.bfloat16, "fp8_quant_per_block_fwd"),
        (2884, torch.float32, "fp8_quant_per_block_fwd"),
        (4099, torch.float16, "fp8_quant_per_block_unaligned_fwd"),
    ],
)
def test_each_fp8_per_block_region_selects_its_one_implementation(cols, dtype, key) -> None:
    """Rows that all start on a 16-byte vector take the register kernel, any other K the
    unaligned one."""
    call = QuantizeCall(arch=90, sm_count=132, rows=256, cols=cols, dtype=dtype)
    assert FP8QuantPerBlockFwdOp().select_implementation("fp8_quant_per_block_fwd", call) == key
