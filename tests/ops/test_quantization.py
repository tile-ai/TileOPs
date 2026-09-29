from functools import partial

import pytest
import torch

from tests.test_base import TestBase, allclose_compare, exact_compare
from tileops.backend import OpNotAvailableError
from tileops.kernels.quantization import (
    INT8QuantPerChannelFwdKernel,
    INT8QuantPerTensorFwdKernel,
    QuantizeCall,
)
from tileops.ops import GemmW4A16FwdOp
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
from workloads.gemm import unrepack_w4a16_weight
from workloads.quantization import (
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


def _fp8_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Within one ``float8_e4m3fn`` mantissa step."""
    allclose_compare(output.float(), output_ref.float(), atol=2.0**-9, rtol=2.0**-3)


_COMPARE = {
    INT8QuantPerBlockFwdOp: [exact_compare, exact_compare],
    FP8QuantPerBlockFwdOp: [_fp8_compare, _scale_compare],
    INT4QuantPerGroupFwdOp: [exact_compare, exact_compare, exact_compare],
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


def _case(op_cls, rows, cols, dtype):
    return pytest.param(op_cls, rows, cols, dtype, id=f"{op_cls.__name__}-{dtype}")


@pytest.mark.smoke
@pytest.mark.parametrize(
    "op_cls, rows, cols, dtype",
    [
        _case(INT8QuantPerTensorFwdOp, 128, 1024, torch.float16),
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
        (2880, torch.bfloat16, "int8_quant_per_block_fwd"),
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
@pytest.mark.parametrize("op_cls", list(_WORKLOADS), ids=lambda c: c.__name__)
def test_generated_checks_reject_an_invalid_call(op_cls) -> None:
    inputs = _WORKLOADS[op_cls](256, 256, torch.float64).gen_inputs()
    with pytest.raises(ValueError, match="dtype"):
        op_cls()(*(t.cpu() for t in inputs))


@pytest.mark.smoke
def test_int4_reference_round_trips_a_group_of_one_sign() -> None:
    """A group of one sign round-trips within half a step, and a tiny range keeps a scale."""
    w = torch.empty(2, 128, dtype=torch.float16, device=run_device())
    w[0] = torch.linspace(1.0, 1.5, 128)
    w[1] = torch.linspace(0.0, 1e-5, 128)
    packed, scale, zero = int4_quant_per_group(w, 128)
    raw = unrepack_w4a16_weight(packed).to(torch.int32)
    q = torch.stack((raw & 0xF, raw >> 4), dim=-1).view(2, 128)
    step = scale.float()
    restored = (q - zero.to(torch.int32)) * step
    assert (step > 0).all()
    assert ((restored - w.float()).abs() <= step / 2 + 1e-6).all()


def _int4_special_groups(w: torch.Tensor) -> torch.Tensor:
    """128-element groups on the edges of the grid: all zero (scale 1), constant (the range
    widened to 0), one sign, below the scale floor, rounding ties, and a clamp at 15."""
    w = w.clone()
    groups = w.view(-1, 128)
    groups[0] = 0
    groups[1] = 0.75
    groups[2] = groups[2].abs()
    groups[3] = torch.linspace(0.0, 1e-4, 128)
    # Scale 1.5 and zero point 8 from a tie; every other element is a tie w / 1.5 = n + 1/2.
    groups[4] = (torch.arange(128, device=w.device) % 15 - 7.5) * 1.5
    groups[4, :2] = torch.tensor([-11.25, 11.25])
    # The scale rounds down to 0.11761474609375, so hi / scale + zero rounds to 16.
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
    test.check(INT4QuantPerGroupFwdOp(group_size), make(w), compare=[exact_compare] * 3)


@pytest.mark.smoke
@pytest.mark.parametrize("group_size", [128, 1024])
def test_int4_per_group_round_trip(group_size: int) -> None:
    """``GemmW4A16FwdOp`` with the op's outputs reproduces ``w`` within the grid's error.

    The activation is the identity, so output ``(k, n)`` is the dequantized ``w[n, k]``
    rounded to float16 once, and the reference ``torch.matmul`` in float32 is ``w[n, k]``.
    Rounding to the nearest code moves ``w`` by at most half a step ``s`` of its group. The
    float16 scale may round below the range by 2^-11 of itself, which lets the largest
    element round to code 16 and clamp to 15: at most ``15 * 2^-11 * s`` more. The float32
    roundings of the range, its product by 1/15 and the quotients add less than
    ``2^-20 * s``, and the float16 output adds ``2^-11`` of its magnitude.
    """
    rows, cols = 256, 1024
    groups = rows * cols // group_size
    magnitude = torch.logspace(-2, 2, groups, device=run_device())
    offset = torch.rand(groups, device=run_device()) - 0.5
    w = (torch.randn(groups, group_size, device=run_device()) + offset[:, None]) * magnitude[
        :, None
    ]
    w = w.view(rows, cols).half()
    packed, scale, zero = INT4QuantPerGroupFwdOp(group_size)(w)
    # The GEMM reads 128-element groups; a wider group is the same scale and zero repeated.
    repeat = group_size // 128
    eye = torch.eye(cols, dtype=torch.float16, device=run_device())
    out = GemmW4A16FwdOp()(
        eye,
        packed,
        scale.repeat_interleave(repeat, 1).contiguous(),
        zero.repeat_interleave(repeat, 1).contiguous(),
    )
    ref = torch.matmul(eye.float(), w.float().T)
    step = scale.float().repeat_interleave(group_size, 1).T
    bound = step * (0.5 + 15 * 2**-11 + 2**-20) + 2**-11 * out.float().abs()
    err = (out.float() - ref).abs()
    assert (err <= bound).all(), f"max excess {(err - bound).max().item()}"
