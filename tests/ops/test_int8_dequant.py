import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.backend import OpNotAvailableError
from tileops.quantization import (
    INT8DequantPerBlockFwdOp,
    INT8DequantPerChannelFwdOp,
    INT8DequantPerTensorFwdOp,
    INT8QuantPerChannelFwdOp,
)
from workloads.device import run_device
from workloads.int8_dequant import (
    INT8DequantPerBlockWorkload,
    INT8DequantPerChannelWorkload,
    INT8DequantPerTensorWorkload,
)
from workloads.quantization import int8_quant_per_channel


class INT8DequantPerTensorTest(INT8DequantPerTensorWorkload, TestBase):
    pass


class INT8DequantPerChannelTest(INT8DequantPerChannelWorkload, TestBase):
    pass


class INT8DequantPerBlockTest(INT8DequantPerBlockWorkload, TestBase):
    pass


_TESTS = {
    INT8DequantPerTensorFwdOp: INT8DequantPerTensorTest,
    INT8DequantPerChannelFwdOp: INT8DequantPerChannelTest,
    INT8DequantPerBlockFwdOp: INT8DequantPerBlockTest,
}


# One typical shape per op; the per-block K is not a multiple of 128, so the last block of
# each row is partial.
_SHAPES = {
    INT8DequantPerTensorFwdOp: (256, 1024),
    INT8DequantPerChannelFwdOp: (256, 1024),
    INT8DequantPerBlockFwdOp: (256, 1000),
}


class INT8DequantFixture(FixtureBase):
    PARAMS = [
        (
            "op_cls, m, k, out_dtype",
            [
                pytest.param(op_cls, *shape, dtype, marks=pytest.mark.smoke)
                for op_cls, shape in _SHAPES.items()
                for dtype in (torch.float16, torch.bfloat16, torch.float32)
            ]
            # Per-channel rows shorter than a thread's eight codes, over whole blocks: no
            # block takes the vector path.
            + [
                pytest.param(
                    INT8DequantPerChannelFwdOp, 512, 5, torch.bfloat16, marks=pytest.mark.smoke
                )
            ],
        ),
    ]


@INT8DequantFixture
def test_int8_dequant_op(op_cls: type, m: int, k: int, out_dtype: torch.dtype) -> None:
    test = _TESTS[op_cls](m, k, out_dtype)
    op = op_cls(out_dtype)
    try:
        # One float32 multiply and one cast: a conforming kernel is bit-exact.
        test.check(op, *test.gen_inputs(), atol=0, rtol=0)
    except OpNotAvailableError:
        if op_cls.kernel_types:
            raise
        pytest.skip(f"{op_cls.__name__} has no in-tree kernel")


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
def test_int8_dequant_per_channel_misaligned_input() -> None:
    """A ``q`` whose storage is off the vector boundary, rows that split a thread's codes, a tail."""
    test = INT8DequantPerChannelTest(5, 1001, torch.bfloat16)
    q, scale = test.gen_inputs()
    q = torch.cat([q.new_zeros(1, 1001), q])[1:]
    assert q.is_contiguous() and q.data_ptr() % 16
    test.check(INT8DequantPerChannelFwdOp(torch.bfloat16), q, scale, atol=0, rtol=0)


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
    # FIXME(staged-rollout): quantizes with the reference while the quantize op has no kernel.
    #
    # Broken invariant: the round trip runs through INT8QuantPerChannelFwdOp.
    # Why: its kernel lands in a separate change.
    # Cleanup: INT8QuantPerChannelFwdOp.kernel_types is non-empty.
    if INT8QuantPerChannelFwdOp.kernel_types:
        q, scale = INT8QuantPerChannelFwdOp()(x)
    else:
        q, scale = int8_quant_per_channel(x)
    x_hat = INT8DequantPerChannelFwdOp(dtype)(q, scale)
    bound = scale[:, None] * (0.5 + 2**-20) + torch.finfo(dtype).eps * x.float().abs()
    err = (x_hat.float() - x.float()).abs()
    assert (err <= bound).all(), f"max excess {(err - bound).max().item()}"
