from functools import partial

import pytest
import torch

from tests.test_base import TestBase, allclose_compare, exact_compare
from tileops.backend import OpNotAvailableError
from tileops.kernels.quantization import INT8QuantPerTensorFwdKernel
from tileops.quantization import (
    FP8QuantPerBlockFwdOp,
    INT4QuantPerGroupFwdOp,
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
    FP8QuantPerBlockFwdOp: [_fp8_compare, _scale_compare],
    INT4QuantPerGroupFwdOp: [exact_compare, exact_compare, exact_compare],
}


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
        # Subnormal inputs and scale, which the quotient is scaled out of before dividing.
        pytest.param(torch.float32, lambda x: x * 1e-40, id="subnormal-scale"),
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
