from functools import partial

import pytest
import torch

from tests.test_base import TestBase, allclose_compare, exact_compare
from tileops.backend import OpNotAvailableError
from tileops.quantization import (
    FP8QuantPerBlockFwdOp,
    INT4QuantPerGroupFwdOp,
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
