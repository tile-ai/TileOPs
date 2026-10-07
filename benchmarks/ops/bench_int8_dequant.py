"""Benchmark for the INT8 dequantize ops.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()``. No library baseline: vLLM,
FlashInfer and FlagGems in the runner image have no standalone INT8 dequantize
kernel.
"""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from tileops.quantization import (
    INT8DequantPerBlockFwdOp,
    INT8DequantPerChannelFwdOp,
    INT8DequantPerTensorFwdOp,
)


@pytest.mark.parametrize("case", bench.cases(INT8DequantPerChannelFwdOp), ids=lambda case: case.id)
def test_int8_dequant_per_channel_bench(case) -> None:
    op = INT8DequantPerChannelFwdOp(**case.arguments)
    bench.Runner(op, case).compare(
        {
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
        }
    )


@pytest.mark.parametrize("case", bench.cases(INT8DequantPerTensorFwdOp), ids=lambda case: case.id)
def test_int8_dequant_per_tensor_bench(case) -> None:
    op = INT8DequantPerTensorFwdOp(**case.arguments)
    bench.Runner(op, case).compare(
        {
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
        }
    )


@pytest.mark.parametrize("case", bench.cases(INT8DequantPerBlockFwdOp), ids=lambda case: case.id)
def test_int8_dequant_per_block_bench(case) -> None:
    op = INT8DequantPerBlockFwdOp(**case.arguments)
    bench.Runner(op, case).compare(
        {
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
        }
    )
