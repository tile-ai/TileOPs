"""Benchmark for the INT8 dequantize ops.

No library baseline: vLLM, FlashInfer and FlagGems in the runner image have no standalone INT8
dequantize kernel.
"""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from tileops.quantization import (
    INT8DequantPerBlockFwdOp,
    INT8DequantPerChannelFwdOp,
    INT8DequantPerTensorFwdOp,
)


def _bench(op_cls, case) -> None:
    op = op_cls(**case.arguments)
    bench.Runner(op, case).compare(
        {
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
        }
    )


@pytest.mark.parametrize("case", bench.cases(INT8DequantPerChannelFwdOp))
def test_int8_dequant_per_channel_bench(case) -> None:
    _bench(INT8DequantPerChannelFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(INT8DequantPerTensorFwdOp))
def test_int8_dequant_per_tensor_bench(case) -> None:
    _bench(INT8DequantPerTensorFwdOp, case)


@pytest.mark.parametrize("case", bench.cases(INT8DequantPerBlockFwdOp))
def test_int8_dequant_per_block_bench(case) -> None:
    _bench(INT8DequantPerBlockFwdOp, case)
