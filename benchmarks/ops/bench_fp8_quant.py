"""Benchmark for the FP8 quantization op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()``.
"""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, VLLM_TAG, compiled_reference, vllm_op
from tileops.ops import FP8QuantFwdOp


@pytest.mark.parametrize("case", bench.cases(FP8QuantFwdOp), ids=lambda case: case.id)
def test_fp8_quant_bench(case) -> None:
    op = FP8QuantFwdOp(**case.arguments)
    op.autotune()

    quantize = vllm_op(
        "per_token_group_quant_fp8", "model_executor.layers.quantization.utils.fp8_utils"
    )

    def vllm_fn(x):
        values, scales = quantize(
            x.reshape(-1, x.shape[-1]), x.shape[-1], eps=1e-4, use_ue8m0=False
        )
        return scales.reshape(x.shape[:-1]), values.reshape_as(x)

    bench.Runner(op, case).compare(
        {
            VLLM_TAG: vllm_fn,
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
        }
    )
