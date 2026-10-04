"""Benchmark for the FP8 quantization op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

import pytest

from benchmarks.baselines import TORCH_COMPILE_TAG, VLLM_TAG, compiled_reference, vllm_op
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops import FP8QuantFwdOp
from workloads.numerics import Custom, assert_quantized
from workloads.quantization.fp8_quant import FP8QuantWorkload


@pytest.mark.parametrize("call", manifest_calls(FP8QuantFwdOp))
def test_fp8_quant_bench(call) -> None:
    workload = FP8QuantWorkload.from_call(call)
    inputs = workload.gen_inputs()

    op = FP8QuantFwdOp(**call.arguments({}), tune=True)
    bm = ManifestBenchmark(op, workload)

    quantize = vllm_op(
        "per_token_group_quant_fp8", "model_executor.layers.quantization.utils.fp8_utils"
    )

    def vllm_fn(x):
        values, scales = quantize(
            x.reshape(-1, x.shape[-1]), x.shape[-1], eps=1e-4, use_ue8m0=False
        )
        return scales.reshape(x.shape[:-1]), values.reshape_as(x)

    bm.compare(
        {
            "tileops": op,
            VLLM_TAG: vllm_fn,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
        evidence={
            tag: Custom(assert_quantized, "scales checked; FP8 rounding within one code")
            for tag in ("tileops", VLLM_TAG, TORCH_COMPILE_TAG)
        },
    )
