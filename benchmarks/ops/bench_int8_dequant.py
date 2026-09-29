"""Benchmark for the INT8 dequantize ops.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`. No library baseline: vLLM, FlashInfer and FlagGems in
the runner image have no standalone INT8 dequantize kernel.
"""

import pytest

from benchmarks.baselines import TORCH_COMPILE_TAG, compiled_reference
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.quantization import INT8DequantPerChannelFwdOp, INT8DequantPerTensorFwdOp
from workloads.int8_dequant import INT8DequantPerChannelWorkload, INT8DequantPerTensorWorkload


@pytest.mark.parametrize("call", manifest_calls(INT8DequantPerChannelFwdOp))
def test_int8_dequant_per_channel_bench(call) -> None:
    workload = INT8DequantPerChannelWorkload.from_call(call)
    inputs = workload.gen_inputs()

    op = INT8DequantPerChannelFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
    )


@pytest.mark.parametrize("call", manifest_calls(INT8DequantPerTensorFwdOp))
def test_int8_dequant_per_tensor_bench(call) -> None:
    workload = INT8DequantPerTensorWorkload.from_call(call)
    inputs = workload.gen_inputs()

    op = INT8DequantPerTensorFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)

    bm.compare(
        {
            "tileops": op,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        },
        *inputs,
    )
