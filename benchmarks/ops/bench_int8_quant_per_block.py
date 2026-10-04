"""Benchmark for the per-block INT8 quantization op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

import functools

import pytest

from benchmarks.baselines import TORCH_COMPILE_TAG, VLLM_TAG, compiled_reference, vllm_op
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.quantization import INT8QuantPerBlockFwdOp
from workloads.numerics import Custom, assert_quantized
from workloads.quantization.quantize import INT8QuantPerBlockWorkload

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


@pytest.mark.parametrize("call", manifest_calls(INT8QuantPerBlockFwdOp))
def test_int8_quant_per_block_bench(call) -> None:
    workload = INT8QuantPerBlockWorkload.from_call(call)
    inputs = workload.gen_inputs()

    op = INT8QuantPerBlockFwdOp(**call.arguments({}), tune=_TUNE)
    bm = ManifestBenchmark(op, workload)

    functors = {
        "tileops": op,
        "torch-ref": workload.ref_program,
        TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
    }
    # vllm requires K to be a whole number of groups, so a ragged K has no vllm row.
    if workload.cols % 128 == 0:
        # vllm's per_token_group_quant_int8 with 128-element groups; on CUDA it runs vllm's
        # CUDA kernel.
        vllm_quant = functools.partial(
            vllm_op(
                "per_token_group_quant_int8", "model_executor.layers.quantization.utils.int8_utils"
            ),
            group_size=128,
        )
        # vllm divides by ``max(amax, 1e-10) / 127`` and truncates the quotient, so a code
        # can sit one below the reference's in magnitude; the scales agree to float32 rounding.
        functors[VLLM_TAG] = vllm_quant
    bm.compare(
        functors,
        *inputs,
        evidence=dict.fromkeys(
            functors, Custom(assert_quantized, "scales checked; INT8 rounding within one code")
        ),
    )
