"""Benchmark for the block-scaled FP8 quantization op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

import functools

import pytest
import torch

from benchmarks.baselines import TORCH_COMPILE_TAG, VLLM_TAG, compiled_reference, vllm_op
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.quantization import FP8QuantPerBlockFwdOp
from workloads.quantization.quantize import FP8QuantPerBlockWorkload

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


@pytest.mark.parametrize("call", manifest_calls(FP8QuantPerBlockFwdOp))
def test_fp8_quant_per_block_bench(call) -> None:
    workload = FP8QuantPerBlockWorkload.from_call(call)
    inputs = workload.gen_inputs()

    op = FP8QuantPerBlockFwdOp(**call.arguments({}), tune=_TUNE)
    bm = ManifestBenchmark(op, workload)

    # vllm's per_block_cast_to_fp8 with 128x128 tiles, a torch.compile'd expression.
    vllm_quant = functools.partial(
        vllm_op("per_block_cast_to_fp8", "utils.deep_gemm"), block_size=[128, 128]
    )
    # vllm floors the amax at 1e-4, which no tile of a random input reaches, and multiplies
    # by the reciprocal of the scale, so a code can sit one step from the reference's.
    q, scale = vllm_quant(*inputs)
    q_ref, scale_ref = workload.ref_program(*inputs)
    torch.testing.assert_close(scale, scale_ref, rtol=1e-6, atol=0.0)
    step = (q.view(torch.uint8).int() - q_ref.view(torch.uint8).int()).abs()
    assert step.max().item() <= 1
    bm.compare(
        {
            "tileops": op,
            "torch-ref": workload.ref_program,
            TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
            VLLM_TAG: vllm_quant,
        },
        *inputs,
    )
