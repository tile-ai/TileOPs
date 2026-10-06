"""Benchmark for the block-scaled FP8 quantization op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()``.
"""

import functools

import pytest

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, VLLM_TAG, compiled_reference, vllm_op
from tileops.quantization import FP8QuantPerBlockFwdOp

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


@pytest.mark.parametrize("case", bench.cases(FP8QuantPerBlockFwdOp), ids=lambda case: case.id)
def test_fp8_quant_per_block_bench(case) -> None:
    op = FP8QuantPerBlockFwdOp(**case.arguments, tune=_TUNE)

    # vllm's per_block_cast_to_fp8 with 128x128 tiles, a torch.compile'd expression.
    vllm_quant = functools.partial(
        vllm_op("per_block_cast_to_fp8", "utils.deep_gemm"), block_size=[128, 128]
    )
    # vllm floors the amax at 1e-4, which no tile of a random input reaches, and multiplies
    # by the reciprocal of the scale, so a code can sit one step from the reference's.
    bench.Runner(op, case).compare(
        {
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: bench.Implementation(
                run=compiled_reference(case.reference),
                noncomparable_reason="Inductor replaces division with reciprocal multiplication; block codes are not bitwise equal",
            ),
            VLLM_TAG: bench.Implementation(
                run=vllm_quant,
                noncomparable_reason="vendor clamps tiny scales and uses reciprocal multiplication instead of exact block quantization",
            ),
        }
    )
