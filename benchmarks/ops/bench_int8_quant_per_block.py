"""Benchmark for the per-block INT8 quantization op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()``.
"""

import functools

import pytest

from benchmarks import api as bench
from benchmarks.baselines import TORCH_COMPILE_TAG, VLLM_TAG, compiled_reference, vllm_op
from tileops.quantization import INT8QuantPerBlockFwdOp

# Autotuning is a bench-run policy, not a workload property; manifest
# workloads do not carry it.
_TUNE = True


@pytest.mark.parametrize("case", bench.cases(INT8QuantPerBlockFwdOp), ids=lambda case: case.id)
def test_int8_quant_per_block_bench(case) -> None:
    workload = case.workload

    op = INT8QuantPerBlockFwdOp(**case.arguments)
    if _TUNE:
        op.request_tune()
    implementations = {
        "torch-ref": case.reference,
        TORCH_COMPILE_TAG: bench.Implementation(
            run=compiled_reference(case.reference),
            noncomparable_reason="Inductor lowering does not preserve the reference's exact INT8 codes",
        ),
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
        implementations[VLLM_TAG] = bench.Implementation(
            run=vllm_quant,
            noncomparable_reason=(
                "vendor truncates codes and clamps tiny scales; workload requires round-to-nearest"
            ),
        )
    bench.Runner(op, case).compare(implementations)
