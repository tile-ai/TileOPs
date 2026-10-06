"""Benchmark for the min-p logit mask op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()``.
"""

import dataclasses

import pytest

from benchmarks import api as bench
from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    compiled_reference,
    private_inputs,
    vllm_op,
)
from tileops.sampling import MinPMaskFwdOp


@pytest.mark.parametrize("case", bench.cases(MinPMaskFwdOp), ids=lambda case: case.id)
def test_min_p_mask_bench(case) -> None:
    logits, min_p = case.inputs
    op = MinPMaskFwdOp(**case.arguments)
    processor = vllm_op("MinPLogitsProcessor", "v1.sample.logits_processor.builtin")
    vllm_min_p = processor.__new__(processor)
    vllm_min_p.min_p_count = 1
    vllm_min_p.min_p = min_p[:, None]

    # vllm masks the logits in place.
    vllm_mask = private_inputs(lambda logits, min_p: vllm_min_p.apply(logits), case.inputs, 0)
    bench.Runner(op, case).compare(
        {
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
            VLLM_TAG: dataclasses.replace(
                vllm_mask,
                noncomparable_reason=(
                    "vendor approximates the mask boundary beyond the workload contract"
                ),
            ),
        }
    )
