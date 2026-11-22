"""Benchmark for the min-p logit mask op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()``.
"""

import pytest
import torch

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
    apply_min_p = vllm_op("apply_min_p", "v1.worker.gpu.sample.min_p")
    rows = torch.arange(logits.shape[0], dtype=torch.int32, device=logits.device)

    def vllm_fn(logits, min_p):
        apply_min_p(logits, rows, min_p)
        return logits

    bench.Runner(op, case).compare(
        {
            "torch-ref": case.reference,
            TORCH_COMPILE_TAG: compiled_reference(case.reference),
            # vllm masks the logits in place.
            VLLM_TAG: private_inputs(
                vllm_fn,
                case.inputs,
                0,
            ),
        }
    )
