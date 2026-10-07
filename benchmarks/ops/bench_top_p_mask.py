"""Benchmark for the top-p (nucleus) logit mask op.

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
    private_float32_logits,
    vllm_op,
)
from tileops.sampling import TopPMaskFwdOp


@pytest.mark.parametrize("case", bench.cases(TopPMaskFwdOp), ids=lambda case: case.id)
def test_top_p_mask_bench(case) -> None:
    logits, p = case.inputs
    op = TopPMaskFwdOp(**case.arguments)
    implementations = {
        "torch-ref": case.reference,
        TORCH_COMPILE_TAG: compiled_reference(case.reference),
    }
    apply_top_k_top_p = vllm_op("apply_top_k_top_p", "v1.sample.ops.topk_topp_sampler")

    def vllm_mask(vllm_logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return apply_top_k_top_p(vllm_logits, None, p).to(logits.dtype)

    implementations[VLLM_TAG] = private_float32_logits(
        vllm_mask,
        logits,
        p,
        noncomparable_reason="vendor approximates the mask boundary beyond the workload contract",
    )
    bench.Runner(op, case).compare(implementations)
