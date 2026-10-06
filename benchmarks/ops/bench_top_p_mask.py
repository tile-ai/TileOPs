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
    vllm_op,
)
from tileops.sampling import TopPMaskFwdOp


@pytest.mark.parametrize("case", bench.cases(TopPMaskFwdOp), ids=lambda case: case.id)
def test_top_p_mask_bench(case) -> None:
    logits, p = case.inputs
    op = TopPMaskFwdOp(**case.arguments)
    functors = {
        "torch-ref": case.reference,
        TORCH_COMPILE_TAG: compiled_reference(case.reference),
    }
    apply_top_k_top_p = vllm_op("apply_top_k_top_p", "v1.sample.ops.topk_topp_sampler")
    # vllm masks float32 logits in place.
    vllm_logits = torch.empty_like(logits, dtype=torch.float32)

    def vllm_reset() -> None:
        vllm_logits.copy_(logits)

    def vllm_mask(vllm_logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return apply_top_k_top_p(vllm_logits, None, p).to(logits.dtype)

    functors[VLLM_TAG] = bench.Implementation(
        run=vllm_mask,
        args=(vllm_logits, p),
        reset=vllm_reset,
        noncomparable_reason="vendor approximates the mask boundary beyond the workload contract",
    )
    bench.Runner(op, case).compare(functors)
