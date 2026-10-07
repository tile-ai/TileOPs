"""Benchmark for the per-row top-k logit mask op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()``.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    compiled_reference,
    flashinfer_op,
    vllm_op,
)
from tileops.sampling import TopKMaskFwdOp


@pytest.mark.parametrize("case", bench.cases(TopKMaskFwdOp), ids=lambda case: case.id)
def test_top_k_mask_bench(case) -> None:
    logits, k = case.inputs
    op = TopKMaskFwdOp(**case.arguments)
    implementations = {
        "torch-ref": case.reference,
        TORCH_COMPILE_TAG: compiled_reference(case.reference),
    }
    flashinfer_mask = flashinfer_op("sampling.top_k_mask_logits")
    implementations[FLASHINFER_TAG] = flashinfer_mask
    apply_top_k_only = vllm_op("apply_top_k_only", "v1.sample.ops.topk_topp_sampler")
    # vllm masks the logits in place.
    vllm_logits = torch.empty_like(logits)
    vllm_k = k.clamp(max=logits.shape[-1])

    def vllm_reset() -> None:
        vllm_logits.copy_(logits)

    implementations[VLLM_TAG] = bench.Implementation(
        run=apply_top_k_only, args=(vllm_logits, vllm_k), reset=vllm_reset
    )
    bench.Runner(op, case).compare(implementations)
