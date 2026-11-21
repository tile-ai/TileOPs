"""Benchmark for the per-row top-k logit mask op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()``.
"""

import pytest

from benchmarks import api as bench
from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    compiled_reference,
    flashinfer_op,
    private_inputs,
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
    implementations[FLASHINFER_TAG] = flashinfer_op("sampling.top_k_mask_logits")
    apply_top_k_only = vllm_op("apply_top_k_only", "v1.sample.ops.topk_topp_sampler")
    # vllm masks the logits in place.
    implementations[VLLM_TAG] = private_inputs(
        apply_top_k_only, (logits, k.clamp(max=logits.shape[-1])), 0
    )
    bench.Runner(op, case).compare(implementations)
