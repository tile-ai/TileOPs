"""Benchmark for the top-p (nucleus) logit mask op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

import pytest
import torch

from benchmarks.baselines import (
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    compiled_reference,
    vllm_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.sampling import TopPMaskFwdOp
from workloads.sampling import TopPMaskWorkload


@pytest.mark.parametrize("call", manifest_calls(TopPMaskFwdOp))
def test_top_p_mask_bench(call) -> None:
    workload = TopPMaskWorkload(call)
    logits, p = workload.gen_inputs()
    op = TopPMaskFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)
    functors = {
        "tileops": op,
        "torch-ref": workload.ref_program,
        TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
    }
    apply_top_k_top_p = vllm_op("apply_top_k_top_p", "v1.sample.ops.topk_topp_sampler")
    vllm_logits = torch.empty_like(logits, dtype=torch.float32)

    def vllm_mask(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return apply_top_k_top_p(vllm_logits.copy_(logits), None, p).to(logits.dtype)

    functors[VLLM_TAG] = vllm_mask
    bm.compare(
        functors,
        logits,
        p,
        noncomparable={
            VLLM_TAG: "vendor approximates the mask boundary beyond the workload contract"
        },
    )
