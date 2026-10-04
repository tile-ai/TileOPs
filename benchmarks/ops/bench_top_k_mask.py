"""Benchmark for the per-row top-k logit mask op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

import pytest
import torch

from benchmarks.baselines import (
    FLASHINFER_TAG,
    TORCH_COMPILE_TAG,
    VLLM_TAG,
    assert_output_spec,
    compiled_reference,
    flashinfer_op,
    vllm_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.sampling import TopKMaskFwdOp
from workloads.sampling import TopKMaskWorkload


@pytest.mark.parametrize("call", manifest_calls(TopKMaskFwdOp))
def test_top_k_mask_bench(call) -> None:
    workload = TopKMaskWorkload(call)
    logits, k = workload.gen_inputs()
    spec = call.specs["masked_logits"]
    op = TopKMaskFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)
    functors = {
        "tileops": op,
        "torch-ref": workload.ref_program,
        TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
    }
    flashinfer_mask = flashinfer_op("sampling.top_k_mask_logits")
    functors[FLASHINFER_TAG] = flashinfer_mask
    apply_top_k_only = vllm_op("apply_top_k_only", "v1.sample.ops.topk_topp_sampler")
    vllm_logits = torch.empty_like(logits)
    vllm_k = k.clamp(max=call.ix["V"])

    def vllm_mask(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return apply_top_k_only(vllm_logits.copy_(logits), vllm_k)

    functors[VLLM_TAG] = vllm_mask
    for tag, functor in functors.items():
        assert_output_spec(functor(logits, k), spec, tag)
    bm.compare(functors, logits, k)
