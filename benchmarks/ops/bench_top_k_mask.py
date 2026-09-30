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
    reference = workload.ref_program(logits, k)

    op = TopKMaskFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)

    functors = {
        "tileops": op,
        "torch-ref": workload.ref_program,
        TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
    }

    # flashinfer's top_k_mask_logits keeps ties at the threshold, as the reference does.
    flashinfer_mask = flashinfer_op("sampling.top_k_mask_logits")
    assert torch.equal(flashinfer_mask(logits, k), reference)
    functors[FLASHINFER_TAG] = flashinfer_mask

    # vllm's apply_top_k_only masks its logits in place and subtracts one from its k in
    # place, and a k above V has no meaning to it. It therefore gets a private row buffer,
    # made once so that the timed call does not carry a copy the other tags do not: masking
    # an already-masked row again selects the same values and so leaves it unchanged. Only
    # the per-row k, a few bytes, is refreshed per call.
    apply_top_k_only = vllm_op("apply_top_k_only", "v1.sample.ops.topk_topp_sampler")
    vllm_logits = logits.clone()
    vllm_k = k.clamp(max=call.ix["V"])

    def vllm_mask(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return apply_top_k_only(vllm_logits, vllm_k.clone())

    assert torch.equal(vllm_mask(logits, k), reference)
    functors[VLLM_TAG] = vllm_mask

    bm.compare(functors, logits, k)
