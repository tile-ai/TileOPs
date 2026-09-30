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
    assert_matches_reference,
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

    # flashinfer's top_k_mask_logits keeps ties at the threshold, as the reference does, so
    # the two agree entry for entry and the helper's exact comparison fits.
    flashinfer_mask = flashinfer_op("sampling.top_k_mask_logits")
    assert_matches_reference(flashinfer_mask, workload.ref_program, logits, k, rtol=0.0, atol=0.0)
    functors[FLASHINFER_TAG] = flashinfer_mask

    # vllm's apply_top_k_only masks its logits in place and subtracts one from its k in
    # place, and a k above V has no meaning to it. Both therefore come from private buffers
    # refilled per call: a row it has already masked is not the row the other tags read, and
    # a topk over a row of -inf is another input. ``count_copies`` stays false at the
    # comparison below, so the refills are excluded from ``device_busy_ms``.
    apply_top_k_only = vllm_op("apply_top_k_only", "v1.sample.ops.topk_topp_sampler")
    vllm_logits = torch.empty_like(logits)
    vllm_k = k.clamp(max=call.ix["V"])

    def vllm_mask(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return apply_top_k_only(vllm_logits.copy_(logits), vllm_k.clone())

    assert_matches_reference(vllm_mask, workload.ref_program, logits, k, rtol=0.0, atol=0.0)
    functors[VLLM_TAG] = vllm_mask

    for tag, functor in functors.items():
        assert_output_spec(functor(logits, k), spec, tag)

    bm.compare(functors, logits, k)
