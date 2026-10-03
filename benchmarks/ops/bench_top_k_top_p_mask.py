"""Benchmark for the per-row top-k then top-p logit mask op.

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
from benchmarks.verification import Custom, logit_mask_validator
from tileops.sampling import TopKTopPMaskFwdOp
from workloads.sampling import TopKTopPMaskWorkload, probability_above, top_k_mask

# Rows vLLM's Triton path takes: it reads float32 logits only, and it is the path
# ``apply_top_k_top_p`` chooses at this many rows or more. Below it the sort path runs,
# which takes the logits' own dtype.
_VLLM_TRITON_ROWS = 8
# vLLM cuts at a sorted position instead of keeping or dropping the run of tokens tied at
# the boundary together, so the two disagree inside that run, and a token they disagree on
# sits this close to the row's smallest kept probability, relative to it. A 16-bit row ties
# thousands of tokens there, which is what makes the run wide; the measured worst case over
# the rows vLLM serves is 3.2e-2.
_MARGIN = 5e-2


@pytest.mark.parametrize("call", manifest_calls(TopKTopPMaskFwdOp))
def test_top_k_top_p_mask_bench(call) -> None:
    workload = TopKTopPMaskWorkload(call)
    logits, k, p = workload.gen_inputs()
    reference = workload.ref_program(logits, k, p)

    op = TopKTopPMaskFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)

    functors = {
        "tileops": op,
        "torch-ref": workload.ref_program,
        TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
    }

    # Same FP32 cumulative-probability boundary tolerance as the op tests.
    near = (probability_above(top_k_mask(logits, k).float().softmax(-1)) - p[:, None]).abs() <= 1e-4
    evidence = dict.fromkeys(
        functors,
        Custom(logit_mask_validator(logits, near), "nucleus boundary within 1e-4 probability mass"),
    )

    # FlashInfer 0.6.16 exposes no mask-only top-k-top-p entry point, only
    # ``top_k_top_p_sampling_from_logits``, which draws a token, so no row carries a
    # FlashInfer tag. A row vLLM cannot take in the manifest's dtype carries no vLLM tag.
    if logits.shape[0] < _VLLM_TRITON_ROWS or logits.dtype is torch.float32:
        # vLLM masks its argument in place; it reaches nothing else of its caller's, and a k
        # above V has no meaning to it, so it takes a clamped k and a private logits buffer
        # refilled per call. It is not idempotent either: a second pass renormalizes over the
        # tokens the first left and cuts further. ``count_copies`` stays false at the
        # comparison below, so the refill is excluded from ``device_busy_ms``.
        apply_top_k_top_p = vllm_op("apply_top_k_top_p", "v1.sample.ops.topk_topp_sampler")
        vllm_k = k.clamp(max=call.ix["V"])
        vllm_logits = torch.empty_like(logits)

        def vllm_mask(logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            return apply_top_k_top_p(vllm_logits.copy_(logits), vllm_k, p)

        kept = reference != -float("inf")
        probs = top_k_mask(logits, k).float().softmax(-1)
        lowest = probs.masked_fill(~kept, float("inf")).amin(-1, keepdim=True)
        near = (probs / lowest - 1).abs() <= _MARGIN
        evidence[VLLM_TAG] = Custom(
            logit_mask_validator(logits, near), "mask ties within nucleus rounding boundary"
        )
        functors[VLLM_TAG] = vllm_mask

    bm.compare(functors, logits, k, p, evidence=evidence)
