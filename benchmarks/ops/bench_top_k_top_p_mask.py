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


@pytest.mark.parametrize("call", manifest_calls(TopKTopPMaskFwdOp))
def test_top_k_top_p_mask_bench(call) -> None:
    # vLLM cuts at a sorted position instead of keeping or dropping the run of tokens tied at
    # the boundary together, so the two disagree inside that run, and a token they disagree on
    # sits this close to the row's smallest kept probability, relative to it. A 16-bit row ties
    # thousands of tokens there, which is what makes the run wide; the measured worst case over
    # the rows vLLM serves is 3.2e-2.
    margin = 5e-2
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

    # Conversion kernels are timed; same-dtype copies only reset the private input.
    apply_top_k_top_p = vllm_op("apply_top_k_top_p", "v1.sample.ops.topk_topp_sampler")
    vllm_logits = torch.empty_like(logits, dtype=torch.float32)

    def vllm_mask(logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return apply_top_k_top_p(vllm_logits.copy_(logits), k.clamp(max=logits.shape[-1]), p).to(
            logits.dtype
        )

    kept = reference != -float("inf")
    probs = top_k_mask(logits, k).float().softmax(-1)
    lowest = probs.masked_fill(~kept, float("inf")).amin(-1, keepdim=True)
    near = (probs / lowest - 1).abs() <= margin
    evidence[VLLM_TAG] = Custom(
        logit_mask_validator(logits, near), "mask ties within nucleus rounding boundary"
    )
    functors[VLLM_TAG] = vllm_mask
    bm.compare(functors, logits, k, p, evidence=evidence)
