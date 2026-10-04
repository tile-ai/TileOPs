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
from workloads.numerics import Custom, logit_mask_validator
from workloads.sampling import TopPMaskWorkload, probability_above


@pytest.mark.parametrize("call", manifest_calls(TopPMaskFwdOp))
def test_top_p_mask_bench(call) -> None:
    # Disagreement with the reference is allowed only this close to the cut, as probability
    # mass: vLLM's kernel stops its threshold search after a bounded number of rounds, so a
    # token whose above-mass sits within one round of ``p`` may fall either way.
    margin = 2e-2
    workload = TopPMaskWorkload(call)
    logits, p = workload.gen_inputs()

    op = TopPMaskFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)

    functors = {
        "tileops": op,
        "torch-ref": workload.ref_program,
        TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
    }

    # Same FP32 cumulative-probability boundary tolerance as the op tests.
    near = (probability_above(logits.float().softmax(-1)) - p[:, None]).abs() <= 1e-4
    evidence = dict.fromkeys(
        functors,
        Custom(logit_mask_validator(logits, near), "nucleus boundary within 1e-4 probability mass"),
    )

    # Conversion kernels are timed; same-dtype copies only reset the private input.
    apply_top_k_top_p = vllm_op("apply_top_k_top_p", "v1.sample.ops.topk_topp_sampler")
    vllm_logits = torch.empty_like(logits, dtype=torch.float32)

    def vllm_mask(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return apply_top_k_top_p(vllm_logits.copy_(logits), None, p).to(logits.dtype)

    near = (probability_above(logits.float().softmax(-1)) - p[:, None]).abs() <= margin
    evidence[VLLM_TAG] = Custom(
        logit_mask_validator(logits, near), "mask ties within nucleus rounding boundary"
    )
    functors[VLLM_TAG] = vllm_mask
    bm.compare(functors, logits, p, evidence=evidence)
