"""Benchmark for the top-p (nucleus) logit mask op.

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
from tileops.sampling import TopPMaskFwdOp
from workloads.sampling import TopPMaskWorkload, probability_above

# Disagreement with the reference is allowed only this close to the cut, as probability
# mass: vLLM's kernel stops its threshold search after a bounded number of rounds, so a
# token whose above-mass sits within one round of ``p`` may fall either way.
_MARGIN = 2e-2
_INF = float("inf")


@pytest.mark.parametrize("call", manifest_calls(TopPMaskFwdOp))
def test_top_p_mask_bench(call) -> None:
    workload = TopPMaskWorkload(call)
    logits, p = workload.gen_inputs()

    op = TopPMaskFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)

    ref = workload.ref_program(logits, p)
    near = (probability_above(logits.float().softmax(-1)) - p[:, None]).abs() <= _MARGIN

    # vllm's kernel takes float32 logits and masks them in place, and its threshold search
    # runs longer on an unmasked row: over a 256x128256 float32 row it takes 1863 us on a
    # fresh row against 1131 on one it has already masked. Each iteration therefore refills
    # its buffer, which costs the 62 us that refill measures on that row.
    apply_top_k_top_p = vllm_op("apply_top_k_top_p", "v1.sample.ops.topk_topp_sampler")
    vllm_logits = logits.float()
    vllm_buffer = torch.empty_like(vllm_logits)
    got = apply_top_k_top_p(vllm_logits.clone(), None, p.clone())
    assert not (((got == -_INF) ^ (ref == -_INF)) & ~near).any()

    # flashinfer's entry point renormalizes probabilities rather than masking logits, so
    # its row starts from the softmax the other rows compute, and a masked token comes
    # back as a zero probability.
    top_p_renorm_probs = flashinfer_op("sampling.top_p_renorm_probs")
    probs = logits.float().softmax(-1)
    assert not (((top_p_renorm_probs(probs, p) == 0) ^ (ref == -_INF)) & ~near).any()

    functors = {
        "tileops": op,
        "torch-ref": workload.ref_program,
        TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        VLLM_TAG: (lambda: apply_top_k_top_p(vllm_buffer.copy_(vllm_logits), None, p), ()),
        FLASHINFER_TAG: (top_p_renorm_probs, (probs, p)),
    }
    bm.compare(functors, logits, p)
