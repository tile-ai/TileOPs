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
    assert_output_spec,
    compiled_reference,
    vllm_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.sampling import TopPMaskFwdOp
from workloads.sampling import TopPMaskWorkload, probability_above

# Rows vLLM's Triton path takes: it asserts float32 logits, and it is the path
# ``apply_top_k_top_p`` chooses at this many rows or more. Below it the sort path runs,
# which takes the logits' own dtype.
_VLLM_TRITON_ROWS = 8
# Disagreement with the reference is allowed only this close to the cut, as probability
# mass: vLLM's kernel stops its threshold search after a bounded number of rounds, so a
# token whose above-mass sits within one round of ``p`` may fall either way.
_MARGIN = 2e-2
_INF = float("inf")


@pytest.mark.parametrize("call", manifest_calls(TopPMaskFwdOp))
def test_top_p_mask_bench(call) -> None:
    workload = TopPMaskWorkload(call)
    logits, p = workload.gen_inputs()
    spec = call.specs["masked_logits"]

    op = TopPMaskFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)

    ref = workload.ref_program(logits, p)

    functors = {
        "tileops": op,
        "torch-ref": workload.ref_program,
        TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
    }

    # FlashInfer 0.6.16 exposes no top-p mask over logits. Its ``top_p_renorm_probs``
    # truncates and renormalizes a probability distribution instead: it reads a softmax the
    # op does not take, writes a zero where the op writes -inf, and rescales what it keeps.
    # That is another operation, so no row carries a FlashInfer tag, as in
    # bench_top_k_top_p_mask.py.

    # A row vLLM cannot take in the manifest's dtype carries no vLLM tag: converting the
    # workload to float32 for it would charge the row half its bytes and return an output
    # of another dtype than the entry declares.
    if logits.shape[0] < _VLLM_TRITON_ROWS or logits.dtype is torch.float32:
        # vllm masks its argument in place and its threshold search runs longer on an
        # unmasked row: over a 256x128256 float32 row it takes 1863 us on a fresh row
        # against 1131 on one it has already masked. Each call therefore refills a private
        # buffer. ``count_copies`` stays false at the comparison below, so that refill is
        # excluded from ``device_busy_ms`` rather than charged to this tag.
        apply_top_k_top_p = vllm_op("apply_top_k_top_p", "v1.sample.ops.topk_topp_sampler")
        vllm_logits = torch.empty_like(logits)

        def vllm_mask(logits: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
            return apply_top_k_top_p(vllm_logits.copy_(logits), None, p)

        # Hand-written rather than ``assert_matches_reference``: the helper compares every
        # entry, and the two legitimately disagree inside ``_MARGIN`` of the cut. The mask is
        # checked away from that band and the surviving logits entry for entry, so a
        # comparator masking another set, or returning other values under the same mask, fails.
        near = (probability_above(logits.float().softmax(-1)) - p[:, None]).abs() <= _MARGIN
        got = vllm_mask(logits, p)
        kept, taken = ref != -_INF, got != -_INF
        assert not ((taken ^ kept) & ~near).any()
        assert torch.equal(got[taken & kept], ref[taken & kept])
        functors[VLLM_TAG] = vllm_mask

    for tag, functor in functors.items():
        assert_output_spec(functor(logits, p), spec, tag)

    bm.compare(functors, logits, p)
