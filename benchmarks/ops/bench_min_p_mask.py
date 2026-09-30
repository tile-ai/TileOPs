"""Benchmark for the min-p logit mask op.

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
from tileops.sampling import MinPMaskFwdOp
from workloads.sampling import MinPMaskWorkload

# Disagreement with the reference is allowed only this close to the threshold, as a
# relative probability: vLLM softmaxes and compares in the logits' dtype, so a token
# whose probability sits within one rounding of ``min_p`` may fall either way.
_MARGIN = 2e-2


@pytest.mark.parametrize("call", manifest_calls(MinPMaskFwdOp))
def test_min_p_mask_bench(call) -> None:
    workload = MinPMaskWorkload(call)
    logits, min_p = workload.gen_inputs()
    spec = call.specs["masked_logits"]

    op = MinPMaskFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)

    # vllm's processor is a stateful sampler component: it reads min_p off itself and masks
    # the logits in place. It therefore gets a private buffer, refilled per call: a row it
    # has already masked is not the row the other tags read, and timing a softmax and a
    # compare over a row of -inf measures another input. ``count_copies`` stays false at the
    # comparison below, so the refill is excluded from ``device_busy_ms``.
    processor = vllm_op("MinPLogitsProcessor", "v1.sample.logits_processor.builtin")
    vllm_min_p = processor.__new__(processor)
    vllm_min_p.min_p_count = 1
    vllm_min_p.min_p = min_p[:, None]
    vllm_logits = torch.empty_like(logits)

    def vllm_mask(logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
        return vllm_min_p.apply(vllm_logits.copy_(logits))

    ref = workload.ref_program(logits, min_p)
    relative = logits.float().softmax(-1)
    relative = relative / relative.amax(-1, keepdim=True) - min_p[:, None]
    near = relative.abs() <= _MARGIN
    # Hand-written rather than ``assert_matches_reference``: the helper compares every
    # entry, and the two legitimately disagree within ``_MARGIN`` of the threshold. The
    # mask is checked away from that band and the surviving logits entry for entry, so a
    # comparator masking another set, or returning other values under the same mask, fails.
    got = vllm_mask(logits, min_p)
    kept, taken = ref != -float("inf"), got != -float("inf")
    assert not ((taken ^ kept) & ~near).any()
    assert torch.equal(got[taken & kept], ref[taken & kept])

    functors = {
        "tileops": op,
        "torch-ref": workload.ref_program,
        TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        VLLM_TAG: vllm_mask,
    }

    for tag, functor in functors.items():
        assert_output_spec(functor(logits, min_p), spec, tag)

    bm.compare(functors, logits, min_p)
