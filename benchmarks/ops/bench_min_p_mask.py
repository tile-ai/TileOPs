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
    compiled_reference,
    vllm_op,
)
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from benchmarks.verification import Custom, logit_mask_validator
from tileops.sampling import MinPMaskFwdOp
from workloads.sampling import MinPMaskWorkload


@pytest.mark.parametrize("call", manifest_calls(MinPMaskFwdOp))
def test_min_p_mask_bench(call) -> None:
    workload = MinPMaskWorkload(call)
    logits, min_p = workload.gen_inputs()

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

    relative = logits.float().softmax(-1)
    relative = relative / relative.amax(-1, keepdim=True) - min_p[:, None]
    # Disagreement with the reference is allowed only this close to the threshold, as a
    # relative probability: vLLM softmaxes and compares in the logits' dtype, so a token
    # whose probability sits within one rounding of ``min_p`` may fall either way.
    margin = 2e-2
    near = relative.abs() <= margin
    functors = {
        "tileops": op,
        "torch-ref": workload.ref_program,
        TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        VLLM_TAG: vllm_mask,
    }

    bm.compare(
        functors,
        logits,
        min_p,
        evidence={
            VLLM_TAG: Custom(
                logit_mask_validator(logits, near), "mask ties within min-p rounding boundary"
            )
        },
    )
