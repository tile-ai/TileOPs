"""Benchmark for the min-p logit mask op.

Workload shapes and dtypes come from the ops manifest; roofline FLOP and
byte counts come from the op's ``eval_roofline()`` via
:class:`ManifestBenchmark`.
"""

import pytest

from benchmarks.baselines import TORCH_COMPILE_TAG, VLLM_TAG, compiled_reference, vllm_op
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

    op = MinPMaskFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)

    # vllm's processor is a stateful sampler component: it reads min_p off itself and
    # masks the logits in place. Masking is idempotent, since it moves no logit that
    # survives it and -inf never returns, so its private copy is made once here rather
    # than inside the timed call.
    processor = vllm_op("MinPLogitsProcessor", "v1.sample.logits_processor.builtin")
    vllm_min_p = processor.__new__(processor)
    vllm_min_p.min_p_count = 1
    vllm_min_p.min_p = min_p[:, None]

    ref = workload.ref_program(logits, min_p)
    relative = logits.float().softmax(-1)
    relative = relative / relative.amax(-1, keepdim=True) - min_p[:, None]
    near = relative.abs() <= _MARGIN
    got = vllm_min_p.apply(logits.clone())
    assert not (((got == -float("inf")) ^ (ref == -float("inf"))) & ~near).any()

    functors = {
        "tileops": op,
        "torch-ref": workload.ref_program,
        TORCH_COMPILE_TAG: compiled_reference(workload.ref_program),
        VLLM_TAG: (vllm_min_p.apply, (logits.clone(),)),
    }
    bm.compare(functors, logits, min_p)
