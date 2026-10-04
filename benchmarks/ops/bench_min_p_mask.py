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
from tileops.sampling import MinPMaskFwdOp
from workloads.sampling import MinPMaskWorkload


@pytest.mark.parametrize("call", manifest_calls(MinPMaskFwdOp))
def test_min_p_mask_bench(call) -> None:
    workload = MinPMaskWorkload(call)
    logits, min_p = workload.gen_inputs()
    op = MinPMaskFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, workload)
    processor = vllm_op("MinPLogitsProcessor", "v1.sample.logits_processor.builtin")
    vllm_min_p = processor.__new__(processor)
    vllm_min_p.min_p_count = 1
    vllm_min_p.min_p = min_p[:, None]
    vllm_logits = torch.empty_like(logits)

    def vllm_mask(logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
        return vllm_min_p.apply(vllm_logits.copy_(logits))

    relative = logits.float().softmax(-1)
    relative = relative / relative.amax(-1, keepdim=True) - min_p[:, None]
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
        noncomparable={
            VLLM_TAG: "vendor approximates the mask boundary beyond the workload contract"
        },
    )
