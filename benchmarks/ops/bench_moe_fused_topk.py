"""Benchmark for FusedTopKOp, against vLLM's routing kernels.

vLLM is required, not optional: this file exists to compare against its routers, and a
torch reference is not a comparison worth recording. `fused_topk` takes no correction
bias, so a biased row goes to `fused_topk_bias` instead.

Real model configurations:
  Model              E    K  scoring   renorm
  Kimi K2          384   8  sigmoid   True
  Qwen3-235B-A22B  128   8  softmax   False
"""

import pytest
import torch
from vllm.model_executor.layers.fused_moe import fused_topk as _vllm_fused_topk
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias as _vllm_fused_topk_bias,
)

from benchmarks.benchmark_base import ManifestBenchmark, fields, workload_params
from tileops.manifest import load_workloads
from tileops.ops.moe import FusedTopKOp
from workloads.moe import FusedTopKWorkload


@pytest.mark.parametrize(
    "num_tokens, num_experts, top_k, scoring_func, renormalize, with_correction_bias, dtype",
    workload_params(
        load_workloads(FusedTopKOp),
        fields(
            "num_tokens",
            "num_experts",
            "top_k",
            "scoring_func",
            "renormalize",
            "with_correction_bias",
            dtype_last=True,
        ),
    ),
)
def test_fused_topk_bench(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    scoring_func: str,
    renormalize: bool,
    with_correction_bias: bool,
    dtype: torch.dtype,
) -> None:
    test = FusedTopKWorkload(
        num_tokens,
        num_experts,
        top_k,
        scoring_func,
        renormalize,
        dtype,
        with_correction_bias=with_correction_bias,
    )
    inputs = test.gen_inputs()
    gating_output = inputs[0]

    op = FusedTopKOp(
        top_k=top_k,
        scoring_func=scoring_func,
        renormalize=renormalize,
    )
    bm = ManifestBenchmark(op, test)
    op(*inputs)  # warmup / JIT compile
    torch.cuda.synchronize()

    functors = {"tileops": op}

    # Cast bf16->f32 inside the timed call to match TileOPs' input conditions.
    hidden_dummy = torch.empty(num_tokens, 1, device=gating_output.device)
    if with_correction_bias:

        def _vllm_fn(gating_output, correction_bias):
            return _vllm_fused_topk_bias(
                hidden_states=hidden_dummy,
                gating_output=gating_output.float(),
                scoring_func=scoring_func,
                e_score_correction_bias=correction_bias,
                topk=top_k,
                renormalize=renormalize,
            )

    else:

        def _vllm_fn(gating_output):
            return _vllm_fused_topk(
                hidden_states=hidden_dummy,
                gating_output=gating_output.float(),
                topk=top_k,
                renormalize=renormalize,
                scoring_func=scoring_func,
            )

    _vllm_fn(*inputs)  # warmup
    torch.cuda.synchronize()
    functors["vllm"] = _vllm_fn

    bm.compare(functors, *inputs)
