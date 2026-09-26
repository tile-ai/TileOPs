"""Benchmark for FusedTopKFwdOp, against vLLM's routing kernels.

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

from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.moe import FusedTopKFwdOp
from workloads.moe import FusedTopKWorkload


@pytest.mark.parametrize("call", manifest_calls(FusedTopKFwdOp))
def test_fused_topk_bench(call) -> None:
    test = FusedTopKWorkload(call)
    inputs = test.gen_inputs()
    gating_output, correction_bias = inputs
    if correction_bias is None:
        inputs = (gating_output,)
    op = FusedTopKFwdOp(**call.arguments({}))
    top_k, scoring_func, renormalize = op.top_k, op.scoring_func, op.renormalize
    num_tokens = gating_output.shape[0]
    bm = ManifestBenchmark(op, test)

    weights, _ = op(*inputs)
    ref_weights, _ = test.ref_program(*inputs)
    # Ties may pick different experts; the kept weights agree once sorted.
    torch.testing.assert_close(
        weights.sort(dim=-1).values, ref_weights.sort(dim=-1).values, rtol=1e-3, atol=1e-3
    )

    functors = {"tileops": op}

    # Cast bf16->f32 inside the timed call to match TileOPs' input conditions.
    hidden_dummy = torch.empty(num_tokens, 1, device=gating_output.device)
    if correction_bias is not None:

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
