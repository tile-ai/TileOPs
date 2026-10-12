"""Benchmark for FusedTopKFwdOp, against vLLM's routing kernels.

vLLM is required, not optional: this file exists to compare against its routers, and a
torch reference is not a comparison worth recording. `fused_topk` takes no correction
bias, so a biased row goes to `fused_topk_bias` instead. A grouped row
(`num_expert_group > 1`) goes to vLLM's fused CUDA `grouped_topk`, which applies the
sigmoid, the group selection, renormalization and the scaling factor itself and takes the
logits in their own dtype; it requires a correction bias, so a grouped row without one has
no vLLM comparator. vLLM's torch fallback for grouped routing is not a comparator: it
diverts to the same fused kernel unless `VLLM_USE_FUSED_MOE_GROUPED_TOPK=0` is set before
vLLM is imported, which one process running several benchmark files cannot guarantee.
SGLang's `biased_grouped_topk_gpu` and TensorRT-LLM's `noaux_tc_op` are not in the runner
image (`.github/runner/Dockerfile`), so they are not compared.

Real model configurations:
  Model              E    K  scoring   renorm  groups  kept  scale
  Kimi K2          384   8  sigmoid   True         1     1   1.0
  Qwen3-235B-A22B  128   8  softmax   True         1     1   1.0
  DeepSeek-V3      256   8  sigmoid   True         8     4   2.5
"""

import pytest
import torch
from vllm.model_executor.layers.fused_moe import fused_topk as _vllm_fused_topk
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias as _vllm_fused_topk_bias,
)

from benchmarks import api as bench
from benchmarks.baselines import VLLM_TAG, vllm_op
from tileops.ops.moe import FusedTopKFwdOp

# vLLM's grouped_topk scoring_func argument: 1 applies the sigmoid to the logits.
_VLLM_SIGMOID = 1


@pytest.mark.parametrize("case", bench.cases(FusedTopKFwdOp), ids=lambda case: case.id)
def test_fused_topk_bench(case) -> None:
    gating_output, *bias = case.inputs
    correction_bias = bias[0] if bias else None
    op = FusedTopKFwdOp(**case.arguments)
    top_k, scoring_func, renormalize = op.top_k, op.scoring_func, op.renormalize
    num_tokens = gating_output.shape[0]

    if op.num_expert_group > 1:
        if correction_bias is None or scoring_func != "sigmoid":
            # vLLM's fused grouped_topk takes only a biased sigmoid.
            bench.Runner(op, case).compare({})
            return
        grouped_topk = vllm_op("grouped_topk")

        def _vllm_grouped(gating_output, correction_bias):
            return grouped_topk(
                gating_output,
                op.num_expert_group,
                op.topk_group,
                top_k,
                renormalize,
                op.routed_scaling_factor,
                correction_bias,
                _VLLM_SIGMOID,
            )

        bench.Runner(op, case).compare({VLLM_TAG: _vllm_grouped})
        return

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
                routed_scaling_factor=op.routed_scaling_factor,
            )[:2]

    else:

        def _vllm_fn(gating_output):
            weights, ids, _ = _vllm_fused_topk(
                hidden_states=hidden_dummy,
                gating_output=gating_output.float(),
                topk=top_k,
                renormalize=renormalize,
                scoring_func=scoring_func,
            )
            # fused_topk has no scaling factor; vLLM applies it after routing.
            if op.routed_scaling_factor != 1.0:
                weights = weights * op.routed_scaling_factor
            return weights, ids

    bench.Runner(op, case).compare({VLLM_TAG: _vllm_fn})
