"""Benchmark for FusedTopKFwdOp, against vLLM's routing kernels.

vLLM is required, not optional: this file exists to compare against its routers, and a
torch reference is not a comparison worth recording. `fused_topk` takes no correction
bias, so a biased row goes to `fused_topk_bias` instead. A grouped row
(`num_expert_group > 1`) goes to vLLM's fused CUDA `grouped_topk`, which takes only a
biased sigmoid, so a grouped row without a bias has no vLLM comparator. A biased
sqrt-softplus row goes to `fused_topk_bias`'s DeepSeek-V4 kernel, and a row with given
experts to its hash-table lookup.

Real model configurations:
  Model                E    K  scoring       renorm  groups  kept  scale
  Kimi K2            384   8  sigmoid       True         1     1   1.0
  Qwen3-235B-A22B    128   8  softmax       True         1     1   1.0
  DeepSeek-V3        256   8  sigmoid       True         8     4   2.5
  DeepSeek-V4-Flash  256   6  sqrtsoftplus  True         1     1   1.5
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import VLLM_GROUPED_TOPK_SIGMOID, VLLM_TAG, vllm_op
from tileops._ops.moe import FusedTopKFwdOp


@pytest.mark.parametrize("case", bench.cases(FusedTopKFwdOp), ids=lambda case: case.id)
def test_fused_topk_bench(case) -> None:
    gating_output, *bias = case.inputs
    correction_bias = bias[0] if bias else None
    op = FusedTopKFwdOp(**case.arguments)
    top_k, scoring_func, renormalize = op.top_k, op.scoring_func, op.renormalize
    num_tokens = gating_output.shape[0]
    # Cast bf16->f32 inside the timed call to match TileOPs' input conditions.
    hidden_dummy = torch.empty(num_tokens, 1, device=gating_output.device)
    fused_topk_bias = vllm_op(
        "fused_topk_bias", "model_executor.layers.fused_moe.router.fused_topk_bias_router"
    )
    if len(case.inputs) == 3:
        # Caller-given experts: vLLM looks token t's experts up in a table at row t.
        tokens = torch.arange(num_tokens, dtype=torch.int32, device=gating_output.device)

        def _vllm_given(gating_output, correction_bias, expert_ids):
            return fused_topk_bias(
                hidden_states=hidden_dummy,
                gating_output=gating_output.float(),
                scoring_func=scoring_func,
                e_score_correction_bias=correction_bias,
                topk=top_k,
                renormalize=renormalize,
                input_tokens=tokens,
                hash_indices_table=expert_ids,
                routed_scaling_factor=op.routed_scaling_factor,
            )[:2]

        bench.Runner(op, case).compare({VLLM_TAG: _vllm_given})
        return

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
                VLLM_GROUPED_TOPK_SIGMOID,
            )

        bench.Runner(op, case).compare({VLLM_TAG: _vllm_grouped})
        return

    if correction_bias is not None:

        def _vllm_fn(gating_output, correction_bias):
            return fused_topk_bias(
                hidden_states=hidden_dummy,
                gating_output=gating_output.float(),
                scoring_func=scoring_func,
                e_score_correction_bias=correction_bias,
                topk=top_k,
                renormalize=renormalize,
                routed_scaling_factor=op.routed_scaling_factor,
            )[:2]

    else:
        fused_topk = vllm_op("fused_topk", "model_executor.layers.fused_moe")

        def _vllm_fn(gating_output):
            weights, ids, _ = fused_topk(
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
