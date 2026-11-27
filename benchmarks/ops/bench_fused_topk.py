"""Benchmark for FusedTopKFwdOp, against vLLM's routing kernels.

vLLM is required. An unbiased row goes to `fused_topk`, a biased row or one with given
experts to `fused_topk_bias`, and a grouped row (`num_expert_group > 1`) to the fused CUDA
`grouped_topk`, which takes only a biased sigmoid, so a grouped row without a bias has no
vLLM comparator.
"""

import pytest
import torch

from benchmarks import api as bench
from benchmarks.baselines import VLLM_TAG, vllm_op
from benchmarks.moe_baselines import vllm_router
from tileops._ops.moe import FusedTopKFwdOp


@pytest.mark.parametrize("case", bench.cases(FusedTopKFwdOp))
def test_fused_topk_bench(case) -> None:
    gating_output, *bias = case.inputs
    correction_bias = bias[0] if bias else None
    op = FusedTopKFwdOp(**case.arguments)
    num_tokens = gating_output.shape[0]
    hidden_dummy = torch.empty(num_tokens, 1, device=gating_output.device)
    if len(case.inputs) == 3:
        fused_topk_bias = vllm_op(
            "fused_topk_bias", "model_executor.layers.fused_moe.router.fused_topk_bias_router"
        )
        # Caller-given experts: vLLM looks token t's experts up in a table at row t.
        tokens = torch.arange(num_tokens, dtype=torch.int32, device=gating_output.device)

        def _vllm_given(gating_output, correction_bias, expert_ids):
            return fused_topk_bias(
                hidden_states=hidden_dummy,
                gating_output=gating_output.float(),
                scoring_func=op.scoring_func,
                e_score_correction_bias=correction_bias,
                topk=op.top_k,
                renormalize=op.renormalize,
                input_tokens=tokens,
                hash_indices_table=expert_ids,
                routed_scaling_factor=op.routed_scaling_factor,
            )[:2]

        bench.Runner(op, case).compare({VLLM_TAG: _vllm_given})
        return

    grouped = op.num_expert_group > 1
    if grouped and (correction_bias is None or op.scoring_func != "sigmoid"):
        # vLLM's fused grouped_topk takes only a biased sigmoid.
        bench.Runner(op, case).compare({})
        return
    route = vllm_router(op, op.routed_scaling_factor)

    def _vllm_fn(gating_output, *correction_bias):
        # The ungrouped routers' bf16 -> f32 cast is timed, matching TileOPs' input conditions.
        gating = gating_output if grouped else gating_output.float()
        return route(hidden_dummy, gating, *correction_bias)

    bench.Runner(op, case).compare({VLLM_TAG: _vllm_fn})
