"""Benchmark for FusedTopKFwdOp, against vLLM's routing kernels.

vLLM is required, not optional: this file exists to compare against its routers, and a
torch reference is not a comparison worth recording. `fused_topk` takes no correction
bias, so a biased row goes to `fused_topk_bias` instead.

Real model configurations:
  Model              E    K  scoring   renorm
  Kimi K2          384   8  sigmoid   True
  Qwen3-235B-A22B  128   8  softmax   True
"""

import pytest
import torch
from vllm.model_executor.layers.fused_moe import fused_topk as _vllm_fused_topk
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias as _vllm_fused_topk_bias,
)

from benchmarks.baselines import VLLM_TAG
from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from benchmarks.verification import Custom
from tileops.ops.moe import FusedTopKFwdOp
from workloads.moe import FusedTopKWorkload


@pytest.mark.parametrize("call", manifest_calls(FusedTopKFwdOp))
def test_fused_topk_bench(call) -> None:
    workload = FusedTopKWorkload(call)
    inputs = workload.gen_inputs()
    gating_output, correction_bias = inputs
    if correction_bias is None:
        inputs = (gating_output,)
    op = FusedTopKFwdOp(**call.arguments({}))
    top_k, scoring_func, renormalize = op.top_k, op.scoring_func, op.renormalize
    num_tokens = gating_output.shape[0]
    bm = ManifestBenchmark(op, workload)

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

    functors[VLLM_TAG] = _vllm_fn

    def validate(got, expected):
        weights, ids = got[:2]
        ref_weights, ref_ids = expected
        assert weights.shape == ref_weights.shape and weights.dtype == ref_weights.dtype
        assert ids.shape == ref_ids.shape and ids.dtype == ref_ids.dtype
        assert ((ids >= 0) & (ids < gating_output.shape[-1])).all()
        ordered = ids.sort(-1).values
        assert (ordered[:, 1:] != ordered[:, :-1]).all(), "duplicate expert"
        logits = gating_output.float()
        scores = logits.softmax(-1) if scoring_func == "softmax" else logits.sigmoid()
        selection = scores if correction_bias is None else scores + correction_bias
        # Different expert order and exact ties are valid, but selected scores and
        # the weight attached to each actual expert must both be correct.
        torch.testing.assert_close(
            selection.gather(1, ids.long()).sort(-1).values,
            selection.gather(1, ref_ids.long()).sort(-1).values,
            rtol=1e-5,
            atol=1e-5,
        )
        selected = scores.gather(1, ids.long())
        if renormalize:
            selected = selected / selected.sum(-1, keepdim=True)
        torch.testing.assert_close(weights, selected, rtol=1e-3, atol=1e-3)

    bm.compare(
        functors,
        *inputs,
        evidence=dict.fromkeys(
            functors,
            Custom(validate, "selected experts and their weights, independent of tie order"),
        ),
    )
