"""Benchmark for FusedMoeFwdOp (routed MoE FFN).

Workload shapes come from the op's manifest ``workloads`` (via
``manifest_calls``); the benchmark reports TileOPs latency
alongside the manifest-derived roofline (``op.eval_roofline()``)
and a vLLM / torch-ref baseline.

Coverage:

  Qwen3-235B-A22B (softmax), DeepSeek-V3 (sigmoid), and Kimi K2 (sigmoid with
  a correction bias) — a row passes the bias when it lists it in ``some``.
"""

import pytest
import torch

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts as _vllm_fused_experts,
    )
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        fused_topk as _vllm_fused_topk,
    )

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.moe import FusedMoeFwdOp
from workloads.moe import FusedMoeWorkload


@pytest.mark.parametrize("call", manifest_calls(FusedMoeFwdOp))
def test_fused_moe_fwd_bench(call) -> None:
    test = FusedMoeWorkload(call)
    inputs = test.gen_inputs()
    hidden, gating, w_gate_up, w_down, correction_bias = inputs
    op = FusedMoeFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(op, test)
    torch.testing.assert_close(
        op(*inputs).float(), test.ref_program(*inputs).float(), rtol=3e-2, atol=3e-2
    )

    functors = {"tileops": op}

    # vLLM's ``fused_topk`` has no correction_bias parameter, so routing would diverge
    # from TileOPs on a row that passes one; those rows time the reference instead.
    if _VLLM_AVAILABLE and correction_bias is None:
        top_k, renormalize, scale = op.top_k, op.renormalize, op.routed_scaling_factor

        def _vllm_fn(hidden, gating, w_gate_up, w_down, correction_bias):
            tw, tids, _ = _vllm_fused_topk(
                hidden_states=hidden,
                gating_output=gating,
                topk=top_k,
                renormalize=renormalize,
                scoring_func=op.scoring_func,
            )
            out = _vllm_fused_experts(hidden, w_gate_up, w_down, tw, tids)
            return out * scale if scale != 1.0 else out

        functors["vllm"] = _vllm_fn
    else:
        functors["torch-ref"] = test.ref_program

    bm.compare(functors, *inputs)
