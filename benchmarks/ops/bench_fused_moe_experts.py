"""Benchmark for FusedMoEExpertsNopadPersistent3WGFwdOp.

Measures the permute + grouped-GEMM + unpermute pipeline without routing.
The fused-activation nopad path is compared in one drift-balanced run with the
base default pipeline and vLLM Triton fused_experts.

Workloads match the manifest entries (shared workload set):

  Model              T     H     F     E    K
  Qwen3-235B-A22B   512  7168  2048  128   8   (decode)
  Qwen3-235B-A22B  4096  7168  2048  128   8   (prefill)
  DeepSeek-V3       512  7168  2048  256   8   (decode)
  DeepSeek-V3      4096  7168  2048  256   8   (prefill)

Baselines:
  - base-incumbent: default unfused FusedMoEExpertsNopadPersistent3WGFwdOp
  - vllm-triton:   vLLM Triton fused_experts (default backend)
  - torch-ref:     workload FP32 oracle (only when vLLM is unavailable)

vLLM 0.19.1's CUTLASS MoE module exposes quantized FP8/FP4 paths, not a
matching BF16 entry point, so it is intentionally not reported as an
equivalent baseline.
"""

import pytest
import torch

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts as _vllm_fused_experts,
    )
    _VLLM_TRITON_AVAILABLE = True
except ImportError:
    _VLLM_TRITON_AVAILABLE = False

from benchmarks.benchmark_base import ManifestBenchmark
from tileops.kernels.moe import MoeGroupedGemmPersistent3WGFusedActKernel
from tileops.manifest import load_workloads
from tileops.ops.moe import (
    FusedMoEExpertsNopadPersistent3WGFwdOp,
)
from workloads.moe import MoeExpertsWorkload

_OP_NAME = "FusedMoEExpertsNopadPersistent3WGFwdOp"  # manifest entry name


# Workload


# Benchmark class


# Manifest-driven parametrize


def _manifest_params():
    params = []
    for w in load_workloads(_OP_NAME):
        label = w.get("label", "unlabeled")
        for dtype_str in w["dtypes"]:
            dtype = getattr(torch, dtype_str)
            params.append(pytest.param(
                w["num_tokens"], w["num_experts"], w["top_k"],
                w["hidden_size"], w["ffn_size"], dtype,
                id=f"{label}-{dtype_str}",
            ))
    return params


# Benchmark test


@pytest.mark.parametrize(
    "num_tokens, num_experts, top_k, hidden_size, ffn_size, dtype",
    _manifest_params(),
)
def test_moe_experts_nopad_bench(
    num_tokens: int, num_experts: int, top_k: int, hidden_size: int,
    ffn_size: int, dtype: torch.dtype,
) -> None:
    test = MoeExpertsWorkload(num_tokens, num_experts, top_k, hidden_size, ffn_size, dtype)
    hidden, w1, w2, topk_weights, topk_ids = test.gen_inputs()

    kwargs = dict(
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
        hidden_size=hidden_size, ffn_size=ffn_size,
    )
    candidate_output = torch.empty(
        num_tokens, hidden_size, dtype=dtype, device="cuda"
    )
    incumbent_output = torch.empty_like(candidate_output)
    ws1 = torch.empty(0, dtype=dtype, device="cuda")
    ws2 = torch.empty(0, dtype=dtype, device="cuda")

    candidate = FusedMoEExpertsNopadPersistent3WGFwdOp(
        **kwargs,
        use_fused_activation=True,
    )
    incumbent = FusedMoEExpertsNopadPersistent3WGFwdOp(**kwargs)
    bm = ManifestBenchmark(_OP_NAME, candidate, test)

    def _candidate_fn(hidden, w1, w2, topk_weights, topk_ids):
        candidate.forward(
            candidate_output, hidden, w1, w2, topk_weights, topk_ids,
            expert_map=None, workspace1=ws1, workspace2=ws2, num_experts=num_experts,
        )
        return candidate_output

    def _incumbent_fn(hidden, w1, w2, topk_weights, topk_ids):
        incumbent.forward(
            incumbent_output, hidden, w1, w2, topk_weights, topk_ids,
            expert_map=None, workspace1=ws1, workspace2=ws2, num_experts=num_experts,
        )
        return incumbent_output

    implementations = {
        "tileops-fused-act": _candidate_fn,
        "base-incumbent": _incumbent_fn,
    }
    if _VLLM_TRITON_AVAILABLE:
        implementations["vllm-triton"] = _vllm_fused_experts
    else:
        implementations["torch-ref"] = test.ref_program

    # Compile all paths before timing. The class assertion prevents a measured
    # row from being silently attributed to the fused path after a fallback.
    for fn in implementations.values():
        fn(hidden, w1, w2, topk_weights, topk_ids)
    fused_kernels = candidate._gemm_gate_up.built_kernels(
        "moe_grouped_gemm_fused_act_kernel"
    )
    assert fused_kernels and all(
        isinstance(kernel, MoeGroupedGemmPersistent3WGFusedActKernel)
        for kernel in fused_kernels.values()
    )
    torch.cuda.synchronize()

    bm.compare(
        implementations,
        hidden, w1, w2, topk_weights, topk_ids,
        record_as=candidate,
        params=locals(),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
