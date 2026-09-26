"""Benchmarks for FusedMoEExpertsFwdOp and IndexedExpertMLPFwdOp.

Measures the permute + grouped-GEMM + unpermute pipeline without routing and compares it
against vLLM Triton fused_experts and vLLM CUTLASS fused_experts (when available).

Workloads match the manifest entries (shared workload set):

  Model              T     H     F     E    K
  Qwen3-235B-A22B   512  7168  2048  128   8   (decode)
  Qwen3-235B-A22B  4096  7168  2048  128   8   (prefill)
  DeepSeek-V3       512  7168  2048  256   8   (decode)
  DeepSeek-V3      4096  7168  2048  256   8   (prefill)

Baselines:
  - tileops:            FusedMoEExpertsFwdOp
  - vllm-triton:       vLLM Triton fused_experts (default backend)
  - vllm-cutlass:      vLLM CUTLASS fused_experts (when importable)
  - torch-ref:         per-expert GEMM loop with index_add_ (fallback)

``IndexedExpertMLPFwdOp`` is the small-route backend the composite picks at two routes
per expert or fewer.
Its own workloads sit in that band, and it is measured against the staged pipeline the
composite runs everywhere else, which is what the indexed path has to beat to be chosen.
"""

import warnings

import pytest
import torch

try:
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        fused_experts as _vllm_fused_experts,
    )

    _VLLM_TRITON_AVAILABLE = True
except ImportError:
    _VLLM_TRITON_AVAILABLE = False

try:
    from vllm.model_executor.layers.fused_moe.cutlass_moe import (
        cutlass_moe_fp16 as _vllm_cutlass_moe,
    )

    _VLLM_CUTLASS_AVAILABLE = True
except ImportError:
    try:
        from vllm.model_executor.layers.fused_moe.cutlass_moe import (
            cutlass_moe as _vllm_cutlass_moe,
        )

        _VLLM_CUTLASS_AVAILABLE = True
    except ImportError as _cutlass_import_err:
        _VLLM_CUTLASS_AVAILABLE = False
        warnings.warn(
            f"vLLM CUTLASS MoE baseline unavailable ({_cutlass_import_err}); "
            "the vllm-cutlass column will be omitted from results.",
            RuntimeWarning,
            stacklevel=2,
        )

from benchmarks.benchmark_base import ManifestBenchmark, manifest_calls
from tileops.ops.moe import (
    ContiguousLayoutSpec,
    FusedMoEExpertsFwdOp,
    IndexedExpertMLPFwdOp,
    MoeExpertMLPFwdOp,
    MoePostPermuteFwdOp,
    MoePrePermuteFwdOp,
    RoutingEpilogueSpec,
)
from workloads.moe import IndexedExpertMLPWorkload, MoeExpertsWorkload


def _assert_matches(workload, inputs) -> None:
    torch.testing.assert_close(
        inputs[0].float(), workload.ref_program(*inputs).float(), rtol=3e-2, atol=3e-2
    )


@pytest.mark.parametrize("call", manifest_calls(FusedMoEExpertsFwdOp))
def test_moe_experts_bench(call) -> None:
    test = MoeExpertsWorkload(call)
    inputs = test.gen_inputs()
    output, hidden, w1, w2, topk_weights, topk_ids = inputs
    experts = FusedMoEExpertsFwdOp(**call.arguments({}))
    bm = ManifestBenchmark(experts, test)
    experts(*inputs)
    _assert_matches(test, inputs)

    def _experts_fn(hidden, w1, w2, topk_weights, topk_ids):
        experts.forward(output, hidden, w1, w2, topk_weights, topk_ids)
        return output

    functors = {"tileops": _experts_fn}

    # -- vLLM Triton baseline -------------------------------------------------
    if _VLLM_TRITON_AVAILABLE:

        def _vllm_triton_fn(hidden, w1, w2, topk_weights, topk_ids):
            return _vllm_fused_experts(hidden, w1, w2, topk_weights, topk_ids)

        _vllm_triton_fn(hidden, w1, w2, topk_weights, topk_ids)  # warmup
        torch.cuda.synchronize()

        functors["vllm-triton"] = _vllm_triton_fn

    # -- vLLM CUTLASS baseline ------------------------------------------------
    if _VLLM_CUTLASS_AVAILABLE:
        try:

            def _vllm_cutlass_fn(hidden, w1, w2, topk_weights, topk_ids):
                return _vllm_cutlass_moe(hidden, w1, w2, topk_weights, topk_ids)

            _vllm_cutlass_fn(hidden, w1, w2, topk_weights, topk_ids)  # warmup
            torch.cuda.synchronize()

            functors["vllm-cutlass"] = _vllm_cutlass_fn
        except Exception as e:
            print(f"[vllm-cutlass] skipped: {e}")

    # -- Torch fallback -------------------------------------------------------
    if not _VLLM_TRITON_AVAILABLE:

        def _torch_fn(hidden, w1, w2, topk_weights, topk_ids):
            return test.ref_program(output, hidden, w1, w2, topk_weights, topk_ids)

        functors["torch-ref"] = _torch_fn

    bm.compare(functors, hidden, w1, w2, topk_weights, topk_ids)


@pytest.mark.parametrize("call", manifest_calls(IndexedExpertMLPFwdOp))
def test_indexed_expert_mlp_bench(call) -> None:
    test = IndexedExpertMLPWorkload(call)
    inputs = test.gen_inputs()
    output, hidden, w1, w2, topk_weights, topk_ids = inputs
    indexed = IndexedExpertMLPFwdOp(**call.arguments({}))
    indexed(*inputs)
    _assert_matches(test, inputs)

    def _indexed_fn(hidden, w1, w2, topk_weights, topk_ids):
        indexed.forward(output, hidden, w1, w2, topk_weights, topk_ids)
        return output

    # The staged pipeline is what the composite runs on every other shape, so it is the
    # comparator the indexed path has to beat.
    layout = ContiguousLayoutSpec.tight_physical_psum()
    pre = MoePrePermuteFwdOp(layout, num_local_experts=w1.shape[0])
    mlp = MoeExpertMLPFwdOp(layout)
    epilogue = RoutingEpilogueSpec(routed_scaling_factor=indexed.routed_scaling_factor)
    post = MoePostPermuteFwdOp(layout, epilogue)
    staged_output = torch.empty_like(output)

    def _staged_fn(hidden, w1, w2, topk_weights, topk_ids):
        expert_input, physical_ends, inverse = pre(hidden, topk_ids)
        expert_output = mlp(expert_input, w1, w2, physical_ends)
        post(expert_output, topk_weights, inverse, out=staged_output)
        return staged_output

    functors = {"tileops": _indexed_fn, "staged": _staged_fn}
    for fn in functors.values():
        fn(hidden, w1, w2, topk_weights, topk_ids)
    torch.cuda.synchronize()

    ManifestBenchmark(indexed, test).compare(functors, hidden, w1, w2, topk_weights, topk_ids)
