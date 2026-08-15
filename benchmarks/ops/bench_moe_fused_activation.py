"""Fused vs unfused gate/up activation benchmark for MoE expert GEMM.

Measures the two pipelines the op chooses between, so the evidence behind
MoeGroupedGemmPersistent3WGFusedActKernel.wants_fused_epilogue can be re-taken on
another device or dtype. The op picks the fused pipeline by shape; this benchmark
forces each one in turn, at every shape the manifest lists for the op — both the
decode rows, where the op fuses, and the prefill rows, where it does not.

Timing covers the full experts forward() (permute → gate_up GEMM → activation
→ down GEMM → unpermute/weighted reduce). permute and unpermute are identical
across the fused and unfused variants, so the fused-vs-unfused ratio isolates
the gate_up + activation change even though both endpoints are timed end-to-end.

Memory note: the unfused path materialises a [numel, 2*ffn_size] gate_up
intermediate in HBM before reading it back for the activation, whereas the
fused path eliminates that intermediate entirely, so per-variant HBM traffic
differs beyond what calculate_memory() reports (which counts only weights and
the final token tensors).

Correctness gate: both TileOPs variants run once and their outputs are
compared (rtol=3e-2, atol=3e-2) before any timing starts. Mismatches abort
the run.

Baselines recorded in the report table:
  - torch-ref:       per-expert PyTorch loop (gate_up GEMM → silu_and_mul → down GEMM →
                     weighted index_add_); always available, no external dependency
  - tileops-unfused: separate gate/up GEMM and activation kernels
  - tileops-fused:   activation fused into the gate/up GEMM epilogue
"""

from typing import Any, Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.kernels.moe.moe_grouped_gemm_persistent_3wg_fused_act import (
    MoeGroupedGemmPersistent3WGFusedActKernel,
)
from tileops.manifest import load_workloads
from tileops.ops.moe import FusedMoEExpertsNopadPersistent3WGFwdOp
from workloads.moe import MoeFusedActivationWorkload

_OP_NAME = "FusedMoEExpertsNopadPersistent3WGFwdOp"  # manifest entry name

# The torch reference below computes silu_and_mul, so the op must too.
ACTIVATION = "silu_and_mul"


# Workload


# Torch reference helper (always available — no external dependency)


def _torch_ref_fn(
    workload: "MoeFusedActivationWorkload",
    hidden: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Per-expert PyTorch reference for the full MoE experts forward.

    Implements: for each expert e — select tokens assigned to e, compute
    gate_up GEMM → silu_and_mul → down GEMM → weighted index_add_ into
    the output buffer.  Permute-free (index_add_ scatters in-place).
    """
    output_buf = torch.zeros(
        workload.num_tokens, workload.hidden_size,
        dtype=torch.float32, device=hidden.device,
    )
    ids_i64 = topk_ids.to(torch.int64)
    for e in range(workload.num_experts):
        mask = (ids_i64 == e)
        if not mask.any():
            continue
        t_idx, k_idx = mask.nonzero(as_tuple=True)
        h = hidden[t_idx].float()
        gate_up = h @ w_gate_up[e].float().t()
        ffn_dim = w_gate_up.shape[1] // 2
        act = F.silu(gate_up[:, :ffn_dim]) * gate_up[:, ffn_dim:]
        down = act @ w_down[e].float().t()
        output_buf.index_add_(
            0, t_idx, down * topk_weights[t_idx, k_idx].float().unsqueeze(-1),
        )
    return output_buf.to(hidden.dtype)


# Benchmark class


class MoEFusedActBenchmark(BenchmarkBase[MoeFusedActivationWorkload]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        return t.num_tokens * t.top_k * 6 * t.ffn_size * t.hidden_size

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem = 2  # bfloat16
        weights = t.num_experts * 3 * t.ffn_size * t.hidden_size * elem
        tokens  = 2 * t.num_tokens * t.hidden_size * elem
        return weights + tokens


# Forcing a pipeline


def _build(kwargs: dict, fused: bool) -> FusedMoEExpertsNopadPersistent3WGFwdOp:
    """Build the op with the fused-epilogue answer pinned to ``fused``.

    The op asks the fused kernel class whether the shape wants a fused epilogue, so
    a subclass that answers a fixed value selects the pipeline. This benchmark needs
    both answers at every shape, including the shapes where the stock class declines.
    """

    class Pinned(MoeGroupedGemmPersistent3WGFusedActKernel):
        @classmethod
        def wants_fused_epilogue(cls, *args: Any, **kw: Any) -> bool:
            return fused

    return FusedMoEExpertsNopadPersistent3WGFwdOp(
        **kwargs, kernel_map={"moe_grouped_gemm_fused_act_kernel": Pinned},
    )


# Benchmark test


def _manifest_params():
    params = []
    for w in load_workloads(_OP_NAME):
        label = w.get("label", "unlabeled")
        for dtype_str in w["dtypes"]:
            params.append(pytest.param(
                label, w["num_tokens"], w["num_experts"], w["top_k"],
                w["hidden_size"], w["ffn_size"], getattr(torch, dtype_str),
                id=f"{label}-{dtype_str}",
            ))
    return params


@pytest.mark.parametrize(
    "label, num_tokens, num_experts, top_k, hidden_size, ffn_size, dtype",
    _manifest_params(),
)
def test_moe_fused_activation_bench(
    label: str, num_tokens: int, num_experts: int, top_k: int,
    hidden_size: int, ffn_size: int, dtype: torch.dtype,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device found.")
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        pytest.skip(
            f"SM90 (Hopper) required for 3WG fused-activation kernel; "
            f"device capability is SM{cap[0]}{cap[1]}."
        )

    workload = MoeFusedActivationWorkload(
        num_tokens, hidden_size, ffn_size, num_experts, top_k, dtype
    )
    inputs   = workload.gen_inputs()
    hidden, w_gate_up, w_down, topk_weights, topk_ids = inputs

    ws1 = torch.empty(0, dtype=dtype, device="cuda")
    ws2 = torch.empty(0, dtype=dtype, device="cuda")

    kwargs = dict(
        num_tokens=num_tokens, num_experts=num_experts, top_k=top_k,
        hidden_size=hidden_size, ffn_size=ffn_size,
        activation=ACTIVATION,
    )

    op_unfused = _build(kwargs, fused=False)
    op_fused   = _build(kwargs, fused=True)
    assert op_fused._fuses_activation, "the fused pipeline was not selected"

    bm = MoEFusedActBenchmark(workload)
    out_unfused = torch.empty(num_tokens, hidden_size, dtype=dtype, device="cuda")
    out_fused   = torch.empty(num_tokens, hidden_size, dtype=dtype, device="cuda")

    def _run_unfused(hidden, w_gate_up, w_down, topk_weights, topk_ids):
        out_unfused.zero_()
        op_unfused.forward(
            out_unfused, hidden, w_gate_up, w_down, topk_weights, topk_ids,
            expert_map=None, workspace1=ws1, workspace2=ws2, num_experts=num_experts,
        )
        return out_unfused

    def _run_fused(hidden, w_gate_up, w_down, topk_weights, topk_ids):
        out_fused.zero_()
        op_fused.forward(
            out_fused, hidden, w_gate_up, w_down, topk_weights, topk_ids,
            expert_map=None, workspace1=ws1, workspace2=ws2, num_experts=num_experts,
        )
        return out_fused

    # Warmup / JIT compile
    _run_unfused(hidden, w_gate_up, w_down, topk_weights, topk_ids)
    torch.cuda.synchronize()
    ref = out_unfused.clone()

    _run_fused(hidden, w_gate_up, w_down, topk_weights, topk_ids)
    torch.cuda.synchronize()
    fused_result = out_fused.clone()

    # ---- Correctness check BEFORE timing ------------------------------------
    try:
        torch.testing.assert_close(fused_result, ref, rtol=3e-2, atol=3e-2)
    except AssertionError as e:
        raise AssertionError(
            f"[{label}] Fused and unfused outputs disagree — "
            "do not trust speedup numbers.\n" + str(e)
        ) from e

    # ---- Timing: torch-ref (always runs — unconditional baseline) -----------
    def _run_torch_ref(hidden, w_gate_up, w_down, topk_weights, topk_ids):
        return _torch_ref_fn(workload, hidden, w_gate_up, w_down, topk_weights, topk_ids)

    _run_torch_ref(hidden, w_gate_up, w_down, topk_weights, topk_ids)  # warmup
    torch.cuda.synchronize()

    # Recorded by hand: the fused row belongs to a different op.
    results = bm.compare(
        {
            "torch-ref": _run_torch_ref,
            "tileops-unfused": _run_unfused,
            "tileops-fused": _run_fused,
        },
        hidden, w_gate_up, w_down, topk_weights, topk_ids,
    )
    result_torch = results["torch-ref"]
    result_unfused = results["tileops-unfused"]
    result_fused = results["tileops-fused"]
    BenchmarkReport.record(op_unfused, locals(), result_torch, tag="torch-ref")
    BenchmarkReport.record(op_unfused, locals(), result_unfused, tag="tileops-unfused")
    BenchmarkReport.record(op_fused, locals(), result_fused, tag="tileops-fused")
    ms_torch = result_torch["latency_ms"]
    ms_unfused = result_unfused["latency_ms"]
    ms_fused = result_fused["latency_ms"]

    # ---- Console summary for this workload ----------------------------------
    speedup = ms_unfused / ms_fused if ms_fused > 0 else float("nan")
    note = "  <- fused slower" if speedup < 1.0 else ""
    print(
        f"\n[{label}] num_tokens={num_tokens}"
        f"  torch-ref={ms_torch:.4f}ms"
        f"  unfused={ms_unfused:.4f}ms  fused={ms_fused:.4f}ms"
        f"  speedup(fused/unfused)={speedup:.3f}x{note}"
    )
