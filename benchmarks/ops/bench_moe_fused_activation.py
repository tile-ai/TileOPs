"""Fused vs unfused gate/up activation benchmark for MoE expert GEMM.

Compares FusedMoEExpertsNopadPersistent3WGFwdOp with use_fused_activation=True
against use_fused_activation=False across two representative regimes:

  Regime    num_tokens  Description
  --------  ----------  -----------------------------------------------
  prefill       4096    LLaMA/Mistral-scale prefill (large token batch)
  decode         128    Decode-phase (small live batch, latency-bound)

Shape (fused-eligible: ffn_size % 128 == 0):

  H=2048, F=768, E=8, K=2, dtype=bfloat16, activation=silu_and_mul
  Model reference: Qwen2.5/Mistral-scale MoE configuration.

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
  - tileops-unfused: FusedMoEExpertsNopadPersistent3WGFwdOp (use_fused_activation=False)
  - tileops-fused:   FusedMoEExpertsNopadPersistent3WGFwdOp (use_fused_activation=True)
"""

from typing import Any, Optional

import pytest
import torch
import torch.nn.functional as F

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.ops.moe import FusedMoEExpertsNopadPersistent3WGFwdOp
from workloads.workload_base import WorkloadBase

# ---------------------------------------------------------------------------
# Shape constants
# Model reference: Qwen2.5/Mistral-scale MoE configuration
# ---------------------------------------------------------------------------

HIDDEN_SIZE = 2048   # H — model hidden dimension
FFN_SIZE    = 768    # F — per-expert FFN intermediate dim (768 % 128 == 0)
NUM_EXPERTS = 8      # E — total expert count
TOP_K       = 2      # K — experts per token
DTYPE       = torch.bfloat16
ACTIVATION  = "silu_and_mul"


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

class MoEFusedActWorkload(WorkloadBase):
    """Workload descriptor for fused vs unfused activation benchmark."""

    def __init__(self, num_tokens: int):
        self.num_tokens  = num_tokens
        self.hidden_size = HIDDEN_SIZE
        self.ffn_size    = FFN_SIZE
        self.num_experts = NUM_EXPERTS
        self.top_k       = TOP_K
        self.dtype       = DTYPE
        # Primary shape: (num_tokens, hidden_size) — the token tensor footprint.
        self.shape: tuple[int, int] = (num_tokens, HIDDEN_SIZE)

    def gen_inputs(self) -> tuple[Any, ...]:
        torch.manual_seed(42)
        dev = "cuda"
        hidden = torch.randn(
            self.num_tokens, self.hidden_size, dtype=self.dtype, device=dev,
        )
        w_gate_up = torch.randn(
            self.num_experts, self.ffn_size * 2, self.hidden_size,
            dtype=self.dtype, device=dev,
        ) * 0.02
        w_down = torch.randn(
            self.num_experts, self.hidden_size, self.ffn_size,
            dtype=self.dtype, device=dev,
        ) * 0.02
        topk_weights = torch.softmax(
            torch.randn(self.num_tokens, self.top_k, dtype=torch.float32, device=dev),
            dim=-1,
        )
        topk_ids = torch.randint(
            0, self.num_experts,
            (self.num_tokens, self.top_k), dtype=torch.int32, device=dev,
        )
        return hidden, w_gate_up, w_down, topk_weights, topk_ids

    def ref_program(self, *args: Any) -> torch.Tensor:
        """Per-expert PyTorch reference: gate_up GEMM → silu_and_mul → down GEMM → weighted reduce.

        Delegates to _torch_ref_fn() which is also used as the torch-ref timing baseline.
        """
        return _torch_ref_fn(self, *args)


# ---------------------------------------------------------------------------
# Torch reference helper (always available — no external dependency)
# ---------------------------------------------------------------------------


def _torch_ref_fn(
    workload: "MoEFusedActWorkload",
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


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------

class MoEFusedActBenchmark(BenchmarkBase[MoEFusedActWorkload]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        return t.num_tokens * t.top_k * 6 * t.ffn_size * t.hidden_size

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        elem = 2  # bfloat16
        weights = t.num_experts * 3 * t.ffn_size * t.hidden_size * elem
        tokens  = 2 * t.num_tokens * t.hidden_size * elem
        return weights + tokens


# ---------------------------------------------------------------------------
# Benchmark test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "regime, num_tokens",
    [
        pytest.param("prefill", 4096, id="prefill-4096"),
        pytest.param("decode",  128,  id="decode-128"),
    ],
)
def test_moe_fused_activation_bench(regime: str, num_tokens: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device found.")
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        pytest.skip(
            f"SM90 (Hopper) required for 3WG fused-activation kernel; "
            f"device capability is SM{cap[0]}{cap[1]}."
        )

    workload = MoEFusedActWorkload(num_tokens)
    inputs   = workload.gen_inputs()
    hidden, w_gate_up, w_down, topk_weights, topk_ids = inputs

    ws1 = torch.empty(0, dtype=DTYPE, device="cuda")
    ws2 = torch.empty(0, dtype=DTYPE, device="cuda")

    kwargs = dict(
        num_tokens=num_tokens, num_experts=NUM_EXPERTS, top_k=TOP_K,
        hidden_size=HIDDEN_SIZE, ffn_size=FFN_SIZE, dtype=DTYPE,
        activation=ACTIVATION,
    )

    op_unfused = FusedMoEExpertsNopadPersistent3WGFwdOp(**kwargs, use_fused_activation=False)
    op_fused   = FusedMoEExpertsNopadPersistent3WGFwdOp(**kwargs, use_fused_activation=True)

    if not op_fused.use_fused_activation:
        pytest.skip(
            "use_fused_activation=True was downgraded (eligibility check failed); "
            "cannot compare fused vs unfused meaningfully."
        )

    bm = MoEFusedActBenchmark(workload)
    out_unfused = torch.empty(num_tokens, HIDDEN_SIZE, dtype=DTYPE, device="cuda")
    out_fused   = torch.empty(num_tokens, HIDDEN_SIZE, dtype=DTYPE, device="cuda")

    def _run_unfused(hidden, w_gate_up, w_down, topk_weights, topk_ids):
        out_unfused.zero_()
        op_unfused.forward(
            out_unfused, hidden, w_gate_up, w_down, topk_weights, topk_ids,
            expert_map=None, workspace1=ws1, workspace2=ws2, num_experts=NUM_EXPERTS,
        )
        return out_unfused

    def _run_fused(hidden, w_gate_up, w_down, topk_weights, topk_ids):
        out_fused.zero_()
        op_fused.forward(
            out_fused, hidden, w_gate_up, w_down, topk_weights, topk_ids,
            expert_map=None, workspace1=ws1, workspace2=ws2, num_experts=NUM_EXPERTS,
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
            f"[{regime}] Fused and unfused outputs disagree — "
            "do not trust speedup numbers.\n" + str(e)
        ) from e

    # ---- Timing: torch-ref (always runs — unconditional baseline) -----------
    def _run_torch_ref(hidden, w_gate_up, w_down, topk_weights, topk_ids):
        return _torch_ref_fn(workload, hidden, w_gate_up, w_down, topk_weights, topk_ids)

    _run_torch_ref(hidden, w_gate_up, w_down, topk_weights, topk_ids)  # warmup
    torch.cuda.synchronize()

    result_torch = bm.profile(_run_torch_ref, hidden, w_gate_up, w_down, topk_weights, topk_ids)
    BenchmarkReport.record(op_unfused, locals(), result_torch, tag="torch-ref")
    ms_torch = result_torch["latency_ms"]

    # ---- Timing: unfused ----------------------------------------------------
    result_unfused = bm.profile(
        _run_unfused, hidden, w_gate_up, w_down, topk_weights, topk_ids,
    )
    BenchmarkReport.record(
        op_unfused, locals(), result_unfused, tag="tileops-unfused",
    )
    ms_unfused = result_unfused["latency_ms"]

    # ---- Timing: fused ------------------------------------------------------
    result_fused = bm.profile(
        _run_fused, hidden, w_gate_up, w_down, topk_weights, topk_ids,
    )
    BenchmarkReport.record(
        op_fused, locals(), result_fused, tag="tileops-fused",
    )
    ms_fused = result_fused["latency_ms"]

    # ---- Console summary for this regime ------------------------------------
    speedup = ms_unfused / ms_fused if ms_fused > 0 else float("nan")
    note = "  <- fused slower" if speedup < 1.0 else ""
    print(
        f"\n[{regime}] num_tokens={num_tokens}"
        f"  torch-ref={ms_torch:.4f}ms"
        f"  unfused={ms_unfused:.4f}ms  fused={ms_fused:.4f}ms"
        f"  speedup(fused/unfused)={speedup:.3f}x{note}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
