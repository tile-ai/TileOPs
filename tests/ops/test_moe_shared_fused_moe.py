"""Tests for SharedFusedMoE — FusedMoE with shared expert support.

Verifies:
  - SharedFusedMoE returns (shared_output, routed_output) tuple
  - shared_output matches direct shared_experts_fn call
  - routed_output matches FusedMoe output
  - When shared_experts_fn=None, shared_output is None
"""

import pytest
import torch
import torch.nn as nn

from tileops.ops.moe import FusedMoe, SharedFusedMoE


@pytest.mark.full
def test_shared_fused_moe_basic():
    """Basic test: SharedFusedMoE with simple shared expert MLP."""
    torch.manual_seed(42)
    T, E, K, H, F = 32, 8, 2, 64, 32
    dtype = torch.bfloat16
    dev = "cuda"

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, dtype=dtype, device=dev)
    w_gate_up = torch.randn(E, F * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, F, dtype=dtype, device=dev) * 0.02

    # Create shared expert MLP
    shared_mlp = nn.Sequential(
        nn.Linear(H, F, bias=False, dtype=dtype, device=dev),
        nn.SiLU(),
        nn.Linear(F, H, bias=False, dtype=dtype, device=dev),
    )

    # SharedFusedMoE
    op_shared = SharedFusedMoE(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F,
        scoring_func="softmax", renormalize=False,
        layout="nopad", dtype=dtype,
        shared_experts_fn=shared_mlp,
    )

    shared_out, routed_out = op_shared(hidden, gating, w_gate_up, w_down)

    # Verify shapes and types
    assert shared_out.shape == (T, H)
    assert routed_out.shape == (T, H)
    assert shared_out.dtype == dtype
    assert routed_out.dtype == dtype

    # Verify shared_out matches direct MLP call
    shared_ref = shared_mlp(hidden)
    torch.testing.assert_close(shared_out, shared_ref, rtol=1e-5, atol=1e-5)

    # Verify routed_out matches FusedMoe
    op_routed = FusedMoe(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F,
        scoring_func="softmax", renormalize=False,
        layout="nopad", dtype=dtype,
    )
    routed_ref = op_routed(hidden, gating, w_gate_up, w_down)
    torch.testing.assert_close(routed_out, routed_ref, rtol=1e-5, atol=1e-5)

    print(f"PASS SharedFusedMoE basic [T={T}, E={E}, K={K}, H={H}, F={F}]")


@pytest.mark.full
def test_shared_fused_moe_none():
    """Test SharedFusedMoE with shared_experts_fn=None."""
    torch.manual_seed(42)
    T, E, K, H, F = 16, 4, 2, 32, 16
    dtype = torch.bfloat16
    dev = "cuda"

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, dtype=dtype, device=dev)
    w_gate_up = torch.randn(E, F * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, F, dtype=dtype, device=dev) * 0.02

    op = SharedFusedMoE(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F,
        shared_experts_fn=None,
        dtype=dtype,
    )

    shared_out, routed_out = op(hidden, gating, w_gate_up, w_down)

    assert shared_out is None
    assert routed_out.shape == (T, H)
    print("PASS SharedFusedMoE with shared_experts_fn=None")
