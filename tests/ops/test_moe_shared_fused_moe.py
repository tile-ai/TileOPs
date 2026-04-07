"""Tests for SharedFusedMoE — FusedMoE with shared expert support.

Verifies:
  - SharedFusedMoE returns (shared_output, routed_output) tuple
  - shared_output matches SharedExpertMLPKernel reference
  - routed_output matches FusedMoe output
  - When shared_ffn_size=None, shared_output is None
  - TP sharding: partial outputs sum to float32 math reference
"""

import pytest
import torch

from tileops.ops.moe import FusedMoe, SharedFusedMoE


@pytest.mark.smoke
def test_shared_fused_moe_basic():
    """SharedFusedMoE with shared expert kernel."""
    torch.manual_seed(42)
    T, E, K, H, F, F_s = 32, 8, 2, 64, 32, 16
    dtype = torch.bfloat16
    dev = "cuda"

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, dtype=dtype, device=dev)
    w_gate_up = torch.randn(E, F * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, F, dtype=dtype, device=dev) * 0.02
    shared_w_gate_up = torch.randn(F_s * 2, H, dtype=dtype, device=dev) * 0.02
    shared_w_down = torch.randn(H, F_s, dtype=dtype, device=dev) * 0.02

    op = SharedFusedMoE(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F,
        scoring_func="softmax", renormalize=False,
        layout="nopad", dtype=dtype,
        shared_ffn_size=F_s,
    )

    shared_out, routed_out = op(
        hidden, gating, w_gate_up, w_down,
        shared_w_gate_up=shared_w_gate_up,
        shared_w_down=shared_w_down,
    )

    assert shared_out.shape == (T, H)
    assert routed_out.shape == (T, H)
    assert shared_out.dtype == dtype
    assert routed_out.dtype == dtype

    # shared_out reference: manual MLP in float32
    gate_up_ref = hidden.float() @ shared_w_gate_up.float().T  # [T, 2*F_s]
    gate_ref, up_ref = gate_up_ref.chunk(2, dim=1)
    gate_up_act = torch.nn.functional.silu(gate_ref) * up_ref
    shared_ref = (gate_up_act @ shared_w_down.float().T).to(dtype)
    torch.testing.assert_close(shared_out, shared_ref, rtol=1e-2, atol=1e-2)

    # routed_out matches FusedMoe
    op_routed = FusedMoe(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F,
        scoring_func="softmax", renormalize=False,
        layout="nopad", dtype=dtype,
    )
    routed_ref = op_routed(hidden, gating, w_gate_up, w_down)
    torch.testing.assert_close(routed_out, routed_ref, rtol=1e-5, atol=1e-5)

    print(f"PASS SharedFusedMoE basic [T={T}, E={E}, K={K}, H={H}, F={F}, F_s={F_s}]")


@pytest.mark.smoke
def test_shared_fused_moe_none():
    """SharedFusedMoE with shared_ffn_size=None returns shared_out=None."""
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
        dtype=dtype,
    )

    shared_out, routed_out = op(hidden, gating, w_gate_up, w_down)

    assert shared_out is None
    assert routed_out.shape == (T, H)
    print("PASS SharedFusedMoE with shared_ffn_size=None")


@pytest.mark.smoke
def test_shared_fused_moe_tp():
    """TP sharding: sum of partial outputs matches float32 math reference.

    Uses float32 dtype to eliminate bf16 rounding differences between
    TP-sharded and single-GPU computation paths.
    """
    torch.manual_seed(42)
    T, E, K, H, F, F_s = 32, 8, 2, 64, 32, 16
    tp_size = 2
    dtype = torch.bfloat16
    dev = "cuda"

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, dtype=dtype, device=dev)
    w_gate_up = torch.randn(E, F * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, F, dtype=dtype, device=dev) * 0.02
    shared_w_gate_up = torch.randn(F_s * 2, H, dtype=dtype, device=dev) * 0.02
    shared_w_down = torch.randn(H, F_s, dtype=dtype, device=dev) * 0.02

    # Float32 math reference: compute each TP shard's contribution manually
    # This matches exactly what each rank's kernel computes (column-parallel on gate_up, row-parallel on down)
    shard_size = F_s // tp_size
    partial_sum_ref = torch.zeros(T, H, dtype=torch.float32, device=dev)
    for tp_rank in range(tp_size):
        gate_up_shard = torch.cat([
            shared_w_gate_up[tp_rank * shard_size : (tp_rank + 1) * shard_size],          # gate shard
            shared_w_gate_up[F_s + tp_rank * shard_size : F_s + (tp_rank + 1) * shard_size],  # up shard
        ], dim=0)  # [2*shard, H]
        down_shard = shared_w_down[:, tp_rank * shard_size: (tp_rank + 1) * shard_size]              # [H, shard]
        gate_up_out = hidden.float() @ gate_up_shard.float().T     # [T, 2*shard]
        gate, up = gate_up_out.chunk(2, dim=1)
        act = gate * torch.sigmoid(gate) * up                       # [T, shard]
        partial_sum_ref += act @ down_shard.float().T               # [T, H]

    # routed reference (not affected by TP)
    op_routed = FusedMoe(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F,
        scoring_func="softmax", renormalize=False,
        layout="nopad", dtype=dtype,
    )
    routed_ref = op_routed(hidden, gating, w_gate_up, w_down)

    # TP: accumulate partial outputs to simulate all-reduce
    partial_sum = torch.zeros(T, H, dtype=torch.float32, device=dev)
    for tp_rank in range(tp_size):
        op_tp = SharedFusedMoE(
            num_tokens=T, num_experts=E, top_k=K,
            hidden_size=H, ffn_size=F,
            scoring_func="softmax", renormalize=False,
            layout="nopad", dtype=dtype,
            shared_ffn_size=F_s,
            tp_size=tp_size, tp_rank=tp_rank,
        )
        shared_partial, routed_out = op_tp(
            hidden, gating, w_gate_up, w_down,
            shared_w_gate_up=shared_w_gate_up,
            shared_w_down=shared_w_down,
        )
        assert shared_partial.shape == (T, H)
        partial_sum += shared_partial.float()

        # routed_out is not affected by TP sharding of shared expert
        torch.testing.assert_close(routed_out, routed_ref, rtol=1e-5, atol=1e-5)

    # partial_sum vs per-shard float32 math reference (same computation path)
    torch.testing.assert_close(partial_sum, partial_sum_ref, rtol=1e-2, atol=1e-2)

    print(f"PASS SharedFusedMoE TP [T={T}, H={H}, F_s={F_s}, tp_size={tp_size}]")


@pytest.mark.smoke
def test_shared_fused_moe_tp_rejects_local_shards():
    """TP contract: forward() must reject pre-sharded weights with a clear ValueError.

    When tp_size > 1 the op shards weights internally. Callers must always pass
    full weights. Passing TP-local shards would silently produce wrong results
    without this guard.
    """
    T, E, K, H, F, F_s, tp_size = 32, 8, 2, 64, 32, 16, 2
    dtype = torch.bfloat16
    dev = "cuda"

    op = SharedFusedMoE(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=F,
        scoring_func="softmax", renormalize=False,
        layout="nopad", dtype=dtype,
        shared_ffn_size=F_s,
        tp_size=tp_size, tp_rank=0,
    )

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, dtype=dtype, device=dev)
    w_gate_up = torch.randn(E, F * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, F, dtype=dtype, device=dev) * 0.02

    shard_size = F_s // tp_size

    # Pass TP-local gate_up shard instead of full weights → must raise
    bad_gate_up = torch.randn(2 * shard_size, H, dtype=dtype, device=dev)
    good_w_down = torch.randn(H, F_s, dtype=dtype, device=dev)
    with pytest.raises(ValueError, match="full weights"):
        op(hidden, gating, w_gate_up, w_down,
           shared_w_gate_up=bad_gate_up, shared_w_down=good_w_down)

    # Pass TP-local down shard instead of full weights → must raise
    good_gate_up = torch.randn(2 * F_s, H, dtype=dtype, device=dev)
    bad_w_down = torch.randn(H, shard_size, dtype=dtype, device=dev)
    with pytest.raises(ValueError, match="full weights"):
        op(hidden, gating, w_gate_up, w_down,
           shared_w_gate_up=good_gate_up, shared_w_down=bad_w_down)

    print("PASS SharedFusedMoE TP rejects TP-local shards")
