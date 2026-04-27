"""Tests for SharedFusedMoE — FusedMoE with shared expert support.

Verifies:
  - SharedFusedMoE returns (shared_output, routed_output) tuple
  - shared_output matches SharedExpertMLPKernel reference
  - routed_output matches FusedMoe output
  - When shared_ffn_size=None, shared_output is None
  - TP sharding: partial outputs sum to float32 math reference
  - pre_sharded=True: output is bit-for-bit identical to stateless path
  - pre_sharded=True: rejects full weights with a clear ValueError
  - pre_sharded=True: does not allocate shard copy buffers in forward()
"""

import pytest
import torch
import torch.nn.functional as F

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
    T, E, K, H, ffn, F_s = 32, 8, 2, 64, 32, 16
    tp_size = 2
    dtype = torch.bfloat16
    dev = "cuda"

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, dtype=dtype, device=dev)
    w_gate_up = torch.randn(E, ffn * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, ffn, dtype=dtype, device=dev) * 0.02
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
        act = F.silu(gate) * up                              # [T, shard]
        partial_sum_ref += act @ down_shard.float().T               # [T, H]

    # routed reference (not affected by TP)
    op_routed = FusedMoe(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=ffn,
        scoring_func="softmax", renormalize=False,
        layout="nopad", dtype=dtype,
    )
    routed_ref = op_routed(hidden, gating, w_gate_up, w_down)

    # TP: accumulate partial outputs to simulate all-reduce
    partial_sum = torch.zeros(T, H, dtype=torch.float32, device=dev)
    for tp_rank in range(tp_size):
        op_tp = SharedFusedMoE(
            num_tokens=T, num_experts=E, top_k=K,
            hidden_size=H, ffn_size=ffn,
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
    T, E, K, H, ffn, F_s, tp_size = 32, 8, 2, 64, 32, 16, 2
    dtype = torch.bfloat16
    dev = "cuda"

    op = SharedFusedMoE(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=ffn,
        scoring_func="softmax", renormalize=False,
        layout="nopad", dtype=dtype,
        shared_ffn_size=F_s,
        tp_size=tp_size, tp_rank=0,
    )

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, dtype=dtype, device=dev)
    w_gate_up = torch.randn(E, ffn * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, ffn, dtype=dtype, device=dev) * 0.02

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


@pytest.mark.smoke
@pytest.mark.parametrize("tp_size", [2, 4])
def test_shared_fused_moe_pre_sharded_matches_stateless(tp_size):
    """pre_sharded=True must produce identical output to pre_sharded=False (stateless).

    Uses bfloat16 to match the kernel's default dtype.
    Slices weights with exactly the same logic forward() uses internally, then
    passes them via pre_sharded=True and asserts exact bit-for-bit match
    (rtol=0, atol=0 is valid here: both paths feed identical tensors to the
    same deterministic kernel — no rounding differences are introduced).

    routed_out is not compared because pre_sharded only affects the shared
    expert path; the routed expert call (super().forward()) is unchanged.
    """
    torch.manual_seed(42)
    # F_s=32 is divisible by both tp_size=2 and tp_size=4 (parametrized above)
    T, E, K, H, ffn, F_s = 32, 8, 2, 64, 32, 32
    dtype = torch.bfloat16
    dev = "cuda"

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, dtype=dtype, device=dev)
    w_gate_up = torch.randn(E, ffn * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, ffn, dtype=dtype, device=dev) * 0.02
    shared_w_gate_up = torch.randn(F_s * 2, H, dtype=dtype, device=dev) * 0.02
    shared_w_down = torch.randn(H, F_s, dtype=dtype, device=dev) * 0.02

    for tp_rank in range(tp_size):
        shard_size = F_s // tp_size
        r, s = tp_rank, shard_size

        # Stateless path (full weights, op shards internally)
        op_stateless = SharedFusedMoE(
            num_tokens=T, num_experts=E, top_k=K,
            hidden_size=H, ffn_size=ffn,
            scoring_func="softmax", renormalize=False,
            layout="nopad", dtype=dtype,
            shared_ffn_size=F_s,
            tp_size=tp_size, tp_rank=tp_rank,
        )
        shared_stateless, _ = op_stateless(
            hidden, gating, w_gate_up, w_down,
            shared_w_gate_up=shared_w_gate_up,
            shared_w_down=shared_w_down,
        )

        # Pre-shard weights manually (same logic as forward() does internally)
        gate_up_local = torch.cat([
            shared_w_gate_up[r * s : (r + 1) * s],
            shared_w_gate_up[F_s + r * s : F_s + (r + 1) * s],
        ], dim=0).contiguous()  # [2*s, H]
        down_local = shared_w_down[:, r * s : (r + 1) * s].contiguous()  # [H, s]

        # pre_sharded path (TP-local weights, no internal slicing)
        op_pre = SharedFusedMoE(
            num_tokens=T, num_experts=E, top_k=K,
            hidden_size=H, ffn_size=ffn,
            scoring_func="softmax", renormalize=False,
            layout="nopad", dtype=dtype,
            shared_ffn_size=F_s,
            tp_size=tp_size, tp_rank=tp_rank,
            pre_sharded=True,
        )
        shared_pre, _ = op_pre(
            hidden, gating, w_gate_up, w_down,
            shared_w_gate_up=gate_up_local,
            shared_w_down=down_local,
        )

        # rtol=0, atol=0: both paths feed identical tensors to the kernel,
        # so outputs must be bit-for-bit equal (SharedExpertMLPKernel is deterministic).
        torch.testing.assert_close(shared_pre, shared_stateless, rtol=0, atol=0)

    print(f"PASS pre_sharded matches stateless [tp_size={tp_size}, F_s={F_s}]")


@pytest.mark.smoke
def test_shared_fused_moe_pre_sharded_rejects_full_weights():
    """pre_sharded=True must raise ValueError when full weights are passed.

    The error message must mention 'pre_sharded=True' so callers can diagnose
    the mismatch. This is distinct from the stateless-path error ('full weights').
    """
    T, E, K, H, ffn, F_s, tp_size = 32, 8, 2, 64, 32, 16, 2
    dtype = torch.bfloat16
    dev = "cuda"

    op = SharedFusedMoE(
        num_tokens=T, num_experts=E, top_k=K,
        hidden_size=H, ffn_size=ffn,
        scoring_func="softmax", renormalize=False,
        layout="nopad", dtype=dtype,
        shared_ffn_size=F_s,
        tp_size=tp_size, tp_rank=0,
        pre_sharded=True,
    )
    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, dtype=dtype, device=dev)
    w_gate_up = torch.randn(E, ffn * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, ffn, dtype=dtype, device=dev) * 0.02
    shard_size = F_s // tp_size

    # Pass full gate_up (wrong shape) → must raise with pre_sharded=True mention
    full_gate_up = torch.randn(2 * F_s, H, dtype=dtype, device=dev)
    good_down_shard = torch.randn(H, shard_size, dtype=dtype, device=dev)
    with pytest.raises(ValueError, match="pre_sharded=True"):
        op(hidden, gating, w_gate_up, w_down,
           shared_w_gate_up=full_gate_up, shared_w_down=good_down_shard)

    # Pass full down (wrong shape) → must raise with pre_sharded=True mention
    good_gate_up_shard = torch.randn(2 * shard_size, H, dtype=dtype, device=dev)
    full_down = torch.randn(H, F_s, dtype=dtype, device=dev)
    with pytest.raises(ValueError, match="pre_sharded=True"):
        op(hidden, gating, w_gate_up, w_down,
           shared_w_gate_up=good_gate_up_shard, shared_w_down=full_down)

    print("PASS pre_sharded=True rejects full weights with clear error message")


@pytest.mark.smoke
def test_shared_fused_moe_pre_sharded_zero_transient_alloc():
    """AC-2: pre_sharded=True must not allocate shard copy buffers in forward().

    The stateless path allocates gate_up_shard [2*s, H] and down_shard [H, s]
    via cat+contiguous on every forward() call (~95 MB at Kimi K2 tp8 scale).
    pre_sharded=True must not allocate these copies.

    Measurement strategy:
      - Use large enough H/F_s so shard_alloc_bytes >> other forward() allocations.
      - Warmup first to eliminate JIT/kernel-cache allocations.
      - Compare (pre_sharded=True delta) vs (pre_sharded=False delta):
        the difference should be approximately shard_alloc_bytes.
      - pre_sharded=True delta must be < shard_alloc_bytes (no weight copies).
      - pre_sharded=False delta must be >= shard_alloc_bytes (copies present).
    """
    # Use larger dims so shard copies (MB-scale) dominate over output tensors (KB-scale)
    T, E, K, H, ffn, F_s, tp_size = 16, 8, 2, 512, 32, 512, 2
    dtype = torch.bfloat16
    dev = "cuda"
    shard_size = F_s // tp_size

    # Shard copy budget: gate_up_shard [2*s, H] + down_shard [H, s], bfloat16 = 2 bytes
    # With H=512, s=256: 2*256*512*2 + 512*256*2 = 524288 + 262144 = 786432 bytes (~768KB)
    shard_alloc_bytes = (2 * shard_size * H + H * shard_size) * 2

    shared_w_gate_up = torch.randn(F_s * 2, H, dtype=dtype, device=dev) * 0.02
    shared_w_down = torch.randn(H, F_s, dtype=dtype, device=dev) * 0.02
    gate_up_local = torch.cat([
        shared_w_gate_up[:shard_size],
        shared_w_gate_up[F_s : F_s + shard_size],
    ], dim=0).contiguous()
    down_local = shared_w_down[:, :shard_size].contiguous()

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, dtype=dtype, device=dev)
    w_gate_up_r = torch.randn(E, ffn * 2, H, dtype=dtype, device=dev) * 0.02
    w_down_r = torch.randn(E, H, ffn, dtype=dtype, device=dev) * 0.02

    def make_op(pre_sharded):
        return SharedFusedMoE(
            num_tokens=T, num_experts=E, top_k=K,
            hidden_size=H, ffn_size=ffn,
            scoring_func="softmax", renormalize=False,
            layout="nopad", dtype=dtype,
            shared_ffn_size=F_s,
            tp_size=tp_size, tp_rank=0,
            pre_sharded=pre_sharded,
        )

    def measure_transient(op, w_gate_up_arg, w_down_arg):
        """Returns (peak - before) bytes during a single post-warmup forward()."""
        # Warmup: eliminates JIT, kernel cache, and first-call allocations
        op(hidden, gating, w_gate_up_r, w_down_r,
           shared_w_gate_up=w_gate_up_arg, shared_w_down=w_down_arg)
        torch.cuda.synchronize()
        # Measure
        torch.cuda.reset_peak_memory_stats(dev)
        before = torch.cuda.memory_allocated(dev)
        op(hidden, gating, w_gate_up_r, w_down_r,
           shared_w_gate_up=w_gate_up_arg, shared_w_down=w_down_arg)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(dev)
        return peak - before

    pre_delta = measure_transient(make_op(True), gate_up_local, down_local)
    stateless_delta = measure_transient(make_op(False), shared_w_gate_up, shared_w_down)

    # pre_sharded=True: delta must be below shard copy budget (no weight copies)
    assert pre_delta < shard_alloc_bytes, (
        f"pre_sharded=True allocated {pre_delta} bytes, expected < {shard_alloc_bytes} "
        f"(shard copy budget = {shard_alloc_bytes // 1024}KB). "
        f"Weight copies are being made unexpectedly."
    )
    # pre_sharded=False: delta must be at least shard copy budget (copies present)
    assert stateless_delta >= shard_alloc_bytes, (
        f"pre_sharded=False allocated {stateless_delta} bytes, expected >= {shard_alloc_bytes}. "
        f"Shard copies may have been optimised away unexpectedly."
    )
    print(
        f"PASS transient alloc: pre_sharded=True={pre_delta // 1024}KB "
        f"(< {shard_alloc_bytes // 1024}KB threshold), "
        f"pre_sharded=False={stateless_delta // 1024}KB"
    )
