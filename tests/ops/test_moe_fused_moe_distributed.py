"""Distributed tests for FusedMoe — Multi-GPU Expert Parallelism.

Run with:
    torchrun --nproc_per_node=2 -m pytest tests/ops/test_moe_fused_moe_distributed.py -m smoke -vvs
    torchrun --nproc_per_node=8 -m pytest tests/ops/test_moe_fused_moe_distributed.py -m full -vvs

Note: These tests require multiple GPUs and will be skipped in CI environments.

Verifies:
  - TileOPs expert_map local filtering correctness in real multi-GPU setup
  - Each rank owns E_global/world_size experts, computes partial output
  - All-reduce sum matches single-GPU full computation
  - SharedFusedMoE with shared expert in multi-GPU EP mode
  - TileOPs vs vLLM multi-GPU EP correctness
"""

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from tests.test_base import FixtureBase
from tileops.ops.moe import FusedMoe, SharedFusedMoE

# Skip all tests in this module if not enough GPUs
pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Distributed tests require at least 2 GPUs"
)

# vLLM optional import
try:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts as _vllm_fused_experts
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False
    _vllm_fused_experts = None


def setup_distributed():
    """Initialize distributed environment (called once per process)."""
    if not dist.is_available():
        pytest.skip("torch.distributed not available")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank >= torch.cuda.device_count():
        pytest.skip(f"Rank {rank} exceeds available GPUs ({torch.cuda.device_count()})")

    torch.cuda.set_device(rank)

    return rank, world_size


class DistributedFixture(FixtureBase):
    """Fixture for distributed EP tests."""
    PARAMS = [
        (
            "T, E_global, K, H, F, world_size",
            [
                pytest.param(64, 8, 2, 128, 64, 2, marks=pytest.mark.smoke, id="smoke-ep2"),
                pytest.param(512, 384, 8, 7168, 2048, 8, marks=pytest.mark.full, id="kimi-k2-ep8"),
            ],
        )
    ]


class SharedDistributedFixture(FixtureBase):
    """Fixture for SharedFusedMoE distributed EP tests."""
    PARAMS = [
        (
            "T, E_global, K, H, F, shared_F, world_size",
            [
                pytest.param(64, 8, 2, 128, 64, 32, 2, marks=pytest.mark.smoke, id="smoke-shared-ep2"),
                pytest.param(512, 384, 8, 7168, 2048, 18432, 8, marks=pytest.mark.full, id="kimi-k2-shared-ep8"),
            ],
        )
    ]


@DistributedFixture
def test_kimi_k2_ep_distributed(T, E_global, K, H, F, world_size):
    """Multi-GPU EP test with Kimi K2 config.

    Each rank owns E_global/world_size experts. Verifies:
      sum(rank_outputs) == full_output (on rank-0)
    """
    rank, actual_world_size = setup_distributed()

    if actual_world_size != world_size:
        pytest.skip(f"This test requires {world_size} GPUs, got {actual_world_size}")

    dtype = torch.bfloat16
    dev = f"cuda:{rank}"

    # All ranks use same seed for reproducibility
    torch.manual_seed(42)
    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E_global, dtype=dtype, device=dev)
    correction_bias = torch.randn(E_global, dtype=torch.float32, device=dev) * 0.1

    # Each rank owns E_local experts
    E_local = E_global // world_size
    assert E_global % world_size == 0, "E_global must be divisible by world_size"

    # All ranks generate same full weights, then slice
    w_gate_up_full = torch.randn(E_global, F * 2, H, dtype=dtype, device=dev) * 0.02
    w_down_full = torch.randn(E_global, H, F, dtype=dtype, device=dev) * 0.02

    # Slice local weights
    start_expert = rank * E_local
    end_expert = (rank + 1) * E_local
    w_gate_up_local = w_gate_up_full[start_expert:end_expert].clone()
    w_down_local = w_down_full[start_expert:end_expert].clone()

    # Build expert_map: only local experts are valid
    expert_map = torch.full((E_global,), -1, dtype=torch.int32, device=dev)
    expert_map[start_expert:end_expert] = torch.arange(E_local, dtype=torch.int32, device=dev)

    # TileOPs: local computation with expert_map
    op_local = FusedMoe(
        num_tokens=T, num_experts=E_global, top_k=K,
        hidden_size=H, ffn_size=F,
        scoring_func="sigmoid", renormalize=True, with_correction_bias=True,
        routed_scaling_factor=2.827,
        layout="nopad", dtype=dtype,
        expert_map=expert_map,
    )
    out_local = op_local(hidden, gating, w_gate_up_local, w_down_local, correction_bias)

    # All-reduce to sum partial outputs
    dist.all_reduce(out_local, op=dist.ReduceOp.SUM)

    # Rank-0 computes full reference and compares
    if rank == 0:
        op_full = FusedMoe(
            num_tokens=T, num_experts=E_global, top_k=K,
            hidden_size=H, ffn_size=F,
            scoring_func="sigmoid", renormalize=True, with_correction_bias=True,
            routed_scaling_factor=2.827,
            layout="nopad", dtype=dtype,
        )
        out_full = op_full(hidden, gating, w_gate_up_full, w_down_full, correction_bias)

        # Higher tolerance (5e-2) needed for distributed EP: numerical variance from
        # (1) different reduction order (local expert GEMMs → all-reduce vs full GEMM)
        # (2) bfloat16 accumulation across multiple all-reduce operations
        torch.testing.assert_close(out_local.float(), out_full.float(), rtol=5e-2, atol=5e-2)
        print(f"PASS: 8-GPU EP (Kimi K2) [T={T}, E={E_global}, K={K}, H={H}, F={F}]")

    dist.barrier()
    dist.destroy_process_group()


@SharedDistributedFixture
def test_shared_fused_moe_ep_distributed(T, E_global, K, H, F, shared_F, world_size):
    """Multi-GPU EP test for SharedFusedMoE with shared expert.

    Verifies: sum(rank_routed_outputs) + shared_output == full_output
    """
    rank, actual_world_size = setup_distributed()

    if actual_world_size != world_size:
        pytest.skip(f"This test requires {world_size} GPUs, got {actual_world_size}")

    dtype = torch.bfloat16
    dev = f"cuda:{rank}"

    # All ranks use same seed
    torch.manual_seed(42)
    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E_global, dtype=dtype, device=dev)
    correction_bias = torch.randn(E_global, dtype=torch.float32, device=dev) * 0.1

    # Each rank owns E_local experts
    E_local = E_global // world_size
    assert E_global % world_size == 0

    # All ranks generate same full weights, then slice
    w_gate_up_full = torch.randn(E_global, F * 2, H, dtype=dtype, device=dev) * 0.02
    w_down_full = torch.randn(E_global, H, F, dtype=dtype, device=dev) * 0.02

    # Slice local weights
    start_expert = rank * E_local
    end_expert = (rank + 1) * E_local
    w_gate_up_local = w_gate_up_full[start_expert:end_expert].clone()
    w_down_local = w_down_full[start_expert:end_expert].clone()

    # Build expert_map
    expert_map = torch.full((E_global,), -1, dtype=torch.int32, device=dev)
    expert_map[start_expert:end_expert] = torch.arange(E_local, dtype=torch.int32, device=dev)

    # Create shared expert MLP (same on all ranks)
    shared_mlp = nn.Sequential(
        nn.Linear(H, shared_F, bias=False, dtype=dtype, device=dev),
        nn.SiLU(),
        nn.Linear(shared_F, H, bias=False, dtype=dtype, device=dev),
    )

    # TileOPs: local computation with expert_map + shared expert
    op_local = SharedFusedMoE(
        num_tokens=T, num_experts=E_global, top_k=K,
        hidden_size=H, ffn_size=F,
        scoring_func="sigmoid", renormalize=True, with_correction_bias=True,
        routed_scaling_factor=2.827,
        layout="nopad", dtype=dtype,
        expert_map=expert_map,
        shared_experts=shared_mlp,
    )
    shared_out, routed_out_local = op_local(hidden, gating, w_gate_up_local, w_down_local, correction_bias)

    # All-reduce routed output
    dist.all_reduce(routed_out_local, op=dist.ReduceOp.SUM)

    # Rank-0 computes full reference
    if rank == 0:
        op_full = SharedFusedMoE(
            num_tokens=T, num_experts=E_global, top_k=K,
            hidden_size=H, ffn_size=F,
            scoring_func="sigmoid", renormalize=True, with_correction_bias=True,
            routed_scaling_factor=2.827,
            layout="nopad", dtype=dtype,
            shared_experts=shared_mlp,
        )
        shared_out_full, routed_out_full = op_full(hidden, gating, w_gate_up_full, w_down_full, correction_bias)

        # Verify shared output (should be identical across ranks)
        torch.testing.assert_close(shared_out.float(), shared_out_full.float(), rtol=1e-5, atol=1e-5)

        # Verify routed output (after all-reduce)
        # Higher tolerance needed due to distributed EP numerical variance (see line 155 comment)
        torch.testing.assert_close(routed_out_local.float(), routed_out_full.float(), rtol=5e-2, atol=5e-2)

        print(f"PASS: 8-GPU EP SharedFusedMoE [T={T}, E={E_global}, K={K}, H={H}, F={F}, shared_F={shared_F}]")

    dist.barrier()
    dist.destroy_process_group()


@DistributedFixture
def test_fused_moe_vs_vllm_distributed(T, E_global, K, H, F, world_size):
    """Multi-GPU EP test: TileOPs (expert_map) vs vLLM (single-GPU baseline).

    TileOPs: uses expert_map local filtering, all-reduce routed output
    vLLM: single-GPU execution on rank-0 with full weights (baseline reference)
    Verifies: both produce same final output on rank-0
    """
    if not _VLLM_AVAILABLE:
        pytest.skip("vLLM not installed")

    rank, actual_world_size = setup_distributed()

    if actual_world_size != world_size:
        pytest.skip(f"This test requires {world_size} GPUs, got {actual_world_size}")

    dtype = torch.bfloat16
    dev = f"cuda:{rank}"

    # All ranks use same seed
    torch.manual_seed(42)
    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E_global, dtype=dtype, device=dev)
    correction_bias = torch.randn(E_global, dtype=torch.float32, device=dev) * 0.1

    # Each rank owns E_local experts
    E_local = E_global // world_size
    assert E_global % world_size == 0

    # All ranks generate same full weights, then slice
    w_gate_up_full = torch.randn(E_global, F * 2, H, dtype=dtype, device=dev) * 0.02
    w_down_full = torch.randn(E_global, H, F, dtype=dtype, device=dev) * 0.02

    # Slice local weights
    start_expert = rank * E_local
    end_expert = (rank + 1) * E_local
    w_gate_up_local = w_gate_up_full[start_expert:end_expert].clone()
    w_down_local = w_down_full[start_expert:end_expert].clone()

    # Build expert_map
    expert_map = torch.full((E_global,), -1, dtype=torch.int32, device=dev)
    expert_map[start_expert:end_expert] = torch.arange(E_local, dtype=torch.int32, device=dev)

    # Compute routing (same for both)
    from tileops.ops.moe import FusedTopKOp
    fk = FusedTopKOp(T, E_global, K, "sigmoid", True, with_correction_bias=True)
    topk_weights, topk_ids = fk(gating, correction_bias)

    # ========== TileOPs: expert_map local filtering ==========
    op_tileops = FusedMoe(
        num_tokens=T, num_experts=E_global, top_k=K,
        hidden_size=H, ffn_size=F,
        scoring_func="sigmoid", renormalize=True, with_correction_bias=True,
        routed_scaling_factor=2.827,
        layout="nopad", dtype=dtype,
        expert_map=expert_map,
    )
    out_tileops = op_tileops(hidden, gating, w_gate_up_local, w_down_local, correction_bias)
    dist.all_reduce(out_tileops, op=dist.ReduceOp.SUM)

    # ========== vLLM: use fused_moe with full weights on rank-0 only ==========
    if rank == 0:
        out_vllm = _vllm_fused_experts(
            hidden, w_gate_up_full, w_down_full,
            topk_weights, topk_ids,
            inplace=False,
        ) * 2.827

        # Higher tolerance needed due to distributed EP numerical variance (see line 155 comment)
        torch.testing.assert_close(out_tileops.float(), out_vllm.float(), rtol=5e-2, atol=5e-2)
        print(f"PASS: {world_size}-GPU EP TileOPs vs vLLM [T={T}, E={E_global}, K={K}, H={H}, F={F}]")

    dist.barrier()
    dist.destroy_process_group()

@DistributedFixture
def test_fused_moe_vs_vllm_ep_layer(T, E_global, K, H, F, world_size):
    """Multi-GPU EP test: TileOPs vs vLLM with expert_map simulation.

    Both use expert_map for local filtering (simulating EP without All-to-All).
    TileOPs: uses FusedMoe with expert_map
    vLLM: uses fused_experts with expert_map
    Verifies: both produce same output after all-reduce
    """
    if not _VLLM_AVAILABLE:
        pytest.skip("vLLM not available")

    rank, actual_world_size = setup_distributed()

    if actual_world_size != world_size:
        pytest.skip(f"This test requires {world_size} GPUs, got {actual_world_size}")

    dtype = torch.bfloat16
    dev = f"cuda:{rank}"

    # All ranks use same seed
    torch.manual_seed(42)
    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E_global, dtype=dtype, device=dev)
    correction_bias = torch.randn(E_global, dtype=torch.float32, device=dev) * 0.1

    # Each rank owns E_local experts
    E_local = E_global // world_size
    assert E_global % world_size == 0

    # All ranks generate same full weights, then slice
    w_gate_up_full = torch.randn(E_global, F * 2, H, dtype=dtype, device=dev) * 0.02
    w_down_full = torch.randn(E_global, H, F, dtype=dtype, device=dev) * 0.02

    # Slice local weights
    start_expert = rank * E_local
    end_expert = (rank + 1) * E_local
    w_gate_up_local = w_gate_up_full[start_expert:end_expert].clone()
    w_down_local = w_down_full[start_expert:end_expert].clone()

    # Build expert_map
    expert_map = torch.full((E_global,), -1, dtype=torch.int32, device=dev)
    expert_map[start_expert:end_expert] = torch.arange(E_local, dtype=torch.int32, device=dev)

    # Compute routing
    from tileops.ops.moe import FusedTopKOp
    fk = FusedTopKOp(T, E_global, K, "sigmoid", True, with_correction_bias=True)
    topk_weights, topk_ids = fk(gating, correction_bias)

    # ========== TileOPs: expert_map local filtering ==========
    op_tileops = FusedMoe(
        num_tokens=T, num_experts=E_global, top_k=K,
        hidden_size=H, ffn_size=F,
        scoring_func="sigmoid", renormalize=True, with_correction_bias=True,
        routed_scaling_factor=2.827,
        layout="nopad", dtype=dtype,
        expert_map=expert_map,
    )
    out_tileops = op_tileops(hidden, gating, w_gate_up_local, w_down_local, correction_bias)

    # ========== vLLM: fused_experts with expert_map ==========
    out_vllm = _vllm_fused_experts(
        hidden, w_gate_up_local, w_down_local,
        topk_weights, topk_ids,
        inplace=False,
        expert_map=expert_map,
        global_num_experts=E_global,
    ) * 2.827

    # All-reduce both outputs
    dist.all_reduce(out_tileops, op=dist.ReduceOp.SUM)
    dist.all_reduce(out_vllm, op=dist.ReduceOp.SUM)

    # Compare outputs
    # Higher tolerance needed due to distributed EP numerical variance (see line 155 comment)
    torch.testing.assert_close(out_tileops.float(), out_vllm.float(), rtol=5e-2, atol=5e-2)

    if rank == 0:
        print(f"PASS: {world_size}-GPU EP TileOPs vs vLLM (expert_map) [T={T}, E={E_global}, K={K}, H={H}, F={F}]")

    dist.barrier()
    dist.destroy_process_group()
