"""Distributed TP tests for SharedFusedMoE — shared expert Tensor Parallelism.

Compares TileOPs SharedFusedMoE (tp_size/tp_rank) against vLLM DeepseekV2MLP
(MergedColumnParallelLinear + RowParallelLinear) in a real multi-GPU TP setup.

Run with:
    torchrun --nproc_per_node=2 -m pytest tests/ops/test_moe_shared_fused_moe_distributed.py -m smoke -vvs
    torchrun --nproc_per_node=8 -m pytest tests/ops/test_moe_shared_fused_moe_distributed.py -m full -vvs

Note: These tests require multiple GPUs and will be skipped in single-GPU / CI environments.

Verifies:
  - TileOPs SharedFusedMoE TP partial outputs all-reduce to same result as
    vLLM DeepseekV2MLP (which handles TP internally via RowParallelLinear / MergedColumnParallelLinear)
"""

import contextlib
import functools
import os

import pytest
import torch
import torch.distributed as dist

from tileops.ops.moe import SharedFusedMoE

# Skip entire module if fewer than 2 GPUs
pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Distributed TP tests require at least 2 GPUs",
)

# vLLM optional imports
try:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        get_tensor_model_parallel_rank,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLP
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_vllm_distributed(tp_size: int) -> tuple[int, int]:
    """Initialize torch.distributed + vLLM model parallel groups.

    Returns (rank, world_size).
    Skips if not enough GPUs or vLLM unavailable.
    """
    if not _VLLM_AVAILABLE:
        pytest.skip("vLLM not installed")

    if not dist.is_available():
        pytest.skip("torch.distributed not available")

    if "RANK" not in os.environ:
        pytest.skip("Not launched via torchrun — distributed env vars not set")

    actual_gpus = torch.cuda.device_count()
    if actual_gpus < tp_size:
        pytest.skip(f"Test requires {tp_size} GPUs, got {actual_gpus}")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != tp_size:
        pytest.skip(f"Test requires exactly {tp_size} processes, got {world_size}")

    torch.cuda.set_device(rank)

    # Initialize vLLM model parallel groups (wraps the existing process group)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            local_rank=rank,
            distributed_init_method="env://",
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=tp_size)

    return rank, world_size


def _teardown_vllm_distributed():
    if _VLLM_AVAILABLE:
        with contextlib.suppress(Exception):
            cleanup_dist_env_and_memory()
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class TPFixture:
    """Fixture for shared expert TP tests."""
    PARAMS = [
        (
            "T, H, F_s, tp_size",
            [
                pytest.param(64,   128,  64,  2, marks=pytest.mark.smoke, id="smoke-tp2"),
                pytest.param(512,  7168, 18432, 2, marks=pytest.mark.full,  id="kimi-k2-tp2"),
                pytest.param(512,  7168, 18432, 8, marks=pytest.mark.full,  id="kimi-k2-tp8"),
            ],
        )
    ]

    def __call__(self, fn):
        params_name, params_list = self.PARAMS[0]

        @pytest.mark.parametrize(params_name, params_list)
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper


# ---------------------------------------------------------------------------
# Test: TileOPs SharedFusedMoE TP vs vLLM DeepseekV2MLP TP
# ---------------------------------------------------------------------------

@TPFixture()
def test_shared_expert_tp_vs_vllm(T, H, F_s, tp_size):
    """TileOPs shared expert TP vs vLLM DeepseekV2MLP TP.

    Both compute shared expert output using TP:
      - vLLM: MergedColumnParallelLinear + RowParallelLinear (internal all-reduce)
      - TileOPs: SharedFusedMoE(tp_size, tp_rank) → partial → manual all-reduce

    Verifies outputs match after vLLM all-reduce and TileOPs manual all-reduce.
    """
    rank, world_size = _setup_vllm_distributed(tp_size)
    dev = f"cuda:{rank}"
    dtype = torch.bfloat16

    try:
        # All ranks use same seed → same weights and inputs
        torch.manual_seed(42)
        hidden = torch.randn(T, H, dtype=dtype, device=dev)

        # Shared expert weights in TileOPs layout:
        #   shared_w_gate_up: [2*F_s, H]  (gate and up concatenated on dim=0)
        #   shared_w_down:    [H, F_s]
        shared_w_gate_up = torch.randn(F_s * 2, H, dtype=dtype, device=dev) * 0.02
        shared_w_down = torch.randn(H, F_s, dtype=dtype, device=dev) * 0.02

        # tp_rank available after _setup_vllm_distributed initializes vLLM parallel groups
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = F_s // tp_size

        # ── vLLM: DeepseekV2MLP (TP-aware, reduce_results=True) ──────────────────
        with set_current_vllm_config(VllmConfig()):
            vllm_mlp = DeepseekV2MLP(
                hidden_size=H,
                intermediate_size=F_s,
                hidden_act="silu",
                reduce_results=True,
                prefix="shared_experts",
            ).to(dtype=dtype, device=dev)

            # gate_up: ColumnParallel → each rank holds [2*shard, H]
            gate_up_shard = torch.cat([
                shared_w_gate_up[tp_rank * shard_size : (tp_rank + 1) * shard_size],
                shared_w_gate_up[F_s + tp_rank * shard_size : F_s + (tp_rank + 1) * shard_size],
            ], dim=0)  # [2*shard, H]
            # down: RowParallel → vLLM stores [H, shard] (input dim partitioned, no transpose)
            down_shard = shared_w_down[:, tp_rank * shard_size : (tp_rank + 1) * shard_size].contiguous()

            with torch.no_grad():
                vllm_mlp.gate_up_proj.weight.data.copy_(gate_up_shard.to(vllm_mlp.gate_up_proj.weight.dtype))
                vllm_mlp.down_proj.weight.data.copy_(down_shard.to(vllm_mlp.down_proj.weight.dtype))

            with torch.no_grad():
                vllm_out = vllm_mlp(hidden)  # [T, H] — fully reduced across TP ranks

        # ── TileOPs: SharedFusedMoE TP (partial → manual all-reduce) ─────────────
        # Minimal routed expert placeholders (not under test)
        E, K, F = 8, 2, 32
        gating = torch.randn(T, E, dtype=dtype, device=dev)
        w_gate_up = torch.randn(E, F * 2, H, dtype=dtype, device=dev) * 0.02
        w_down = torch.randn(E, H, F, dtype=dtype, device=dev) * 0.02

        op = SharedFusedMoE(
            num_tokens=T, num_experts=E, top_k=K,
            hidden_size=H, ffn_size=F,
            scoring_func="softmax", renormalize=False,
            layout="nopad", dtype=dtype,
            shared_ffn_size=F_s,
            tp_size=tp_size, tp_rank=tp_rank,
        )

        shared_partial, _ = op(
            hidden, gating, w_gate_up, w_down,
            shared_w_gate_up=shared_w_gate_up,
            shared_w_down=shared_w_down,
        )

        # Manual all-reduce across TP ranks
        tileops_out = shared_partial.clone()
        tp_group = get_tp_group().device_group
        dist.all_reduce(tileops_out, op=dist.ReduceOp.SUM, group=tp_group)

        # ── Compare ───────────────────────────────────────────────────────────────
        # bf16 kernel vs bf16 kernel: both go through same TP shard path → tight tolerance
        torch.testing.assert_close(tileops_out, vllm_out, rtol=1e-2, atol=1e-2)

        if rank == 0:
            print(f"PASS: SharedFusedMoE TP vs vLLM DeepseekV2MLP "
                  f"[T={T}, H={H}, F_s={F_s}, tp_size={tp_size}]")

        dist.barrier()
    finally:
        _teardown_vllm_distributed()
