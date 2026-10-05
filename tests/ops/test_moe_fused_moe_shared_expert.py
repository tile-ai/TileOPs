"""Tests for FusedMoESharedExpertFwdOp — FusedMoE with shared expert support.

Verifies:
  - FusedMoESharedExpertFwdOp returns (shared_output, routed_output) tuple
  - shared_output matches SharedExpertMLPKernel reference
  - routed_output matches FusedMoE output
  - Without the shared weights, shared_output is None
  - TP sharding: partial outputs sum to float32 math reference
"""

import pytest
import torch

from tests.test_base import TestBase
from tileops.kernels.gemm.dense import GemmTMAKernel
from tileops.kernels.gemm.persistent.template import GemmTemplate
from tileops.kernels.moe import SharedExpertMLPKernel
from tileops.ops.moe import FusedMoESharedExpertFwdOp, SharedExpertMLPFwdOp
from tileops.ops.moe.fused_moe import FusedMoEFwdOp
from tileops.utils import get_sm_version
from workloads.device import run_device
from workloads.moe import SharedExpertMLPWorkload, moe_call, moe_verification, ref_shared_expert
from workloads.numerics import compare_outputs


@pytest.mark.smoke
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_the_shared_expert_matches_its_reference(dtype):
    workload = SharedExpertMLPWorkload(
        moe_call("SharedExpertMLPFwdOp", {"D": dtype}, T=32, H=256, S=128)
    )
    inputs = workload.gen_inputs()
    TestBase.check(workload, SharedExpertMLPFwdOp(), *inputs)


@pytest.mark.in_tree_kernels
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize(
    "tokens, fused",
    [pytest.param(1024, True, id="fused-gate-up"), pytest.param(4096, False, id="separate")],
)
def test_the_shared_expert_runs_the_dense_template_past_its_threshold(tokens, fused):
    """A wide shared expert past ``template_min_m`` runs both GEMMs on the SM90 dense template."""
    if get_sm_version(torch.device(run_device()).index) != 90:
        pytest.skip("the dense template serves SM90")
    workload = SharedExpertMLPWorkload(
        moe_call("SharedExpertMLPFwdOp", {"D": "bfloat16"}, T=tokens, H=256, S=512)
    )
    op = SharedExpertMLPFwdOp()
    TestBase.check(workload, op, *workload.gen_inputs())
    (kernel,) = op.built_kernels("shared_expert_mlp").values()
    assert isinstance(kernel._gemm_gate_up, GemmTemplate)
    assert isinstance(kernel._gemm_down, GemmTemplate)
    assert kernel._gemm_gate_up.activation == ("silu_and_mul" if fused else "none")


@pytest.mark.in_tree_kernels
@pytest.mark.cuda_only
@pytest.mark.smoke
@pytest.mark.parametrize("num_tokens", [32, 512])
def test_fused_moe_shared_expert_basic(num_tokens):
    """FusedMoESharedExpertFwdOp with shared expert kernel."""
    torch.manual_seed(42)
    T, E, K, H, F, F_s = num_tokens, 8, 2, 64, 32, 16
    dtype = torch.bfloat16
    dev = "cuda"

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, device=dev)
    w_gate_up = torch.randn(E, F * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, F, dtype=dtype, device=dev) * 0.02
    shared_w_gate_up = torch.randn(F_s * 2, H, dtype=dtype, device=dev) * 0.02
    shared_w_down = torch.randn(H, F_s, dtype=dtype, device=dev) * 0.02

    op = FusedMoESharedExpertFwdOp(
        top_k=K,
        scoring_func="softmax",
        renormalize=False,
    )

    shared_out, routed_out = op(
        hidden,
        gating,
        w_gate_up,
        w_down,
        shared_w_gate_up=shared_w_gate_up,
        shared_w_down=shared_w_down,
    )

    assert shared_out.shape == (T, H)
    assert routed_out.shape == (T, H)
    assert shared_out.dtype == dtype
    assert routed_out.dtype == dtype

    shared_ref = ref_shared_expert(hidden, shared_w_gate_up, shared_w_down)
    compare_outputs(shared_out, shared_ref, moe_verification(1))

    if get_sm_version() == 90:
        shared_kernel = next(iter(op._shared_expert.built_kernels("shared_expert_mlp").values()))
        assert isinstance(shared_kernel._gemm_gate_up, GemmTMAKernel)
        assert isinstance(shared_kernel._gemm_down, GemmTMAKernel)
        if T == 512:
            wide = SharedExpertMLPKernel(512, 7168, 18432, dtype)
            assert isinstance(wide._gemm_gate_up, GemmTemplate)
            assert isinstance(wide._gemm_down, GemmTemplate)

    # routed_out matches FusedMoE
    op_routed = FusedMoEFwdOp(
        top_k=K,
        scoring_func="softmax",
        renormalize=False,
    )
    routed_ref = op_routed(hidden, gating, w_gate_up, w_down)
    assert torch.equal(routed_out, routed_ref)


@pytest.mark.smoke
def test_fused_moe_shared_expert_none():
    """Without the shared weights, shared_out is None."""
    torch.manual_seed(42)
    T, E, K, H, F = 16, 4, 2, 32, 16
    dtype = torch.bfloat16
    dev = run_device()

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, device=dev)
    w_gate_up = torch.randn(E, F * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, F, dtype=dtype, device=dev) * 0.02

    op = FusedMoESharedExpertFwdOp(
        top_k=K,
    )

    shared_out, routed_out = op(hidden, gating, w_gate_up, w_down)

    assert shared_out is None
    assert routed_out.shape == (T, H)


@pytest.mark.smoke
def test_fused_moe_shared_expert_tp():
    """TP sharding: sum of partial outputs matches float32 math reference.

    Uses float32 dtype to eliminate bf16 rounding differences between
    TP-sharded and single-GPU computation paths.
    """
    torch.manual_seed(42)
    T, E, K, H, F, F_s = 32, 8, 2, 64, 32, 16
    tp_size = 2
    dtype = torch.bfloat16
    dev = run_device()

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, device=dev)
    w_gate_up = torch.randn(E, F * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, F, dtype=dtype, device=dev) * 0.02
    shared_w_gate_up = torch.randn(F_s * 2, H, dtype=dtype, device=dev) * 0.02
    shared_w_down = torch.randn(H, F_s, dtype=dtype, device=dev) * 0.02

    # Float32 math reference: compute each TP shard's contribution manually
    # This matches exactly what each rank's kernel computes (column-parallel on gate_up, row-parallel on down)
    shard_size = F_s // tp_size
    partial_sum_ref = torch.zeros(T, H, dtype=torch.float32, device=dev)
    for tp_rank in range(tp_size):
        gate_up_shard = torch.cat(
            [
                shared_w_gate_up[tp_rank * shard_size : (tp_rank + 1) * shard_size],  # gate shard
                shared_w_gate_up[
                    F_s + tp_rank * shard_size : F_s + (tp_rank + 1) * shard_size
                ],  # up shard
            ],
            dim=0,
        )  # [2*shard, H]
        down_shard = shared_w_down[
            :, tp_rank * shard_size : (tp_rank + 1) * shard_size
        ]  # [H, shard]
        partial_sum_ref += ref_shared_expert(hidden.float(), gate_up_shard, down_shard)

    # routed reference (not affected by TP)
    op_routed = FusedMoEFwdOp(
        top_k=K,
        scoring_func="softmax",
        renormalize=False,
    )
    routed_ref = op_routed(hidden, gating, w_gate_up, w_down)

    # TP: accumulate partial outputs to simulate all-reduce
    partial_sum = torch.zeros(T, H, dtype=torch.float32, device=dev)
    for tp_rank in range(tp_size):
        op_tp = FusedMoESharedExpertFwdOp(
            top_k=K,
            scoring_func="softmax",
            renormalize=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        shared_partial, routed_out = op_tp(
            hidden,
            gating,
            w_gate_up,
            w_down,
            shared_w_gate_up=shared_w_gate_up,
            shared_w_down=shared_w_down,
        )
        assert shared_partial.shape == (T, H)
        partial_sum += shared_partial.float()

        # routed_out is not affected by TP sharding of shared expert
        assert torch.equal(routed_out, routed_ref)

    # partial_sum vs per-shard float32 math reference (same computation path)
    compare_outputs(partial_sum, partial_sum_ref, moe_verification(1))


@pytest.mark.smoke
def test_fused_moe_shared_expert_tp_rejects_local_shards():
    """TP contract: the op shards complete weights itself, so a TP-local shard paired with a
    complete weight disagrees on the shared width and is refused."""
    T, E, K, H, F, F_s, tp_size = 32, 8, 2, 64, 32, 16, 2
    dtype = torch.bfloat16
    dev = run_device()

    op = FusedMoESharedExpertFwdOp(
        top_k=K,
        scoring_func="softmax",
        renormalize=False,
        tp_size=tp_size,
        tp_rank=0,
    )

    hidden = torch.randn(T, H, dtype=dtype, device=dev)
    gating = torch.randn(T, E, device=dev)
    w_gate_up = torch.randn(E, F * 2, H, dtype=dtype, device=dev) * 0.02
    w_down = torch.randn(E, H, F, dtype=dtype, device=dev) * 0.02

    shard_size = F_s // tp_size

    # Pass TP-local gate_up shard instead of full weights → must raise
    bad_gate_up = torch.randn(2 * shard_size, H, dtype=dtype, device=dev)
    good_w_down = torch.randn(H, F_s, dtype=dtype, device=dev)
    with pytest.raises(ValueError, match="shared_w"):
        op(
            hidden,
            gating,
            w_gate_up,
            w_down,
            shared_w_gate_up=bad_gate_up,
            shared_w_down=good_w_down,
        )

    # Pass TP-local down shard instead of full weights → must raise
    good_gate_up = torch.randn(2 * F_s, H, dtype=dtype, device=dev)
    bad_w_down = torch.randn(H, shard_size, dtype=dtype, device=dev)
    with pytest.raises(ValueError, match="shared_w"):
        op(
            hidden,
            gating,
            w_gate_up,
            w_down,
            shared_w_gate_up=good_gate_up,
            shared_w_down=bad_w_down,
        )


@pytest.mark.in_tree_kernels
@pytest.mark.smoke
def test_a_replaced_shared_expert_kernel_is_the_one_built():
    """The shared half is reachable through kernel_map, like the routed half."""
    built = []

    class Replacement(SharedExpertMLPKernel):
        def __init__(self, **kwargs):
            built.append(kwargs)
            super().__init__(**kwargs)

    T, E, K, H, F, F_s = 32, 8, 2, 64, 32, 16
    op = FusedMoESharedExpertFwdOp(
        top_k=K,
        kernel_map={"shared_expert_mlp": Replacement},
    )
    assert op.kernel_map["shared_expert_mlp"] is Replacement

    dtype, dev = torch.bfloat16, run_device()
    torch.manual_seed(7)
    op(
        torch.randn(T, H, dtype=dtype, device=dev),
        torch.randn(T, E, device=dev),
        torch.randn(E, F * 2, H, dtype=dtype, device=dev) * 0.02,
        torch.randn(E, H, F, dtype=dtype, device=dev) * 0.02,
        shared_w_gate_up=torch.randn(F_s * 2, H, dtype=dtype, device=dev) * 0.02,
        shared_w_down=torch.randn(H, F_s, dtype=dtype, device=dev) * 0.02,
    )
    assert built, "the replacement was never constructed"
