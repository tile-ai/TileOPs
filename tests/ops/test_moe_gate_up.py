"""Op-level tests for MoeGateUpFwdOp.

Covers the op's contract — it validates its inputs, halves the GEMM width, and
matches a per-expert PyTorch reference — at both schedule regimes. Kernel-level
coverage of activations, partial tiles and empty experts lives in
tests/kernels/test_moe_grouped_gemm_3wg_fused_act.py.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase
from tileops.ops.moe.routed_expert.gate_up import (
    MoeGateUpFwdOp,
)
from workloads.moe import MoeGroupedGemmNopadWorkload


def _ref_fused_gate_up(a, b, true_sizes, true_offsets, ffn):
    """Per-expert NT matmul over the tight layout, then silu_and_mul."""
    c = torch.zeros(a.shape[0], ffn, dtype=a.dtype, device=a.device)
    sizes, offsets = true_sizes.tolist(), true_offsets.tolist()
    for e, (size_e, off_e) in enumerate(zip(sizes, offsets, strict=True)):
        if size_e == 0:
            continue
        gate_up = a[off_e:off_e + size_e].float() @ b[e].float().T
        c[off_e:off_e + size_e] = (
            F.silu(gate_up[:, :ffn]) * gate_up[:, ffn:]
        ).to(a.dtype)
    return c


class MoeGroupedGemmNopadFusedActFixture(FixtureBase):
    PARAMS = [
        ("numel, num_experts, ffn, k, distribution", [
            # 16 rows per expert — the short-group schedule.
            pytest.param(64, 4, 128, 128, "uniform",
                         marks=pytest.mark.smoke, id="short-group-uniform"),
            # 256 rows per expert — the default schedule, skewed routing.
            pytest.param(1024, 4, 128, 128, "skewed",
                         marks=pytest.mark.full, id="full-tiles-skewed"),
        ]),
    ]


@MoeGroupedGemmNopadFusedActFixture
def test_moe_gate_up_op(numel, num_experts, ffn, k, distribution):
    test = MoeGroupedGemmNopadWorkload(
        numel, num_experts, 2 * ffn, k, torch.bfloat16, distribution=distribution
    )
    a, b, true_sizes, true_offsets = test.gen_inputs()

    op = MoeGateUpFwdOp(numel, num_experts, ffn, k)
    c = op(a, b, true_sizes, true_offsets)

    assert c.shape == (numel, ffn), f"expected ({numel}, {ffn}), got {tuple(c.shape)}"
    assert c.dtype == torch.bfloat16
    torch.testing.assert_close(
        c.float(), _ref_fused_gate_up(a, b, true_sizes, true_offsets, ffn).float(),
        atol=2e-2, rtol=2e-2,
    )


@pytest.mark.smoke
def test_the_op_rejects_a_dtype_the_manifest_does_not_declare():
    """The synthesized validator only exists because the op is in the manifest."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device found.")
    numel, num_experts, ffn, k = 64, 4, 128, 128
    a = torch.randn(numel, k, dtype=torch.float32, device="cuda")
    b = torch.randn(num_experts, 2 * ffn, k, dtype=torch.float32, device="cuda")
    sizes = torch.full((num_experts,), numel // num_experts, dtype=torch.int32, device="cuda")
    offsets = torch.arange(num_experts, dtype=torch.int32, device="cuda") * (numel // num_experts)

    op = MoeGateUpFwdOp(numel, num_experts, ffn, k)
    with pytest.raises((ValueError, TypeError)):
        op(a, b, sizes, offsets)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
