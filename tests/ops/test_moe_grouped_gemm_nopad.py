"""Op-level tests for MoeGroupedGemmNopadFwdOp (tight, no-pad grouped GEMM).

Verifies that the tile-scheduled grouped GEMM produces the same per-expert
NT matmul as a pure-PyTorch reference, across:
  - uniform and skewed token-to-expert distributions
  - bf16 and fp16 activations
  - K block-aligned (TMA fast path) and K non-aligned (predicated path)
"""

import pytest
import torch

from tests.test_base import FixtureBase
from tileops.ops.moe import MoeGroupedGemmNopadFwdOp
from workloads.workload_base import WorkloadBase


def _make_sizes_offsets(numel: int, num_experts: int, distribution: str, device: str):
    """Build (true_sizes, true_offsets) for a fixed token-to-expert distribution.

    Args:
        numel: Total token-expert pairs (T * top_k).
        num_experts: Number of experts E.
        distribution: "uniform" — evenly split; "skewed" — most tokens on first
            20% of experts (one-token floor for the rest).
        device: CUDA device string.

    Returns:
        true_sizes [E] int32, true_offsets [E] int32.
    """
    if distribution == "uniform":
        base = max(1, numel // num_experts)
        sizes = torch.full((num_experts,), base, dtype=torch.int32, device=device)
        sizes[-1] = numel - base * (num_experts - 1)
    elif distribution == "skewed":
        sizes = torch.ones(num_experts, dtype=torch.int32, device=device)
        extra = numel - num_experts
        top_experts = max(1, num_experts // 5)
        per_top = extra // top_experts
        sizes[:top_experts] += per_top
        sizes[0] += extra - per_top * top_experts
    else:
        raise ValueError(f"unknown distribution: {distribution}")

    offsets = torch.zeros(num_experts, dtype=torch.int32, device=device)
    offsets[1:] = torch.cumsum(sizes[:-1], dim=0)
    assert int(sizes.sum().item()) == numel
    return sizes, offsets


def _ref_grouped_gemm_nopad(
    a: torch.Tensor,
    b: torch.Tensor,
    true_sizes: torch.Tensor,
    true_offsets: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch reference: per-expert NT matmul over the tight permute layout.

    Args:
        a: [numel, K] tight permuted activations.
        b: [num_experts, N, K] expert weights (NT: B^T applied).
        true_sizes: [E] int32 token count per expert.
        true_offsets: [E] int32 start offset per expert into a.

    Returns:
        c: [numel, N] reference output. Rows belonging to expert e are
            `a[off:off+size] @ b[e].T`; rows outside any expert range (none
            for valid inputs) are left zero.
    """
    numel, _ = a.shape
    num_experts, N, _ = b.shape
    c = torch.zeros(numel, N, dtype=a.dtype, device=a.device)
    sizes_l = true_sizes.tolist()
    offsets_l = true_offsets.tolist()
    for e in range(num_experts):
        size_e = sizes_l[e]
        if size_e == 0:
            continue
        off_e = offsets_l[e]
        a_e = a[off_e:off_e + size_e].to(torch.float32)
        b_e = b[e].to(torch.float32)
        c[off_e:off_e + size_e] = (a_e @ b_e.T).to(a.dtype)
    return c


class MoeGroupedGemmNopadTest(WorkloadBase):
    """Generates inputs for the tight grouped-GEMM op."""

    def __init__(self, numel, num_experts, n, k, distribution, dtype):
        self.numel = numel
        self.num_experts = num_experts
        self.n = n
        self.k = k
        self.distribution = distribution
        self.dtype = dtype

    def gen_inputs(self):
        torch.manual_seed(42)
        dev = "cuda"
        true_sizes, true_offsets = _make_sizes_offsets(
            self.numel, self.num_experts, self.distribution, dev
        )
        # Small scale keeps fp16 accumulation well within the parity tolerance.
        a = torch.randn(self.numel, self.k, dtype=self.dtype, device=dev) * 0.02
        b = torch.randn(self.num_experts, self.n, self.k, dtype=self.dtype, device=dev) * 0.02
        return a, b, true_sizes, true_offsets


class MoeGroupedGemmNopadFixture(FixtureBase):
    PARAMS = [
        ("numel, num_experts, n, k, distribution, dtype", [
            # K block_k-aligned (TMA fast path), uniform distribution, bf16.
            pytest.param(
                64, 4, 128, 64, "uniform", torch.bfloat16,
                marks=pytest.mark.smoke, id="aligned-uniform-bf16",
            ),
            # K block_k-aligned, skewed distribution, fp16 — covers dtype branch.
            pytest.param(
                64, 4, 128, 64, "skewed", torch.float16,
                marks=pytest.mark.smoke, id="aligned-skewed-fp16",
            ),
            # K NOT block_k-aligned (default block_k=64 → K=96 falls in predicated path).
            pytest.param(
                128, 8, 256, 96, "uniform", torch.bfloat16,
                marks=pytest.mark.full, id="unaligned-uniform-bf16",
            ),
        ]),
    ]


@MoeGroupedGemmNopadFixture
def test_moe_grouped_gemm_nopad_op(numel, num_experts, n, k, distribution, dtype):
    test = MoeGroupedGemmNopadTest(numel, num_experts, n, k, distribution, dtype)
    a, b, true_sizes, true_offsets = test.gen_inputs()

    op = MoeGroupedGemmNopadFwdOp(numel, num_experts, n, k, dtype=dtype)
    c = op(a, b, true_sizes, true_offsets)
    c_ref = _ref_grouped_gemm_nopad(a, b, true_sizes, true_offsets)

    assert c.shape == (numel, n), f"expected ({numel}, {n}), got {tuple(c.shape)}"
    assert c.dtype == dtype

    # bf16/fp16 accumulation in fp32 — match kernel's accum_dtype.
    torch.testing.assert_close(
        c.float(), c_ref.float(), atol=1e-2, rtol=1e-2,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
