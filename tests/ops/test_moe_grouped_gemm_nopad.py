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
from workloads.moe import MoeGroupedGemmNopadWorkload


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
    test = MoeGroupedGemmNopadWorkload(
        numel, num_experts, n, k, dtype, distribution=distribution
    )
    a, b, true_sizes, true_offsets = test.gen_inputs()

    op = MoeGroupedGemmNopadFwdOp(numel, num_experts, n, k)
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
