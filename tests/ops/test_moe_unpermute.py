"""Op-level tests for MoeUnpermuteOp (cutlass path).

Verifies:
  - output: correct weighted scatter-add of mm2_out rows
  - bf16 and fp16 input/output
  - K=1, K=2, K=8 (DeepSeek-V3 scale)
  - skewed distribution (all tokens to expert 0)
"""


import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.moe import MoeUnpermuteOp

# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _ref_moe_unpermute(
    mm2_out: torch.Tensor,
    inv_permuted_idx: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch reference for moe_unpermute.

    Args:
        mm2_out: [T*K, H] bf16/fp16
        inv_permuted_idx: [T*K] int32 — inv_permuted_idx[slot] = flat_idx (token*K + k)
        topk_weights: [T, K] float32

    Returns:
        output: [T, H] same dtype as mm2_out
    """
    numel, H = mm2_out.shape
    T, K = topk_weights.shape
    dtype = mm2_out.dtype

    output = torch.zeros(T, H, dtype=torch.float32, device=mm2_out.device)
    # kernel iterates: for token i, for k in [0,K): slot = i*K+k
    # perm_row = inv_permuted_idx[i*K+k], weight = topk_weights[i, k]
    for i in range(T):
        for k in range(K):
            slot = i * K + k
            perm_row = inv_permuted_idx[slot].item()
            w = topk_weights[i, k].item()
            output[i] += mm2_out[perm_row].float() * w

    return output.to(dtype)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


class MoeUnpermuteFixture(FixtureBase):
    PARAMS = [
        ("total_tokens, top_k, hidden_size, dtype", [
            pytest.param(4,    2,   64,  torch.bfloat16, marks=pytest.mark.smoke, id="tiny-bf16"),
            pytest.param(4,    2,   64,  torch.float16,  marks=pytest.mark.smoke, id="tiny-fp16"),
            pytest.param(16,   2,   128, torch.bfloat16, marks=pytest.mark.full,  id="small"),
            pytest.param(128,  4,   256, torch.bfloat16, marks=pytest.mark.full,  id="medium"),
            pytest.param(1024, 8,   128, torch.bfloat16, marks=pytest.mark.full,  id="qwen3-scale"),
            pytest.param(1,    2,   64,  torch.bfloat16, marks=pytest.mark.full,  id="single-token"),
            pytest.param(8,    1,   64,  torch.bfloat16, marks=pytest.mark.full,  id="top-k-1"),
            pytest.param(32,   4,   64,  torch.bfloat16, marks=pytest.mark.full,  id="skewed"),
        ]),
    ]


# ---------------------------------------------------------------------------
# TestBase subclass
# ---------------------------------------------------------------------------


class MoeUnpermuteTest(TestBase):

    def __init__(self, total_tokens, top_k, hidden_size, dtype):
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.dtype = dtype

    def gen_inputs(self):
        numel = self.total_tokens * self.top_k
        mm2_out = torch.randn(numel, self.hidden_size, dtype=self.dtype, device="cuda")
        # inv_permuted_idx: each slot maps to a flat_idx in [0, numel)
        # simulate a valid permutation: random shuffle of [0, numel)
        inv_permuted_idx = torch.randperm(numel, dtype=torch.int32, device="cuda")
        topk_weights = torch.rand(
            self.total_tokens, self.top_k, dtype=torch.float32, device="cuda"
        )
        return mm2_out, inv_permuted_idx, topk_weights

    def ref_program(self, mm2_out, inv_permuted_idx, topk_weights):
        return _ref_moe_unpermute(mm2_out, inv_permuted_idx, topk_weights)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@MoeUnpermuteFixture
def test_moe_unpermute_op(total_tokens, top_k, hidden_size, dtype):
    test = MoeUnpermuteTest(total_tokens, top_k, hidden_size, dtype)
    op = MoeUnpermuteOp(total_tokens, top_k, hidden_size, dtype)
    mm2_out, inv_permuted_idx, topk_weights = test.gen_inputs()

    output = op(mm2_out, inv_permuted_idx, topk_weights)
    output_ref = test.ref_program(mm2_out, inv_permuted_idx, topk_weights)

    # bf16/fp16 accumulation has rounding error; use loose atol
    atol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    assert torch.allclose(output.float(), output_ref.float(), atol=atol), (
        f"moe_unpermute mismatch: max_err={( output.float() - output_ref.float()).abs().max()}"
    )
    print(f"PASS [{total_tokens}tok, top{top_k}, H={hidden_size}, {dtype}]")


@pytest.mark.smoke
def test_moe_unpermute_skewed():
    """All tokens routed to expert 0 — inv_permuted_idx maps all slots to token 0."""
    T, K, H = 32, 4, 64
    numel = T * K
    mm2_out = torch.randn(numel, H, dtype=torch.bfloat16, device="cuda")
    # All flat_idx map to token 0: flat_idx in [0, K)
    inv_permuted_idx = torch.arange(numel, dtype=torch.int32, device="cuda") % K
    topk_weights = torch.rand(T, K, dtype=torch.float32, device="cuda")

    op = MoeUnpermuteOp(T, K, H, torch.bfloat16)
    output = op(mm2_out, inv_permuted_idx, topk_weights)
    output_ref = _ref_moe_unpermute(mm2_out, inv_permuted_idx, topk_weights)

    assert torch.allclose(output.float(), output_ref.float(), atol=1e-2), (
        f"skewed mismatch: max_err={(output.float() - output_ref.float()).abs().max()}"
    )
    print("PASS skewed (all slots → token 0)")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
