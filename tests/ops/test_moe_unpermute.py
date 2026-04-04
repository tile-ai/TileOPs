"""Op-level tests for MoeUnpermuteOp (cutlass path).

Verifies:
  - output: correct weighted scatter-add using fwd_idx mapping
  - bf16 and fp16 input/output
  - K=1, K=2, K=8 (DeepSeek-V3 scale)
  - skewed distribution (all tokens to expert 0)

Interface note:
  MoeUnpermuteOp now accepts mm2_pad [padded_batch_sum, H] and
  fwd_idx [T*K] (forward mapping: flat_idx → padded slot).
  Tests use padded_batch_sum = T*K (no actual padding) for simplicity.
"""


import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.moe import MoeUnpermuteOp
from workloads.ops.moe_unpermute import MoeUnpermuteTest as _MoeUnpermuteTestWorkload


def _ref_moe_unpermute(
    mm2_pad: torch.Tensor,
    fwd_idx: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch reference for moe_unpermute."""
    _, H = mm2_pad.shape
    T, K = topk_weights.shape
    dtype = mm2_pad.dtype

    output = torch.zeros(T, H, dtype=torch.float32, device=mm2_pad.device)
    for i in range(T):
        for k in range(K):
            flat_idx = i * K + k
            padded_slot = fwd_idx[flat_idx].item()
            w = topk_weights[i, k].item()
            output[i] += mm2_pad[padded_slot].float() * w

    return output.to(dtype)


class MoeUnpermuteTest(_MoeUnpermuteTestWorkload, TestBase):
    def ref_program(self, mm2_pad, fwd_idx, topk_weights):
        return _ref_moe_unpermute(mm2_pad, fwd_idx, topk_weights)


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@MoeUnpermuteFixture
def test_moe_unpermute_op(total_tokens, top_k, hidden_size, dtype):
    test = MoeUnpermuteTest(total_tokens, top_k, hidden_size, dtype)
    op = MoeUnpermuteOp(total_tokens, top_k, hidden_size, dtype,
                        padded_batch_sum=test.padded_batch_sum)
    mm2_pad, fwd_idx, topk_weights = test.gen_inputs()

    output = op(mm2_pad, fwd_idx, topk_weights)
    output_ref = test.ref_program(mm2_pad, fwd_idx, topk_weights)

    rtol = 1.6e-2 if dtype == torch.bfloat16 else 1e-3
    atol = 1.6e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(output.float(), output_ref.float(), rtol=rtol, atol=atol)
    print(f"PASS [{total_tokens}tok, top{top_k}, H={hidden_size}, {dtype}]")


@pytest.mark.smoke
def test_moe_unpermute_skewed():
    """All tokens routed to expert 0 — fwd_idx maps all slots to first K padded positions."""
    T, K, H = 32, 4, 64
    numel = T * K
    mm2_pad = torch.randn(numel, H, dtype=torch.bfloat16, device="cuda")
    # All flat_idx map to padded_slot in [0, K): fwd_idx[i*K+k] = k
    fwd_idx = torch.arange(numel, dtype=torch.int32, device="cuda") % K
    topk_weights = torch.rand(T, K, dtype=torch.float32, device="cuda")

    op = MoeUnpermuteOp(T, K, H, torch.bfloat16, padded_batch_sum=numel)
    output = op(mm2_pad, fwd_idx, topk_weights)
    output_ref = _ref_moe_unpermute(mm2_pad, fwd_idx, topk_weights)

    assert torch.allclose(output.float(), output_ref.float(), atol=1e-2), (
        f"skewed mismatch: max_err={(output.float() - output_ref.float()).abs().max()}"
    )
    print("PASS skewed (all slots → first K padded positions)")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
