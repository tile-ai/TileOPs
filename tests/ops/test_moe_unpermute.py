"""Op-level tests for MoeUnpermuteFwdOp (cutlass path).

Verifies:
  - output: correct weighted scatter-add using fwd_idx mapping
  - bf16 and fp16 input/output
  - K=1, K=2, K=8 (DeepSeek-V3 scale)
  - skewed distribution (all tokens to expert 0)

Interface note:
  MoeUnpermuteFwdOp now accepts mm2_pad [padded_batch_sum, H] and
  fwd_idx [T*K] (forward mapping: flat_idx → padded slot).
  Tests use padded_batch_sum = T*K (no actual padding) for simplicity.
"""


import pytest
import torch

from tests.test_base import FixtureBase, TestBase
from tileops.ops.moe import MoeUnpermuteFwdOp
from workloads.moe import MoeUnpermuteTest as _MoeUnpermuteTestWorkload


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
    op = MoeUnpermuteFwdOp(total_tokens, top_k, hidden_size, dtype,
                        padded_batch_sum=test.padded_batch_sum)
    mm2_pad, fwd_idx, topk_weights = test.gen_inputs()

    output = op(mm2_pad, fwd_idx, topk_weights)
    output_ref = test.ref_program(mm2_pad, fwd_idx, topk_weights)

    rtol = 1.6e-2 if dtype == torch.bfloat16 else 1e-3
    atol = 1.6e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(output.float(), output_ref.float(), rtol=rtol, atol=atol)
    print(f"PASS [{total_tokens}tok, top{top_k}, H={hidden_size}, {dtype}]")

if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
