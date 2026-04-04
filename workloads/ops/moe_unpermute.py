import torch

from workloads.base import WorkloadBase


def _ref_moe_unpermute(
    mm2_pad: torch.Tensor,
    fwd_idx: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch reference for moe_unpermute.

    Args:
        mm2_pad: [padded_batch_sum, H] bf16/fp16
        fwd_idx: [T*K] int32 — fwd_idx[flat_idx] = padded_slot
        topk_weights: [T, K] float32

    Returns:
        output: [T, H] same dtype as mm2_pad
    """
    _, H = mm2_pad.shape
    T, K = topk_weights.shape
    dtype = mm2_pad.dtype

    output = torch.zeros(T, H, dtype=torch.float32, device=mm2_pad.device)
    # kernel iterates: for token i, for k in [0,K): flat_idx = i*K+k
    # padded_slot = fwd_idx[flat_idx], weight = topk_weights[i, k]
    for i in range(T):
        for k in range(K):
            flat_idx = i * K + k
            padded_slot = fwd_idx[flat_idx].item()
            w = topk_weights[i, k].item()
            output[i] += mm2_pad[padded_slot].float() * w

    return output.to(dtype)

class MoeUnpermuteTest(WorkloadBase):

    def __init__(self, total_tokens, top_k, hidden_size, dtype):
        self.total_tokens = total_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.dtype = dtype
        # Use padded_batch_sum = T*K (no actual padding) for standalone tests.
        self.padded_batch_sum = total_tokens * top_k

    def gen_inputs(self):
        numel = self.total_tokens * self.top_k
        mm2_pad = torch.randn(numel, self.hidden_size, dtype=self.dtype, device="cuda")
        # fwd_idx: each flat_idx maps to a padded_slot in [0, padded_batch_sum)
        # simulate a valid mapping: random shuffle of [0, numel)
        fwd_idx = torch.randperm(numel, dtype=torch.int32, device="cuda")
        topk_weights = torch.rand(
            self.total_tokens, self.top_k, dtype=torch.float32, device="cuda"
        )
        return mm2_pad, fwd_idx, topk_weights
