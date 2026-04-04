import torch

from workloads.base import WorkloadBase


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
